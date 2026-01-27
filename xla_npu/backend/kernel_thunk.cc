#include "xla/xla_npu/backend/kernel_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"
#include "xla/backends/gpu/runtime/print_buffer_contents.h"
// #include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

#include "xla/xla_npu/backend/thunk.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/third_party/register/aclInterface.h"
#include "xla/xla_npu/third_party/register/aclnnInterface.h"
#include "xla/xla_npu/third_party/register/aclnnopInterface.h"

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;
using tsl::profiler::TraceMeLevel;

namespace xla {
namespace gpu {

absl::StatusOr<std::vector<aclTensor*>> KernelThunk::ExtractAclInputOutTensorPtr(
    const BufferAllocations* allocations, const std::vector<Shape>& input_param_shapes,
    const std::vector<Shape>& output_param_shapes) {
    std::vector<aclTensor*> acl_tensor_ptr_vector;

    for (int i = 0; i < input_param_shapes.size(); ++i) {
        TF_ASSIGN_OR_RETURN(auto input_param_mem, allocations->GetDeviceAddress(arguments_buffers_[i]));
        void* input_param_device_addr = input_param_mem.opaque();
        auto input_tensor_ptr = NpuTool::CreateAclTensorPtr(input_param_shapes[i], input_param_shapes[i].element_type(),
                                                            input_param_device_addr);
        acl_tensor_ptr_vector.push_back(input_tensor_ptr);

        // TODO: only for DEBUG
        // 只能执行前打印,因为算子执行会复用内存,执行结果会覆盖到入参地址上
        NpuLog::Print(NpuLog::Level::DEBUG, "---- Input param %d info: ----", i);
        NpuLog::Print(NpuLog::Level::DEBUG, "  Buffer allocation: %v", arguments_buffers_[i].ToString());
        NpuLog::Print(NpuLog::Level::DEBUG, "  Tensor Shape: %v", input_param_shapes[i].ToString(true));
        NpuLog::Print(NpuLog::Level::DEBUG, "  Device addr: %p", input_param_device_addr);
        NpuTool::PrintNpuDeviceMemoryValue<float>("  Value on device:", NpuTool::GetXlaShapeSize(input_param_shapes[i]),
                                                  input_param_device_addr);
    }

    for (int i = 0; i < output_param_shapes.size(); ++i) {
        TF_ASSIGN_OR_RETURN(auto output_param_mem, allocations->GetDeviceAddress(results_buffers_[i]));
        void* output_param_device_addr = output_param_mem.opaque();
        auto output_tensor_ptr = NpuTool::CreateAclTensorPtr(
            output_param_shapes[i], output_param_shapes[i].element_type(), output_param_device_addr);
        acl_tensor_ptr_vector.push_back(output_tensor_ptr);

        // TODO: only for DEBUG
        NpuLog::Print(NpuLog::Level::DEBUG, "---- output param %d info: ----", i);
        NpuLog::Print(NpuLog::Level::DEBUG, "  Buffer allocation: %v", results_buffers_[i].ToString());
        NpuLog::Print(NpuLog::Level::DEBUG, "  Tensor Shape: %v", output_param_shapes[i].ToString(true));
        NpuLog::Print(NpuLog::Level::DEBUG, "  Device addr: %p", output_param_device_addr);
        // 待执行后打印输出内存地址内容
    }

    return acl_tensor_ptr_vector;
}

absl::Status KernelThunk::ExecuteAclnn(aclrtStream& stream, uint64_t workspaceSize, aclOpExecutor* executor,
                                       AclnnExecFunc aclnnExecFunc) {
    aclnnStatus ret = 0;

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        NpuLog::Print(NpuLog::Level::DEBUG, "  Additional aclnn WorkspaceSize is %d, need to apply", workspaceSize);
        ret = npu::aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            NpuLog::Print(NpuLog::Level::ERROR, "Allocate workspace failed. Error code: %d", ret);
            return absl::InternalError("");
        }
    } else {
        NpuLog::Print(NpuLog::Level::DEBUG, "  Additional aclnn WorkspaceSize is 0, no need to apply");
    }
    // 调用aclnnExpand第二段接口
    ret = npu::aclnnExpand(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        NpuLog::Print(NpuLog::Level::ERROR, "Execute aclnnExpand failed. Error code: %d", ret);
        return absl::InternalError("");
    }

    ret = npu::aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        NpuLog::Print(NpuLog::Level::ERROR, "Synchronize stream failed. Error code: %d", ret);
        return absl::InternalError("");
    }

    // ret = npu::aclrtDestroyStream(&stream);
    // if (ret != ACL_SUCCESS) {
    //     NpuLog::Print(NpuLog::Level::WARN, "Destroy stream failed. Error code: %d", ret);
    // }

    return absl::OkStatus();
}

absl::Status KernelThunk::PrintOutputTensor(const BufferAllocations* allocations,
                                            const std::vector<Shape>& output_param_shapes) {
    for (int i = 0; i < output_param_shapes.size(); ++i) {
        TF_ASSIGN_OR_RETURN(auto output_param_mem, allocations->GetDeviceAddress(results_buffers_[i]));
        void* output_param_device_addr = output_param_mem.opaque();
        NpuTool::PrintNpuDeviceMemoryValue<float>(
            "  Value on device:", NpuTool::GetXlaShapeSize(output_param_shapes[i]), output_param_device_addr);
    }

    return absl::OkStatus();
}

KernelThunk::KernelThunk(NpuThunk::ThunkInfo thunk_info, std::string kernel_name,
                         const emitters::KernelArguments& kernel_arguments, LaunchDimensions launch_dimensions,
                         std::optional<se::ClusterDim> cluster_dim, int64_t shmem_bytes,
                         stream_executor::gpu::TmaMetadata tma_metadata)
    : NpuThunk(Kind::kKernel, std::move(thunk_info)),
      args_(kernel_arguments.GetArgumentBufferSlices()),
      written_(kernel_arguments.GetArgumentOutputFlags()),
      kernel_name_(std::move(kernel_name)),
      launch_dimensions_(std::move(launch_dimensions)),
      cluster_dim_(std::move(cluster_dim)),
      shmem_bytes_(shmem_bytes),
      tma_metadata_(std::move(tma_metadata)) {}

std::string KernelThunk::ToString(int indent) const {
    return absl::StrFormat(", kernel = %s, launch dimensions = %s, cluster_dim = %s", kernel_name_,
                           launch_dimensions_.ToString(),
                           cluster_dim_.has_value() ? cluster_dim_->ToString() : "nullopt");
}

absl::StatusOr<ThunkProto> KernelThunk::ToProto() const {
    ThunkProto proto;
    *proto.mutable_thunk_info() = thunk_info().ToProto();

    auto* kernel_proto = proto.mutable_kernel_thunk();
    for (const auto& arg : args_) {
        TF_ASSIGN_OR_RETURN(*kernel_proto->add_args(), arg.ToProto());
    }
    for (bool written : written_) {
        kernel_proto->add_written(written);
    }
    kernel_proto->set_kernel_name(kernel_name_);
    *kernel_proto->mutable_launch_dimensions() = launch_dimensions_.ToProto();
    if (cluster_dim_) {
        *kernel_proto->mutable_cluster_dim() = cluster_dim_->ToProto();
    }
    kernel_proto->set_shmem_bytes(shmem_bytes_);
    *kernel_proto->mutable_tma_metadata() = tma_metadata_.ToProto();
    return proto;
}

absl::StatusOr<std::unique_ptr<KernelThunk>> KernelThunk::FromProto(
    ThunkInfo thunk_info, const KernelThunkProto& proto, absl::Span<const BufferAllocation> buffer_allocations) {
    TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions, LaunchDimensions::FromProto(proto.launch_dimensions()));
    std::optional<stream_executor::ClusterDim> cluster_dim;
    if (proto.has_cluster_dim()) {
        TF_ASSIGN_OR_RETURN(cluster_dim.emplace(), stream_executor::ClusterDim::FromProto(proto.cluster_dim()));
    }

    if (proto.written().size() != proto.args().size()) {
        return absl::InvalidArgumentError("Proto fields `written` and `args` need to have the same cardinality.");
    }

    std::vector<emitters::KernelArgument> arguments;
    arguments.reserve(proto.args().size());
    for (int i = 0; i < proto.args().size(); ++i) {
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                            BufferAllocation::Slice::FromProto(proto.args().at(i), buffer_allocations));
        emitters::KernelArgument argument{Shape{}, slice};
        argument.set_written(proto.written().at(i));
        arguments.push_back(std::move(argument));
    }

    TF_ASSIGN_OR_RETURN(stream_executor::gpu::TmaMetadata tma_metadata,
                        stream_executor::gpu::TmaMetadata::FromProto(proto.tma_metadata()));

    return std::make_unique<KernelThunk>(thunk_info, proto.kernel_name(),
                                         emitters::KernelArguments(std::move(arguments)), launch_dimensions,
                                         cluster_dim, proto.shmem_bytes(), tma_metadata);
}

absl::Status KernelThunk::Initialize(const InitializeParams& params) {
    absl::MutexLock lock(mutex_);

    // // Load the kernel into the device if necessary.
    // //
    // // We could alternatively do this within ExecuteOnStream, but doing it here
    // // lets the time spent loading the kernel not count towards our execution
    // // profiles.
    // if (!kernel_cache_.contains(params.executor)) {
    //   std::unique_ptr<se::Kernel> kernel;
    //   if (!params.src.binary.empty()) {
    //     TF_ASSIGN_OR_RETURN(
    //         kernel, CreateKernel(kernel_name_, args_.size(), params.src.binary,
    //                              params.executor, shmem_bytes_));

    //   } else {
    //     TF_ASSIGN_OR_RETURN(
    //         kernel, CreateKernel(kernel_name_, args_.size(), params.src.text,
    //                              params.executor, shmem_bytes_));
    //   }

    //   kernel_cache_.emplace(params.executor, std::move(kernel));
}

return absl::OkStatus();
}

absl::Status KernelThunk::ExecuteOnStream(const ExecuteParams& params) {
    TraceMe trace([] { return TraceMeEncode("KernelThunk::ExecuteOnStream", {}); },
                  /*level=*/TraceMeLevel::kVerbose);

    NpuLog::Print(NpuLog::Level::DEBUG,
                  "==== Launch host kernel '%s' with %d arguments and %d results: #workgroups=%v ====", kernel_name_,
                  arguments_buffers_.size(), results_buffers_.size(), num_workgroups_);
    NpuLog::Print(NpuLog::Level::DEBUG, "---- Corresponding hlo instruction: %s ----", thunk_info().hloInstrDesc);

    if (params.stream == nullptr) {
        NpuLog::Print(NpuLog::Level::ERROR, "Stream need to be created or init before execute npu thunk.");
        return absl::InternalError("");
    }
    auto stream_handle = params.stream->platform_specific_handle();
    if (stream_handle.stream == nullptr) {
        NpuLog::Print(NpuLog::Level::ERROR, "Stream need to be created or init before execute npu thunk.");
        return absl::InternalError("");
    }
    auto stream = stream_handle.stream;

    const BufferAllocations* allocations = params.buffer_allocations;
    const auto& input_param_shapes = thunk_info().input_param_shapes;
    const auto& output_param_shapes = thunk_info().output_param_shapes;
    aclnnStatus ret = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto exec_ret = absl::OkStatus();

    if (!CheckInputOuputParamsSize(thunk_info().op_code, input_param_shapes, output_param_shapes)) {
        return absl::InternalError("");
    }

    TF_ASSIGN_OR_RETURN(auto acl_tensor_ptr_vector,
                        ExtractAclInputOutTensorPtr(allocations, input_param_shapes, output_param_shapes));

    switch (thunk_info().op_code) {
        case HloOpcode::kBroadcast: {
            std::vector<int64_t> output_param_size_data =
                NpuTool::ConvertXlaShape2VectorShape(output_param_shapes.front());
            aclIntArray* output_param_size =
                npu::aclCreateIntArray(output_param_size_data.data(), output_param_size_data.size());

            // 调用aclnnExpand第一段接口
            ret = npu::aclnnExpandGetWorkspaceSize(acl_tensor_ptr_vector[0], output_param_size,
                                                   acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                NpuLog::Print(NpuLog::Level::ERROR, "Call aclnnExpandGetWorkspaceSize failed. Error code: %d", ret);
                return absl::InternalError("");
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnExpand);
            break;
        }

        case HloOpcode::kAdd: {
            float alphaValue = 1.0;
            aclScalar* alpha = npu::aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

            // 调用aclnnAdd第一段接口
            ret = npu::aclnnAddGetWorkspaceSize(acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1], alpha,
                                                acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                NpuLog::Print(NpuLog::Level::ERROR, "Call aclnnAddGetWorkspaceSize failed. Error code: %d", ret);
                return absl::InternalError("");
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnAdd);
            break;
        }

        case HloOpcode::kTanh: {
            ret = npu::aclnnTanhGetWorkspaceSize(acl_tensor_ptr_vector[0], acl_tensor_ptr_vector.back(), &workspaceSize,
                                                 &executor);
            if (ret != ACL_SUCCESS) {
                NpuLog::Print(NpuLog::Level::ERROR, "Call aclnnTanGetWorkspaceSize failed. Error code: %d", ret);
                return absl::InternalError("");
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnTanh);
            break;
        }

        case HloOpcode::kMultiply: {
            ret = npu::aclnnMulGetWorkspaceSize(acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
                                                acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                NpuLog::Print(NpuLog::Level::ERROR, "Call aclnnMulGetWorkspaceSize failed. Error code: %d", ret);
                return absl::InternalError("");
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnMul);
            break;
        }

        default:
            NpuLog::Print(NpuLog::Level::ERROR, "Op code(%v) does not support execute now",
                          HloOpcodeString(info().op_code));
            return absl::InternalError("");
    }

    PrintOutputTensor(allocations, output_param_shapes);

    return exec_ret;

    // // Load the kernel.
    // se::StreamExecutor* executor = params.stream->parent();
    // se::Kernel* kernel = nullptr;

    // se::Stream* stream = nullptr;
    // {
    //     TraceMe trace([] { return TraceMeEncode("KernelThunk::ExecuteOnStream/GetStreamForExecution", {}); },
    //                   /*level=*/TraceMeLevel::kVerbose);
    //     TF_ASSIGN_OR_RETURN(stream, GetStreamForExecution(NpuThunk::execution_stream_id(), params));
    // }

    // {
    //     TraceMe trace([] { return TraceMeEncode("KernelThunk::ExecuteOnStream/mutex", {}); },
    //                   /*level=*/TraceMeLevel::kVerbose);
    //     absl::MutexLock lock(mutex_);
    //     TraceMe trace_find([] { return TraceMeEncode("KernelThunk::ExecuteOnStream/mutex/find", {}); },
    //                        /*level=*/TraceMeLevel::kVerbose);
    //     auto it = kernel_cache_.find(executor);
    //     CHECK(it != kernel_cache_.end()) << "Initialize() not called for StreamExecutor " << executor;
    //     kernel = it->second.get();
    // }

    // absl::InlinedVector<se::KernelArgument, 4> kernel_args;
    // {
    //     TraceMe trace([] { return TraceMeEncode("KernelThunk::ExecuteOnStream/kernel_args", {}); },
    //                   /*level=*/TraceMeLevel::kVerbose);
    //     int device_ordinal = executor->device_ordinal();
    //     VLOG(3) << "[" << device_ordinal << "] Launching " << kernel->name();
    //     for (const auto& [idx, arg] : llvm::enumerate(args_)) {
    //         se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
    //         VLOG(3) << "[" << device_ordinal << "] Arg: alloc #" << arg.index() << ", offset: " << arg.offset() << ":
    //         "
    //                 << buf.opaque() << " (" << buf.size() << "B)";

    //         if (auto it = tma_metadata_.arg_index_to_tma_info.find(idx);
    //             it != tma_metadata_.arg_index_to_tma_info.end()) {
    //             // TMA descriptor argument.
    //             const se::gpu::TmaDescriptor& tma_desc = it->second;
    //             TF_ASSIGN_OR_RETURN(se::TensorMap tensor_map, executor->CreateTensorMap(tma_desc, buf.opaque()));
    //             VLOG(3) << "[" << device_ordinal << "]  Using TensorMap for arg #" << idx << ": "
    //                     << tma_desc.ToString();
    //             kernel_args.push_back(std::move(tensor_map));
    //         } else {
    //             // Buffer argument.
    //             kernel_args.push_back(buf);
    //         }
    //     }
    // }

    // if (VLOG_IS_ON(100)) {
    //     PrintBufferContents(stream, kernel_args);
    // }

    // return ExecuteKernelOnStream(*kernel, absl::Span<se::KernelArgument>(kernel_args.data(), kernel_args.size()),
    //                              launch_dimensions_, cluster_dim_, stream);
}

NpuThunk::BufferUses KernelThunk::buffer_uses() const {
    NpuThunk::BufferUses buffers;
    buffers.reserve(args_.size());
    for (int i = 0; i < args_.size(); ++i) {
        // We assume that any buffer is either an input or an output of the
        // kernel, and inout buffers are represented as 2 separate arguments.
        if (written_[i]) {
            buffers.push_back(BufferUse::Write(args_[i]));
        } else {
            buffers.push_back(BufferUse::Read(args_[i]));
        }
    }
    return buffers;
}

}  // namespace gpu
}  // namespace xla
