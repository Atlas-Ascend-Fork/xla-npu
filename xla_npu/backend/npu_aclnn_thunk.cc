#include "xla/xla_npu/backend/npu_aclnn_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tsl/profiler/lib/nvtx_utils.h"
// #include "xla/backends/gpu/runtime/cudnn_thunk.h"
// #include "xla/backends/gpu/runtime/thunk.h"
// #include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/npu_thunk.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/service/npu_buffer_allocations.h"
#include "xla/xla_npu/third_party/register/aclInterface.h"
#include "xla/xla_npu/third_party/register/aclnnInterface.h"
#include "xla/xla_npu/third_party/register/aclnnopInterface.h"

namespace xla {
namespace gpu {

static bool CheckInputOuputParamsSize(HloOpcode op_code, const std::vector<Shape>& param_shapes) {
    switch (op_code) {
        case HloOpcode::kBroadcast:
        case HloOpcode::kTanh:
            if (param_shapes.size() != 2) {
                NpuLog::Print(NpuLog::Level::ERROR, "Param shape size %d does not meet target value 2",
                              param_shapes.size());
                return false;
            }
            break;
        case HloOpcode::kAdd:
        case HloOpcode::kMultiply:
            if (param_shapes.size() != 3) {
                NpuLog::Print(NpuLog::Level::ERROR, "Param shape size %d does not meet target value 3",
                              param_shapes.size());
                return false;
            }
            break;
        default:
            NpuLog::Print(NpuLog::Level::WARN, "Op code(%v) does not support check now", HloOpcodeString(op_code));
    }

    return true;
}

NpuAclnnThunk::NpuAclnnThunk(std::string fingerprint, ThunkInfo thunk_info, std::vector<BufferAllocation::Slice> args,
                             std::vector<bool> output_args, std::optional<int64_t> sdpa_dropout_seed)
    : NpuThunk(Kind::kCuDnn, std::move(thunk_info)),
      fingerprint_(std::move(fingerprint)),
      graph_(std::make_shared<se::dnn::LazyDnnGraph>(nullptr)),
      args_(std::move(args)),
      output_args_(std::move(output_args)),
      sdpa_dropout_seed_(sdpa_dropout_seed) {}

absl::Status NpuAclnnThunk::Initialize(const InitializeParams& params) {
    absl::Status ret = absl::OkStatus();
    // // Calling AsDnn outside call_once ensures that cuDNN handles get created for
    // // all GPUs in programs using cuDNN during the executable initialization
    // // phase. It's sufficient to deserialize the graph once using just one of
    // // them.
    // se::dnn::DnnSupport* dnn = params.stream->parent()->AsDnn();
    // absl::call_once(once_flag_, [&] {
    //     auto result = dnn->DeserializeGraph(*params.stream, params.src.dnn_compiled_graphs.at(fingerprint_));
    //     std::string().swap(fingerprint_);
    //     if (result.ok()) {
    //         graph_->swap(*result);
    //         if (sdpa_dropout_seed_.has_value()) {
    //             graph_->get()->InitDropoutState(params.local_device_count, *sdpa_dropout_seed_, 16);
    //         }
    //     }
    //     ret = result.status();
    // });
    return ret;
}

absl::Status NpuAclnnThunk::ExecuteOnStream(const ExecuteParams& params) {
    NpuLog::Print(NpuLog::Level::DEBUG, "==== Launch aclnn kernel with %d param ====", args_.size());
    NpuLog::Print(NpuLog::Level::DEBUG, "---- Corresponding hlo instruction: %s ----", thunk_info().hloInstrDesc);

    InitializeParams initialize_params;
    TF_RETURN_IF_ERROR(Initialize(initialize_params));  // TODO 排查外部是否有额外调用

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

    const NpuBufferAllocations* allocations = params.buffer_allocations;
    const auto& param_shapes = thunk_info().param_shapes;
    aclnnStatus ret = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto exec_ret = absl::OkStatus();

    if (!CheckInputOuputParamsSize(thunk_info().op_code, param_shapes)) {
        return absl::InternalError("");
    }

    TF_ASSIGN_OR_RETURN(auto acl_tensor_ptr_vector, ExtractAclInputOutTensorPtr(allocations, param_shapes));

    switch (thunk_info().op_code) {
        case HloOpcode::kBroadcast: {
            std::vector<int64_t> output_param_size_data = NpuTool::ConvertXlaShape2VectorShape(param_shapes.back());
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
                          HloOpcodeString(thunk_info().op_code));
            return absl::InternalError("");
    }

    PrintOutputTensor(allocations, param_shapes);

    return exec_ret;

    // std::vector<se::DeviceMemoryBase> buffer_args;
    // buffer_args.reserve(args_.size());
    // for (const BufferAllocation::Slice& arg : args_) {
    //     auto addr = params.buffer_allocations->GetDeviceAddress(arg);
    //     if (output_args_[buffer_args.size()]) {
    //         tsl::profiler::MarkMemoryInitialized(
    //             addr.opaque(), addr.size(),
    //             static_cast<tsl::profiler::StreamHandle>(params.stream->platform_specific_handle().stream));
    //     }
    //     buffer_args.push_back(addr);
    // }
    // return graph_->get()->Execute(*params.stream, absl::Span<se::DeviceMemoryBase>(buffer_args),
    //                               params.collective_params->local_device_ordinal);
}

// absl::StatusOr<ThunkProto> NpuAclnnThunk::ToProto() const {
//     ThunkProto proto;
//     *proto.mutable_thunk_info() = thunk_info().ToProto();
//     proto.mutable_cudnn_thunk()->set_fingerprint(fingerprint_);

//     for (const BufferAllocation::Slice& arg : args_) {
//         TF_ASSIGN_OR_RETURN(*proto.mutable_cudnn_thunk()->add_args(), arg.ToProto());
//     }
//     for (const bool is_output : output_args_) {
//         proto.mutable_cudnn_thunk()->add_output_args(is_output);
//     }
//     if (sdpa_dropout_seed_.has_value()) {
//         proto.mutable_cudnn_thunk()->set_sdpa_dropout_seed(static_cast<int64_t>(*sdpa_dropout_seed_));
//     }
//     return proto;
// }

// absl::StatusOr<std::unique_ptr<NpuAclnnThunk>> NpuAclnnThunk::FromProto(
//     ThunkInfo thunk_info, const CudnnThunkProto& proto, absl::Span<const BufferAllocation> buffer_allocations) {
//     std::vector<BufferAllocation::Slice> args;
//     args.reserve(proto.args_size());
//     for (const buffer_assignment::NpuBufferAllocationsliceProto& arg : proto.args()) {
//         TF_ASSIGN_OR_RETURN(args.emplace_back(), BufferAllocation::Slice::FromProto(arg, buffer_allocations));
//     }
//     std::vector<bool> output_args;
//     output_args.reserve(proto.output_args_size());
//     for (const bool output_arg : proto.output_args()) {
//         output_args.push_back(output_arg);
//     }
//     std::optional<uint64_t> sdpa_dropout_seed;
//     if (proto.has_sdpa_dropout_seed()) {
//         sdpa_dropout_seed = static_cast<uint64_t>(proto.sdpa_dropout_seed());
//     }
//     return std::make_unique<NpuAclnnThunk>(proto.fingerprint(), std::move(thunk_info), std::move(args),
//                                            std::move(output_args), sdpa_dropout_seed);
// }

absl::StatusOr<std::vector<aclTensor*>> NpuAclnnThunk::ExtractAclInputOutTensorPtr(
    const NpuBufferAllocations* allocations, const std::vector<Shape>& param_shapes) {
    std::vector<aclTensor*> acl_tensor_ptr_vector;

    for (int i = 0; i < args_.size(); ++i) {
        auto addr = allocations->GetDeviceAddress(args_[i]);
        void* device_addr = addr.opaque();
        auto tensor_ptr = NpuTool::CreateAclTensorPtr(param_shapes[i], param_shapes[i].element_type(), device_addr);
        acl_tensor_ptr_vector.push_back(tensor_ptr);
        if (!output_args_[i]) {
            // 只能执行前打印,因为算子执行会复用内存,执行结果会覆盖到入参地址上
            NpuLog::Print(NpuLog::Level::DEBUG, "---- Param(input) %d info: ----", i);
            NpuTool::PrintNpuDeviceMemoryValue<float>("  Value on device:", NpuTool::GetXlaShapeSize(param_shapes[i]),
                                                      device_addr);
        } else {
            NpuLog::Print(NpuLog::Level::DEBUG, "---- Param(output) %d info: ----", i);
            // 待执行后打印输出内存地址内容
        }
        NpuLog::Print(NpuLog::Level::DEBUG, "  Buffer allocation: %v", args_[i].ToString());
        NpuLog::Print(NpuLog::Level::DEBUG, "  Tensor Shape: %v", param_shapes[i].ToString(true));
        NpuLog::Print(NpuLog::Level::DEBUG, "  Device addr: %p", device_addr);
    }

    return acl_tensor_ptr_vector;
}

absl::Status NpuAclnnThunk::ExecuteAclnn(aclrtStream& stream, uint64_t workspaceSize, aclOpExecutor* executor,
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

absl::Status NpuAclnnThunk::PrintOutputTensor(const NpuBufferAllocations* allocations,
                                              const std::vector<Shape>& param_shapes) {
    for (int i = 0; i < args_.size(); ++i) {
        if (output_args_[i]) {
            auto addr = allocations->GetDeviceAddress(args_[i]);
            void* device_addr = addr.opaque();
            NpuTool::PrintNpuDeviceMemoryValue<float>("  Value on device:", NpuTool::GetXlaShapeSize(param_shapes[i]),
                                                      device_addr);
        }
    }

    return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
