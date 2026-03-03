#include "xla/xla_npu/backend/runtime/npu_aclnn_thunk.h"

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
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/header/acl_nn_op/aclnn_matmul.h"
#include "xla/xla_npu/dependency/header/acl_nn_op/aclnn_cat.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_op_interface.h"
#include "xla/xla_npu/service/npu_buffer_allocations.h"

namespace xla {

static bool CheckInputOuputParamsSize(HloOpcode op_code, const std::vector<Shape>& param_shapes) {
    switch (op_code) {
        case HloOpcode::kBroadcast:
        case HloOpcode::kTanh:
        case HloOpcode::kReshape:
        case HloOpcode::kNegate:
        case HloOpcode::kExp:
            if (param_shapes.size() != 2) {
                NPU_LOG_FORMAT_PRINT(ERROR, "Param shape size %d does not meet target value 2", param_shapes.size());
                return false;
            }
            break;
        case HloOpcode::kAdd:
        case HloOpcode::kMultiply:
        case HloOpcode::kDot:
        case HloOpcode::kDivide:
        case HloOpcode::kSubtract:
        case HloOpcode::kMaximum:
            if (param_shapes.size() != 3) {
                NPU_LOG_FORMAT_PRINT(ERROR, "Param shape size %d does not meet target value 3", param_shapes.size());
                return false;
            }
            break;
        case HloOpcode::kConcatenate:
            // Concatenate 是变长参数算子,至少需要 2 个输入 + 1 个输出 = 至少 3 个参数
            if (param_shapes.size() < 3) {
                NPU_LOG_FORMAT_PRINT(ERROR, "Concatenate param shape size %d is less than 3", param_shapes.size());
                return false;
            }
            break;
        case HloOpcode::kReduce:
            // Reduce 操作: 输入 + 初始值 + 输出 = 3 个参数
            if (param_shapes.size() != 3) {
                NPU_LOG_FORMAT_PRINT(ERROR, "Reduce param shape size %d does not meet target value 3", param_shapes.size());
                return false;
            }
            break;
        default:
            NPU_LOG_FORMAT_PRINT(WARN, "Op code(%v) does not support check now", HloOpcodeString(op_code));
    }

    return true;
}

NpuAclnnThunk::NpuAclnnThunk(std::string fingerprint, ThunkInfo thunk_info, std::vector<BufferAllocation::Slice> args,
                             std::vector<bool> output_args, std::optional<int64_t> sdpa_dropout_seed)
    : NpuThunk(Kind::kCuDnn, std::move(thunk_info)),
      fingerprint_(std::move(fingerprint)),
      graph_(std::make_shared<se::dnn::LazyDnnGraph>(nullptr)),
      args_(std::move(args)),
      //   output_args_(std::move(output_args)), // output_args_表示args_中哪些内存地址是会被写入的,由于存在内存复用,
      // 假设add的args_大小为3,则可能后2个arg内存地址相同,都为可写入
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
    NPU_LOG_FORMAT_PRINT(DEBUG, "==== Launch aclnn kernel with %d param ====", args_.size());
    NPU_LOG_FORMAT_PRINT(DEBUG, "---- Corresponding hlo instruction: %s ----", thunk_info().hloInstrDesc);

    InitializeParams initialize_params;
    TF_RETURN_IF_ERROR(Initialize(initialize_params));  // TODO 排查外部是否有额外调用

    if (params.stream == nullptr) {
        NPU_LOG_FORMAT_PRINT(ERROR, "Stream need to be created or init before execute npu thunk.");
        return absl::InternalError("");
    }
    auto stream_handle = params.stream->platform_specific_handle();
    if (stream_handle.stream == nullptr) {
        NPU_LOG_FORMAT_PRINT(ERROR, "Stream need to be created or init before execute npu thunk.");
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
        case HloOpcode::kReshape: {
            // Reshape operation: copy data from input to output using aclrtMemcpy
            // Even if shapes are different, the underlying data size is the same
            NPU_LOG_FORMAT_PRINT(DEBUG, "Reshape operation: using aclrtMemcpy to copy data");

            // Get input and output buffer addresses
            auto input_addr = allocations->GetDeviceAddress(args_[0]);
            auto output_addr = allocations->GetDeviceAddress(args_.back());
            void* input_device_addr = input_addr.opaque();
            void* output_device_addr = output_addr.opaque();

            // Calculate data size in bytes
            int64_t input_size = NpuTool::GetXlaShapeSize(param_shapes[0]) *
                                 ShapeUtil::ByteSizeOfPrimitiveType(param_shapes[0].element_type());

            // Perform device-to-device memory copy
            ret = npu::aclrtMemcpy(output_device_addr, input_size, input_device_addr, input_size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg = absl::StrFormat("aclrtMemcpy for reshape failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            // Synchronize stream to ensure copy completes
            ret = npu::aclrtSynchronizeStream(stream);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg = absl::StrFormat("Synchronize stream after reshape failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            exec_ret = absl::OkStatus();
            break;
        }

        case HloOpcode::kBroadcast: {
            std::vector<int64_t> output_param_size_data = NpuTool::ConvertXlaShape2VectorShape(param_shapes.back());
            aclIntArray* output_param_size =
                npu::aclCreateIntArray(output_param_size_data.data(), output_param_size_data.size());

            // 调用aclnnExpand第一段接口
            ret = npu::aclnnExpandGetWorkspaceSize(acl_tensor_ptr_vector[0], output_param_size,
                                                   acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg =
                    absl::StrFormat("Call aclnnExpandGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
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
                const std::string err_msg =
                    absl::StrFormat("Call aclnnAddGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnAdd);
            break;
        }

        case HloOpcode::kTanh: {
            ret = npu::aclnnTanhGetWorkspaceSize(acl_tensor_ptr_vector[0], acl_tensor_ptr_vector.back(), &workspaceSize,
                                                 &executor);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg =
                    absl::StrFormat("Call aclnnTanhGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnTanh);
            break;
        }

        case HloOpcode::kMultiply: {
            ret = npu::aclnnMulGetWorkspaceSize(acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
                                                acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg =
                    absl::StrFormat("Call aclnnMulGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnMul);
            break;
        }

        case HloOpcode::kDot: {
            int8_t cubeMathType = 1; // ALLOW_FP32_DOWN_PRECISION
            ret =
                npu::aclnnMatmulGetWorkspaceSize(acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
                                                 acl_tensor_ptr_vector.back(), cubeMathType, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg =
                    absl::StrFormat("Call aclnnMatmulGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnMatmul);
            break;
        }

        case HloOpcode::kDivide: {
            ret = npu::aclnnDivGetWorkspaceSize(acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
                                                acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg =
                    absl::StrFormat("Call aclnnDivGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnDiv);
            break;
        }

        case HloOpcode::kSubtract: {
            float alphaValue = 1.0;
            aclScalar* alpha = npu::aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

            ret = npu::aclnnSubGetWorkspaceSize(acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1], alpha,
                                                acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg =
                    absl::StrFormat("Call aclnnSubGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnSub);
            break;
        }

        case HloOpcode::kNegate: {
            ret = npu::aclnnNegGetWorkspaceSize(acl_tensor_ptr_vector[0], acl_tensor_ptr_vector.back(), &workspaceSize,
                                                &executor);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg =
                    absl::StrFormat("Call aclnnNegGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnNeg);
            break;
        }

        case HloOpcode::kExp: {
            ret = npu::aclnnExpGetWorkspaceSize(acl_tensor_ptr_vector[0], acl_tensor_ptr_vector.back(), &workspaceSize,
                                                &executor);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg =
                    absl::StrFormat("Call aclnnExpGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnExp);
            break;
        }

        case HloOpcode::kMaximum: {
            ret = npu::aclnnMaximumGetWorkspaceSize(acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
                                                    acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg =
                    absl::StrFormat("Call aclnnMaximumGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnMaximum);
            break;
        }

        case HloOpcode::kConcatenate: {
            // 获取concatenate维度, dimensions向量中应该只有一个元素
            if (thunk_info().dimensions.empty()) {
                const std::string err_msg = "Concatenate dimensions is empty";
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }
            int64_t dim = thunk_info().dimensions[0];

            // 创建aclTensorList,包含所有输入tensor(除最后一个输出外)
            std::vector<aclTensor*> input_tensors;
            for (size_t i = 0; i < acl_tensor_ptr_vector.size() - 1; ++i) {
                input_tensors.push_back(acl_tensor_ptr_vector[i]);
            }
            aclTensorList* tensor_list = npu::aclCreateTensorList(
                input_tensors.data(), input_tensors.size());
            if (tensor_list == nullptr) {
                const std::string err_msg = "Failed to create aclTensorList for concatenate";
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            // 调用aclnnCatGetWorkspaceSize
            ret = npu::aclnnCatGetWorkspaceSize(tensor_list, dim, acl_tensor_ptr_vector.back(),
                                                &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg =
                    absl::StrFormat("Call aclnnCatGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                npu::aclDestroyTensorList(tensor_list);
                return absl::InternalError(err_msg);
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnCat);

            // 清理tensor_list
            npu::aclDestroyTensorList(tensor_list);
            break;
        }

        case HloOpcode::kReduce: {
            // 获取 reduce 的维度信息
            if (thunk_info().dimensions.empty()) {
                const std::string err_msg = "Reduce dimensions is empty";
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            // 创建维度数组
            aclIntArray* dims = npu::aclCreateIntArray(thunk_info().dimensions.data(), thunk_info().dimensions.size());
            if (dims == nullptr) {
                const std::string err_msg = "Failed to create aclIntArray for reduce dimensions";
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
            }

            // 获取输出数据类型
            aclDataType out_dtype = NpuTool::ConvertXlaType2AclDataType(param_shapes.back().element_type());

            // 调用 aclnnReduceSum 第一段接口
            // aclnnReduceSum 接口参数: input, dims, keepDims, dtype, output, workspaceSize, executor
            ret = npu::aclnnReduceSumGetWorkspaceSize(acl_tensor_ptr_vector[0], dims, false, out_dtype,
                                                      acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                const std::string err_msg = absl::StrFormat("Call aclnnReduceSumGetWorkspaceSize failed. Error code: %d", ret);
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                npu::aclDestroyIntArray(dims);
                return absl::InternalError(err_msg);
            }

            exec_ret = ExecuteAclnn(stream, workspaceSize, executor, npu::aclnnReduceSum);

            // 清理维度数组
            npu::aclDestroyIntArray(dims);
            break;
        }

        default:
            const std::string err_msg =
                absl::StrFormat("Op code(%v) does not support execute now", HloOpcodeString(thunk_info().op_code));
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
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

        if (i < args_.size() - 1) {
            // 只能执行前打印,因为算子执行会复用内存,执行结果会覆盖到入参地址上
            NPU_LOG_FORMAT_PRINT(DEBUG, "---- Param(input) %d info: ----", i);
            NPU_LOG_FORMAT_PRINT(DEBUG, "  Buffer allocation: %v", args_[i].ToString());
            NPU_LOG_FORMAT_PRINT(DEBUG, "  Tensor Shape: %v", param_shapes[i].ToString(true));
            NPU_LOG_FORMAT_PRINT(DEBUG, "  Device addr: %p", device_addr);
            NpuTool::PrintNpuDeviceMemoryValue<float>("  Value on device:", NpuTool::GetXlaShapeSize(param_shapes[i]),
                                                      device_addr);
        }
        // 最后一个追加的tensor_ptr为出参tensor
    }

    return acl_tensor_ptr_vector;
}

absl::Status NpuAclnnThunk::ExecuteAclnn(aclrtStream& stream, uint64_t workspaceSize, aclOpExecutor* executor,
                                         AclnnExecFunc aclnnExecFunc) {
    aclnnStatus ret = 0;

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        NPU_LOG_FORMAT_PRINT(DEBUG, "Additional aclnn WorkspaceSize is %d, need to apply", workspaceSize);
        ret = npu::aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat("Allocate workspace failed. Error code: %d", ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
        }
    } else {
        NPU_LOG_FORMAT_PRINT(DEBUG, "Additional aclnn WorkspaceSize is 0, no need to apply");
    }
    // 调用aclnnExecFunc第二段接口
    ret = aclnnExecFunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat("Execute aclnnExecFunc failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
    }

    ret = npu::aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat("Synchronize stream failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
    }

    return absl::OkStatus();
}

absl::Status NpuAclnnThunk::PrintOutputTensor(const NpuBufferAllocations* allocations,
                                              const std::vector<Shape>& param_shapes) {
    // 对于当前aclnn处理的简单算子来说,只有最后一个参数为输出参数
    const auto& output_arg = args_.back();
    const auto& outout_arg_param = param_shapes.back();

    auto addr = allocations->GetDeviceAddress(output_arg);
    void* device_addr = addr.opaque();
    NPU_LOG_FORMAT_PRINT(DEBUG, "---- Param(output) 0 info: ----");
    NPU_LOG_FORMAT_PRINT(DEBUG, "  Buffer allocation: %v", output_arg.ToString());
    NPU_LOG_FORMAT_PRINT(DEBUG, "  Tensor Shape: %v", outout_arg_param.ToString(true));
    NPU_LOG_FORMAT_PRINT(DEBUG, "  Device addr: %p", device_addr);
    NpuTool::PrintNpuDeviceMemoryValueByShape("  Value on device:", outout_arg_param, device_addr);

    return absl::OkStatus();
}

}  // namespace xla
