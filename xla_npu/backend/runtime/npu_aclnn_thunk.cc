/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/xla_npu/backend/runtime/npu_aclnn_thunk.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_workspace_memory_pool.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_op_interface.h"
#include "xla/xla_npu/dependency/tool/config_loader.h"
#include "xla/xla_npu/service/npu_buffer_allocations.h"

namespace xla {

static bool CheckInputOuputParamsSize(HloOpcode op_code,
                                      const std::vector<Shape>& param_shapes) {
  switch (op_code) {
    case HloOpcode::kBroadcast:
    case HloOpcode::kTanh:
    case HloOpcode::kReshape:
    case HloOpcode::kNegate:
    case HloOpcode::kExp:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSin:
    case HloOpcode::kCos:
      if (param_shapes.size() != 2) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "Param shape size %d does not meet target value 2",
                             param_shapes.size());
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
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "Param shape size %d does not meet target value 3",
                             param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kConcatenate:
      // Concatenate 是变长参数算子，至少需要 2 个输入 + 1 个输出 = 至少 3 个参数
      if (param_shapes.size() < 3) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "Concatenate param shape size %d is less than 3",
                             param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kReduce:
      // Reduce 操作：输入 + 初始值 + 输出 = 3 个参数
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Reduce param shape size %d does not meet target value 3",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kTranspose:
      // Transpose 操作：输入 + 输出 = 2 个参数
      if (param_shapes.size() != 2) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Transpose param shape size %d does not meet target value 2",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kSlice:
      // Slice 操作：输入 + 输出 = 2 个参数
      // Slice 是静态切片，参数在编译时确定，不需要额外的 start_indices 参数
      if (param_shapes.size() != 2) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Slice param shape size %d does not meet target value 2",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kDynamicSlice:
      // DynamicSlice 操作：输入 + start_indices... + 输出 = 至少 3 个参数
      // start_indices 数量 = param_shapes.size() - 2（每个切片维度一个）
      if (param_shapes.size() < 3) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "DynamicSlice param shape size %d is less than 3 "
                             "(input + start_indices + output)",
                             param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kGather:
      // Gather 操作：输入 + 索引 + 输出 = 3 个参数
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Gather param shape size %d does not meet target value 3",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kCompare:
      // Compare 操作：输入 1 + 输入 2 + 输出 = 3 个参数
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Compare param shape size %d does not meet target value 3",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kSelect:
      // Select (Where) 操作：condition + self + other + output = 4 个参数
      if (param_shapes.size() != 4) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Select param shape size %d does not meet target value 4",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kAnd:
      // And (BitwiseAnd/LogicalAnd) 操作：input1 + input2 + output = 3 个参数
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "And param shape size %d does not meet target value 3",
            param_shapes.size());
        return false;
      }
      break;
    default:
      NPU_LOG_FORMAT_PRINT(ERROR, "Op code(%v) does not support check now",
                           HloOpcodeString(op_code));
  }

  return true;
}

NpuAclnnThunk::NpuAclnnThunk(std::string fingerprint, ThunkInfo thunk_info,
                             std::vector<BufferAllocation::Slice> args,
                             std::vector<bool> output_args,
                             std::optional<int64_t> sdpa_dropout_seed)
    : NpuThunk(Kind::kCuDnn, std::move(thunk_info)),
      fingerprint_(std::move(fingerprint)),
      graph_(std::make_shared<se::dnn::LazyDnnGraph>(nullptr)),
      args_(std::move(args)),
      // output_args_(std::move(output_args)) 表示args_中哪些内存地址是会被写入的,用于存在内存复用,
      // 假设add的args_大小为3,则可能后2个arg内存地址相同,都为可写入
      sdpa_dropout_seed_(sdpa_dropout_seed) {}

absl::Status NpuAclnnThunk::Initialize(const InitializeParams& params) {
  absl::Status ret = absl::OkStatus();

  // 在 Initialize 阶段处理所有常量 buffer 中的特殊值 (-inf, +inf, nan)
  // 这些特殊值只在图分配时写入，计算过程中不会产生
  TF_RETURN_IF_ERROR(NpuThunk::InitializeSpecialValues(params, args_));
  return ret;
}

absl::Status NpuAclnnThunk::ExecuteOnStream(const ExecuteParams& params) {
  NPU_LOG_FORMAT_PRINT(
      DEBUG, "==== Launch aclnn kernel with %d param ====", args_.size());
  NPU_LOG_FORMAT_PRINT(DEBUG, "---- Corresponding hlo instruction: %s ----",
                       thunk_info().hloInstrDesc);

  // Initialize 应该在执行前已经调用过，这里不应该再次调用

  if (params.stream == nullptr) {
    NPU_LOG_FORMAT_PRINT(ERROR,
                         "Stream object is null before executing NPU thunk.");
    return absl::InternalError("");
  }
  auto stream_handle = params.stream->platform_specific_handle();
  if (stream_handle.stream == nullptr) {
    NPU_LOG_FORMAT_PRINT(
        ERROR, "Native stream handle is null before executing NPU thunk.");
    return absl::InternalError("");
  }
  auto stream = stream_handle.stream;
  auto acl_stream = static_cast<aclrtStream>(stream_handle.stream);

  const NpuBufferAllocations* allocations = params.buffer_allocations;
  const auto& param_shapes = thunk_info().param_shapes;
  aclnnStatus ret = 0;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  auto exec_ret = absl::OkStatus();

  if (!CheckInputOuputParamsSize(thunk_info().op_code, param_shapes)) {
    return absl::InternalError("");
  }

  // 执行前打印输入参数
  if (IsPrintIOValuesEnabled()) {
    int device_ordinal = params.stream->parent()->device_ordinal();
    for (int i = 0; i < args_.size() - 1; ++i) {
      auto addr = allocations->GetDeviceAddress(args_[i]);
      PrintParamInfo(i, "input", device_ordinal, args_[i], param_shapes[i],
                     addr.opaque());
    }
  }

  TF_ASSIGN_OR_RETURN(auto acl_tensor_ptr_vector,
                      ExtractAclInputOutTensorPtr(allocations, param_shapes));

  // 打印输入输出的 allocation_index，用于诊断 alias 问题
  if (args_.size() >= 2) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Allocation indices: input=%d, output=%d",
                         args_[0].index(), args_.back().index());
  }

  switch (thunk_info().op_code) {
    case HloOpcode::kReshape: {
      // Reshape operation: copy data from input to output using aclrtMemcpy
      // Even if shapes are different, the underlying data size is the same
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "Reshape operation: using aclrtMemcpy to copy data");

      // Get input and output buffer addresses
      auto input_addr = allocations->GetDeviceAddress(args_[0]);
      auto output_addr = allocations->GetDeviceAddress(args_.back());
      void* input_device_addr = input_addr.opaque();
      void* output_device_addr = output_addr.opaque();

      // Calculate data size in bytes
      int64_t input_size =
          NpuTool::GetXlaShapeSize(param_shapes[0]) *
          ShapeUtil::ByteSizeOfPrimitiveType(param_shapes[0].element_type());

      // Perform device-to-device memory copy
      ret = dep::aclrtMemcpy(output_device_addr, input_size, input_device_addr,
                             input_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "aclrtMemcpy for reshape failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = absl::OkStatus();
      break;
    }

    case HloOpcode::kBroadcast: {
      std::vector<int64_t> output_param_size_data =
          NpuTool::ConvertXlaShape2VectorShape(param_shapes.back());
      std::vector<int64_t> input_shape_data =
          NpuTool::ConvertXlaShape2VectorShape(param_shapes[0]);

      // Debug: print broadcast shapes
      std::string shape_str = "[";
      for (size_t i = 0; i < output_param_size_data.size(); ++i) {
        if (i > 0) shape_str += ", ";
        shape_str += std::to_string(output_param_size_data[i]);
      }
      shape_str += "]";
      NPU_LOG_FORMAT_PRINT(DEBUG, "Broadcast: input shape %s, target size %s",
                           NpuTool::GetXlaShapeString(param_shapes[0]).c_str(),
                           shape_str.c_str());

      // Check if input and output shapes are the same (identity broadcast)
      if (input_shape_data == output_param_size_data) {
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Identity broadcast detected, using stream Memcpy "
                             "instead of aclnnExpand");

        // Get input and output buffer addresses
        auto input_addr = allocations->GetDeviceAddress(args_[0]);
        auto output_addr = allocations->GetDeviceAddress(args_.back());

        // Calculate data size
        int64_t data_size = NpuTool::GetXlaShapeSize(param_shapes.back()) *
                            ShapeUtil::ByteSizeOfPrimitiveType(
                                param_shapes.back().element_type());

        // Perform device-to-device memory copy using stream
        exec_ret = params.stream->Memcpy(&output_addr, input_addr, data_size);
        if (!exec_ret.ok()) {
          NPU_LOG_FORMAT_PRINT(
              ERROR, "Stream Memcpy for identity broadcast failed: %s",
              exec_ret.ToString().c_str());
          return exec_ret;
        }

        break;
      }

      // Handle dimension expansion for aclnnExpand
      // aclnnExpand requires input and output to have the same number of dimensions
      // If output has more dimensions, we need to create a view with expanded dimensions
      aclTensor* input_tensor = acl_tensor_ptr_vector[0];

      if (output_param_size_data.size() > input_shape_data.size()) {
        NPU_LOG_FORMAT_PRINT(
            DEBUG, "Broadcast: need to expand dimensions from %d to %d",
            input_shape_data.size(), output_param_size_data.size());

        // Get dimensions parameter from HLO instruction
        const auto& broadcast_dims = thunk_info().dimensions;

        // Create expanded shape and strides for the view tensor
        std::vector<int64_t> expanded_shape(output_param_size_data.size(), 1);
        std::vector<int64_t> expanded_strides(output_param_size_data.size(), 0);

        // Map input dimensions to output dimensions based on broadcast_dims
        for (size_t i = 0; i < broadcast_dims.size(); ++i) {
          int64_t out_dim = broadcast_dims[i];
          expanded_shape[out_dim] = input_shape_data[i];
        }

        // Calculate strides for the view tensor
        // Dimensions with size 1 should have stride 0 (for broadcasting)
        // Dimensions mapped from input should have proper strides
        int64_t stride = 1;
        for (int i = output_param_size_data.size() - 1; i >= 0; --i) {
          if (expanded_shape[i] == 1) {
            expanded_strides[i] = 0;  // stride 0 for broadcasting dimension
          } else {
            expanded_strides[i] = stride;
            stride *= expanded_shape[i];
          }
        }

        // Debug print
        std::string expanded_shape_str = "[";
        for (size_t i = 0; i < expanded_shape.size(); ++i) {
          if (i > 0) expanded_shape_str += ", ";
          expanded_shape_str += std::to_string(expanded_shape[i]);
        }
        expanded_shape_str += "]";

        std::string expanded_strides_str = "[";
        for (size_t i = 0; i < expanded_strides.size(); ++i) {
          if (i > 0) expanded_strides_str += ", ";
          expanded_strides_str += std::to_string(expanded_strides[i]);
        }
        expanded_strides_str += "]";

        NPU_LOG_FORMAT_PRINT(
            DEBUG, "Broadcast: expanded view shape %s, strides %s",
            expanded_shape_str.c_str(), expanded_strides_str.c_str());

        // Create a view tensor with expanded dimensions
        aclDataType data_type =
            NpuTool::ConvertXlaType2AclDataType(param_shapes[0].element_type());

        auto input_addr = allocations->GetDeviceAddress(args_[0]);
        void* device_addr = input_addr.opaque();

        input_tensor = dep::aclCreateTensor(
            expanded_shape.data(), expanded_shape.size(), data_type,
            expanded_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            expanded_shape.data(), expanded_shape.size(), device_addr);
        if (input_tensor == nullptr) {
          const std::string err_msg = "Failed to create expanded view tensor";
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }
      }

      aclIntArray* output_param_size = dep::aclCreateIntArray(
          output_param_size_data.data(), output_param_size_data.size());

      // 调用 aclnnExpand 第一段接口
      ret = dep::aclnnExpandGetWorkspaceSize(input_tensor, output_param_size,
                                             acl_tensor_ptr_vector.back(),
                                             &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnExpandGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnExpand);

      // Clean up the output_param_size array
      dep::aclDestroyIntArray(output_param_size);
      break;
    }

    case HloOpcode::kAdd: {
      // 优化：对于 RMSNorm 中的 variance + epsilon 场景
      // 检测输入类型，如果是 FP16/BF16，尝试使用 FP32 计算以减少精度损失
      aclDataType input_type =
          NpuTool::ConvertXlaType2AclDataType(param_shapes[0].element_type());

      // RMSNorm 关键优化：FP16/BF16 输入时，使用 FP32 计算
      bool use_fp32_compute =
          (input_type == ACL_FLOAT16 || input_type == ACL_BF16);

      if (use_fp32_compute) {
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Add: Using FP32 compute for FP16/BF16 input to "
                             "improve precision (RMSNorm optimization)");
      }

      // 使用 union 存储不同类型的 alpha 值
      union {
        int32_t i32;
        int64_t i64;
        float f32;
        double f64;
        aclFloat16 f16;
        uint16_t bf16;
      } alpha_val;

      // 如果启用 FP32 计算，使用 FP32 类型创建 alpha
      aclDataType alpha_type = use_fp32_compute ? ACL_FLOAT : input_type;

      // 根据类型设置 alpha 值（统一为 1 或 1.0）
      switch (alpha_type) {
        case ACL_INT32:
        case ACL_INT16:
        case ACL_INT8:
          alpha_val.i32 = 1;
          break;
        case ACL_INT64:
          alpha_val.i64 = 1;
          break;
        case ACL_UINT8:
          alpha_val.i32 = 1;
          break;
        case ACL_FLOAT:
          alpha_val.f32 = 1.0f;
          break;
        case ACL_DOUBLE:
          alpha_val.f64 = 1.0;
          break;
        case ACL_FLOAT16:
          alpha_val.f16 = 1.0f;
          break;
        case ACL_BF16:
          alpha_val.bf16 = 0x3F80;
          break;
        default:
          alpha_val.f32 = 1.0f;
          alpha_type = ACL_FLOAT;
          break;
      }

      aclScalar* alpha = dep::aclCreateScalar(&alpha_val, alpha_type);

      ret = dep::aclnnAddGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1], alpha,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnAddGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        dep::aclDestroyScalar(alpha);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnAdd);

      dep::aclDestroyScalar(alpha);
      break;
    }

    case HloOpcode::kTanh: {
      ret = dep::aclnnTanhGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                           acl_tensor_ptr_vector.back(),
                                           &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnTanhGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnTanh);
      break;
    }

    case HloOpcode::kMultiply: {
      NPU_LOG_FORMAT_PRINT(
          DEBUG,
          "Multiply: input[0]=%p (shape=%s), input[1]=%p (shape=%s), output=%p",
          acl_tensor_ptr_vector[0],
          NpuTool::GetXlaShapeString(param_shapes[0]).c_str(),
          acl_tensor_ptr_vector[1],
          NpuTool::GetXlaShapeString(param_shapes[1]).c_str(),
          acl_tensor_ptr_vector.back());

      // XLA会在multiply之前插入显式broadcast操作
      // 所以multiply的两个输入形状应该相同，直接执行即可
      ret = dep::aclnnMulGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);

      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnMulGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnMul);
      break;
    }

    case HloOpcode::kDot: {
      int8_t cubeMathType = 0;
      ret = dep::aclnnMatmulGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), cubeMathType, &workspaceSize,
          &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnMatmulGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnMatmul);
      break;
    }

    case HloOpcode::kDivide: {
      ret = dep::aclnnDivGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnDivGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnDiv);
      break;
    }

    case HloOpcode::kSubtract: {
      float alphaValue = 1.0;
      aclScalar* alpha =
          dep::aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

      ret = dep::aclnnSubGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1], alpha,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSubGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSub);
      break;
    }

    case HloOpcode::kNegate: {
      ret = dep::aclnnNegGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnNegGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnNeg);
      break;
    }

    case HloOpcode::kExp: {
      ret = dep::aclnnExpGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnExpGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnExp);
      break;
    }

    case HloOpcode::kRsqrt: {
      // 优化：对于 RMSNorm 等场景，检查输入类型
      // Rsqrt 在 FP16 输入时可能有精度损失，但 ACLNN 没有特殊处理
      aclDataType input_dtype =
          NpuTool::ConvertXlaType2AclDataType(param_shapes[0].element_type());

      if (input_dtype == ACL_FLOAT16 || input_dtype == ACL_BF16) {
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Rsqrt: FP16/BF16 input detected, using default "
                             "ACLNN implementation");
      }

      ret = dep::aclnnRsqrtGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                            acl_tensor_ptr_vector.back(),
                                            &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnRsqrtGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnRsqrt);
      break;
    }

    case HloOpcode::kSin: {
      ret = dep::aclnnSinGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSinGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSin);
      break;
    }

    case HloOpcode::kCos: {
      ret = dep::aclnnCosGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnCosGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnCos);
      break;
    }

    case HloOpcode::kMaximum: {
      ret = dep::aclnnMaximumGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnMaximumGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnMaximum);
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
      aclTensorList* tensor_list =
          dep::aclCreateTensorList(input_tensors.data(), input_tensors.size());
      if (tensor_list == nullptr) {
        const std::string err_msg =
            "Failed to create aclTensorList for concatenate";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 调用 aclnnCatGetWorkspaceSize
      ret = dep::aclnnCatGetWorkspaceSize(tensor_list, dim,
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnCatGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnCat);

      // 清理 tensor_list
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
      aclIntArray* dims = dep::aclCreateIntArray(
          thunk_info().dimensions.data(), thunk_info().dimensions.size());
      if (dims == nullptr) {
        const std::string err_msg =
            "Failed to create aclIntArray for reduce dimensions";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 获取输入数据类型
      aclDataType input_dtype =
          NpuTool::ConvertXlaType2AclDataType(param_shapes[0].element_type());

      // 优化：对于 FP16/BF16 输入，使用 FP32 累加以减少精度损失
      // 对于 FP32 输入，使用输入类型以保持数据类型匹配
      aclDataType accumulate_dtype = input_dtype;
      if (input_dtype == ACL_FLOAT16 || input_dtype == ACL_BF16) {
        accumulate_dtype = ACL_FLOAT;
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "ReduceSum: Using FP32 accumulation for FP16/BF16 "
                             "input to reduce precision loss");
      }

      // 对于reduce操作,aclnnReduceSum只需要输入和输出,不需要初始值
      // XLA的reduce有3个参数:输入、初始值、输出
      // 但aclnnReduceSum接口只接受输入和输出,内部使用0作为初始值
      // 所以这里使用acl_tensor_ptr_vector[0]作为输入,acl_tensor_ptr_vector[2]作为输出
      // 注意:acl_tensor_ptr_vector[1]是初始值,不应该传给aclnnReduceSum
      if (acl_tensor_ptr_vector.size() < 3) {
        const std::string err_msg = absl::StrFormat(
            "Reduce operation expects 3 parameters (input, init, output), but "
            "got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        dep::aclDestroyIntArray(dims);
        return absl::InternalError(err_msg);
      }

      // 根据hloInstrDesc判断reduce类型
      // reduce_max: %reduce_max.x = ... maximum(...)
      // reduce_sum: %reduce_sum.x = ... add(...)
      bool is_reduce_max =
          (thunk_info().hloInstrDesc.find("reduce_max") != std::string::npos) ||
          (thunk_info().hloInstrDesc.find("maximum") != std::string::npos);
      bool is_reduce_sum =
          (thunk_info().hloInstrDesc.find("reduce_sum") != std::string::npos) ||
          (thunk_info().hloInstrDesc.find("add") != std::string::npos);

      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "Reduce operation: hloInstrDesc=%s, "
                           "is_reduce_max=%d, is_reduce_sum=%d",
                           thunk_info().hloInstrDesc.c_str(), is_reduce_max,
                           is_reduce_sum);

      // 打印维度信息
      std::string dims_str = "[";
      for (int i = 0; i < thunk_info().dimensions.size(); ++i) {
        if (i > 0) dims_str += ", ";
        dims_str += std::to_string(thunk_info().dimensions[i]);
      }
      dims_str += "]";
      NPU_LOG_FORMAT_PRINT(DEBUG, "Reduce dimensions: %s", dims_str.c_str());

      if (is_reduce_max) {
        // 使用 aclnnAmax 实现 ReduceMax
        // aclnnAmax 返回张量在指定维度上的最大值（不包含索引）
        NPU_LOG_FORMAT_PRINT(DEBUG, "ReduceMax using aclnnAmax");
        NPU_LOG_FORMAT_PRINT(
            DEBUG, "Input shape: %s, Output shape: %s",
            NpuTool::GetXlaShapeString(param_shapes[0]).c_str(),
            NpuTool::GetXlaShapeString(param_shapes[2]).c_str());

        // 调用 aclnnAmax 第一段接口
        // aclnnAmax 接口参数: input, dim, keepDim, out, workspaceSize, executor
        ret = dep::aclnnAmaxGetWorkspaceSize(acl_tensor_ptr_vector[0], dims,
                                             false, acl_tensor_ptr_vector[2],
                                             &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          const std::string err_msg = absl::StrFormat(
              "Call aclnnAmaxGetWorkspaceSize failed. Error code: %d", ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          dep::aclDestroyIntArray(dims);
          return absl::InternalError(err_msg);
        }

        exec_ret =
            ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnAmax);
        if (!exec_ret.ok()) {
          NPU_LOG_FORMAT_PRINT(ERROR, "ExecuteAclnn for aclnnAmax failed: %s",
                               exec_ret.ToString().c_str());
          dep::aclDestroyIntArray(dims);
          return exec_ret;
        }
      } else {
        // 默认使用 ReduceSum
        // 调用 aclnnReduceSum 第一段接口
        // aclnnReduceSum 接口参数: input, dims, keepDims, dtype, output,
        // workspaceSize, executor
        ret = dep::aclnnReduceSumGetWorkspaceSize(
            acl_tensor_ptr_vector[0], dims, false, accumulate_dtype,
            acl_tensor_ptr_vector[2], &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          const std::string err_msg = absl::StrFormat(
              "Call aclnnReduceSumGetWorkspaceSize failed. Error code: %d",
              ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          dep::aclDestroyIntArray(dims);
          return absl::InternalError(err_msg);
        }

        exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                dep::aclnnReduceSum);
      }

      // 清理维度数组
      dep::aclDestroyIntArray(dims);
      break;
    }

    case HloOpcode::kTranspose: {
      // Transpose 需要 dimensions 参数作为 permute 的维度顺序
      if (thunk_info().dimensions.empty()) {
        const std::string err_msg = "Transpose dimensions is empty";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 创建 aclIntArray 作为 permute 维度
      aclIntArray* perm = dep::aclCreateIntArray(
          thunk_info().dimensions.data(), thunk_info().dimensions.size());
      if (perm == nullptr) {
        const std::string err_msg = "Failed to create perm array for transpose";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 调用 aclnnPermuteGetWorkspaceSize
      ret = dep::aclnnPermuteGetWorkspaceSize(acl_tensor_ptr_vector[0], perm,
                                              acl_tensor_ptr_vector.back(),
                                              &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnPermuteGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        dep::aclDestroyIntArray(perm);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnPermute);

      // 清理 perm 数组
      dep::aclDestroyIntArray(perm);
      break;
    }

    case HloOpcode::kSlice: {
      // Slice 操作：从 HLO 指令描述中解析 starts, ends, axes, steps
      // Slice 是静态的，参数在编译时确定
      const auto& input_shape = param_shapes[0];
      const auto& output_shape = param_shapes.back();

      // 从 hloInstrDesc 解析 slice 参数
      // 格式示例: "slice={[0:1], [0:14], [0:8], [32:64]}"
      std::vector<int64_t> starts_data;
      std::vector<int64_t> ends_data;
      std::vector<int64_t> axes_data;
      std::vector<int64_t> steps_data;

      const std::string& desc = thunk_info().hloInstrDesc;

      // 尝试从 hloInstrDesc 解析 slice 参数
      // 查找 "slice={" 的位置
      size_t slice_pos = desc.find("slice={");
      if (slice_pos != std::string::npos) {
        // 提取 slice 参数部分
        size_t start_pos = slice_pos + strlen("slice={");
        size_t end_pos = desc.find("}", start_pos);
        if (end_pos != std::string::npos) {
          std::string slice_str = desc.substr(start_pos, end_pos - start_pos);
          NPU_LOG_FORMAT_PRINT(DEBUG, "Parsing slice params from: %s",
                               slice_str.c_str());

          // 解析每个维度的 [start:end]
          int dim = 0;
          size_t bracket_pos = 0;
          while ((bracket_pos = slice_str.find("[", bracket_pos)) !=
                 std::string::npos) {
            size_t close_bracket = slice_str.find("]", bracket_pos);
            if (close_bracket == std::string::npos) break;

            std::string range_str = slice_str.substr(
                bracket_pos + 1, close_bracket - bracket_pos - 1);

            // 解析 start:end
            size_t colon_pos = range_str.find(":");
            if (colon_pos != std::string::npos) {
              std::string start_str = range_str.substr(0, colon_pos);
              std::string end_str = range_str.substr(colon_pos + 1);

              try {
                int64_t start_val = std::stoll(start_str);
                int64_t end_val = std::stoll(end_str);

                starts_data.push_back(start_val);
                ends_data.push_back(end_val);
                // CANN 的 aclnnSliceV2 期望逻辑维度顺序的 axes
                // XLA 的 slice 参数已经是逻辑维度顺序
                axes_data.push_back(dim);
                steps_data.push_back(1);

                NPU_LOG_FORMAT_PRINT(INFO, "Slice logical dim %d: [%ld:%ld]",
                                     dim, start_val, end_val);
              } catch (...) {
                NPU_LOG_FORMAT_PRINT(ERROR, "Failed to parse slice range: %s",
                                     range_str.c_str());
              }
            }

            dim++;
            bracket_pos = close_bracket + 1;
          }
        }
      }

      // 如果解析失败，使用默认参数
      if (starts_data.empty()) {
        NPU_LOG_FORMAT_PRINT(
            WARN,
            "Failed to parse slice params from hloInstrDesc, using default");
        for (int64_t i = 0; i < input_shape.dimensions().size(); ++i) {
          starts_data.push_back(0);
          ends_data.push_back((i < output_shape.dimensions().size())
                                  ? output_shape.dimensions()[i]
                                  : input_shape.dimensions()[i]);
          // CANN 的 aclnnSliceV2 期望逻辑维度顺序的 axes
          axes_data.push_back(i);
          steps_data.push_back(1);
        }
      }

      NPU_LOG_FORMAT_PRINT(DEBUG, "Slice: input=%s, output=%s",
                           NpuTool::GetXlaShapeString(input_shape).c_str(),
                           NpuTool::GetXlaShapeString(output_shape).c_str());

      // 创建 aclIntArray
      aclIntArray* starts =
          dep::aclCreateIntArray(starts_data.data(), starts_data.size());
      aclIntArray* ends =
          dep::aclCreateIntArray(ends_data.data(), ends_data.size());
      aclIntArray* axes =
          dep::aclCreateIntArray(axes_data.data(), axes_data.size());
      aclIntArray* steps =
          dep::aclCreateIntArray(steps_data.data(), steps_data.size());

      if (starts == nullptr || ends == nullptr || axes == nullptr ||
          steps == nullptr) {
        const std::string err_msg = "Failed to create aclIntArray for slice";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        if (starts) dep::aclDestroyIntArray(starts);
        if (ends) dep::aclDestroyIntArray(ends);
        if (axes) dep::aclDestroyIntArray(axes);
        if (steps) dep::aclDestroyIntArray(steps);
        return absl::InternalError(err_msg);
      }

      // 调用 aclnnSliceV2GetWorkspaceSize
      ret = dep::aclnnSliceV2GetWorkspaceSize(
          acl_tensor_ptr_vector[0], starts, ends, axes, steps,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);

      // 清理 aclIntArray
      dep::aclDestroyIntArray(starts);
      dep::aclDestroyIntArray(ends);
      dep::aclDestroyIntArray(axes);
      dep::aclDestroyIntArray(steps);

      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSliceV2GetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSliceV2);
      break;
    }

    case HloOpcode::kDynamicSlice: {
      // DynamicSlice 操作：输入 + start_indices... + 输出
      // 从 HLO 指令中解析 slice_sizes 和 start_indices
      const auto& slice_sizes = thunk_info().dimensions;

      if (slice_sizes.empty()) {
        const std::string err_msg = "DynamicSlice dimensions is empty";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 获取输入 tensor 的 shape
      const auto& input_shape = param_shapes[0];
      const auto& output_shape = param_shapes.back();

      // start_indices 参数数量 = 总参数数 - 2（输入和输出）
      size_t num_start_indices = acl_tensor_ptr_vector.size() - 2;

      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "DynamicSlice: input_shape=%s, output_shape=%s, "
                           "slice_sizes=%zu, num_start_indices=%zu",
                           NpuTool::GetXlaShapeString(input_shape).c_str(),
                           NpuTool::GetXlaShapeString(output_shape).c_str(),
                           slice_sizes.size(), num_start_indices);

      // output_shape.dimensions() 返回的是逻辑维度顺序的 slice sizes
      // CANN 的 aclnnSliceV2 也期望逻辑维度顺序的 axes/starts/ends
      // 所以直接使用 output_shape.dimensions() 作为 slice_sizes_logical
      std::vector<int64_t> slice_sizes_logical(
          output_shape.dimensions().begin(), output_shape.dimensions().end());

      NPU_LOG_FORMAT_PRINT(
          INFO,
          "DynamicSlice: slice_sizes_from_thunk=[%s], "
          "slice_sizes_from_output=[%s]",
          [&slice_sizes]() {
            std::string s;
            for (size_t i = 0; i < slice_sizes.size(); ++i) {
              if (i > 0) s += ", ";
              s += std::to_string(slice_sizes[i]);
            }
            return s;
          }()
              .c_str(),
          [&slice_sizes_logical]() {
            std::string s;
            for (size_t i = 0; i < slice_sizes_logical.size(); ++i) {
              if (i > 0) s += ", ";
              s += std::to_string(slice_sizes_logical[i]);
            }
            return s;
          }()
              .c_str());

      // 构建 starts, ends, axes, steps 数组
      // axes 使用逻辑维度，与 starts/ends 一一对应
      std::vector<int64_t> starts_data;
      std::vector<int64_t> ends_data;
      std::vector<int64_t> axes_data;
      std::vector<int64_t> steps_data;

      // 处理每个切片维度（按逻辑维度顺序）
      for (size_t i = 0; i < slice_sizes_logical.size(); ++i) {
        int64_t start_val = 0;

        // 如果存在对应的 start_indices 参数，从 device 内存读取
        if (i < num_start_indices) {
          // start_indices 参数从 args_[1] 开始（args_[0] 是输入，args_.back() 是输出）
          auto start_idx_addr = allocations->GetDeviceAddress(args_[i + 1]);
          void* start_idx_device_addr = start_idx_addr.opaque();

          // 获取 start_indices 的数据类型和形状
          xla::PrimitiveType start_dtype = param_shapes[i + 1].element_type();
          const auto& start_shape = param_shapes[i + 1];

          NPU_LOG_FORMAT_PRINT(
              DEBUG,
              "DynamicSlice: start_indices[%zu] shape=%s, dtype=%s, "
              "device_addr=%p",
              i, NpuTool::GetXlaShapeString(start_shape).c_str(),
              xla::PrimitiveType_Name(start_dtype).c_str(),
              start_idx_device_addr);

          // 根据数据类型读取值
          if (start_dtype == xla::S32) {
            int32_t host_start_val = 0;
            aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
            aclError memcpy_ret =
                dep::aclrtMemcpy(&host_start_val, sizeof(int32_t),
                                 start_idx_device_addr, sizeof(int32_t), kind);
            if (memcpy_ret == ACL_SUCCESS) {
              start_val = static_cast<int64_t>(host_start_val);
              NPU_LOG_FORMAT_PRINT(
                  DEBUG,
                  "DynamicSlice: read start_indices[%zu] = %d (S32) from "
                  "device",
                  i, host_start_val);
            } else {
              NPU_LOG_FORMAT_PRINT(WARN,
                                   "Failed to read start_indices[%zu] from "
                                   "device, using default 0",
                                   i);
              start_val = 0;
            }
          } else if (start_dtype == xla::S64) {
            int64_t host_start_val = 0;
            aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
            aclError memcpy_ret =
                dep::aclrtMemcpy(&host_start_val, sizeof(int64_t),
                                 start_idx_device_addr, sizeof(int64_t), kind);
            if (memcpy_ret == ACL_SUCCESS) {
              start_val = host_start_val;
              NPU_LOG_FORMAT_PRINT(
                  DEBUG,
                  "DynamicSlice: read start_indices[%zu] = %ld (S64) from "
                  "device",
                  i, start_val);
            } else {
              NPU_LOG_FORMAT_PRINT(WARN,
                                   "Failed to read start_indices[%zu] from "
                                   "device, using default 0",
                                   i);
              start_val = 0;
            }
          } else {
            NPU_LOG_FORMAT_PRINT(
                WARN, "Unsupported start_indices dtype %s, using default 0",
                xla::PrimitiveType_Name(start_dtype).c_str());
          }
        }

        // 计算 end 值：start + slice_size
        int64_t end_val = start_val + slice_sizes_logical[i];

        // 确保 end 不超过输入维度大小
        if (i < input_shape.dimensions().size()) {
          end_val = std::min(end_val, input_shape.dimensions()[i]);
        }

        starts_data.push_back(start_val);
        ends_data.push_back(end_val);
        // 使用逻辑维度作为 axes
        axes_data.push_back(static_cast<int64_t>(i));
        steps_data.push_back(1);

        NPU_LOG_FORMAT_PRINT(INFO,
                             "DynamicSlice logical_dim %zu: start=%ld, "
                             "end=%ld, axis=%zu, step=1",
                             i, start_val, end_val, i);
      }

      // 创建 aclIntArray
      aclIntArray* starts =
          dep::aclCreateIntArray(starts_data.data(), starts_data.size());
      aclIntArray* ends =
          dep::aclCreateIntArray(ends_data.data(), ends_data.size());
      aclIntArray* axes =
          dep::aclCreateIntArray(axes_data.data(), axes_data.size());
      aclIntArray* steps =
          dep::aclCreateIntArray(steps_data.data(), steps_data.size());

      if (starts == nullptr || ends == nullptr || axes == nullptr ||
          steps == nullptr) {
        const std::string err_msg = "Failed to create aclIntArray for slice";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        if (starts) dep::aclDestroyIntArray(starts);
        if (ends) dep::aclDestroyIntArray(ends);
        if (axes) dep::aclDestroyIntArray(axes);
        if (steps) dep::aclDestroyIntArray(steps);
        return absl::InternalError(err_msg);
      }

      // 调用 aclnnSliceV2GetWorkspaceSize
      ret = dep::aclnnSliceV2GetWorkspaceSize(
          acl_tensor_ptr_vector[0], starts, ends, axes, steps,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);

      // 清理 aclIntArray
      dep::aclDestroyIntArray(starts);
      dep::aclDestroyIntArray(ends);
      dep::aclDestroyIntArray(axes);
      dep::aclDestroyIntArray(steps);

      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSliceV2GetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSliceV2);
      break;
    }

    case HloOpcode::kGather: {
      // Gather 操作：输入 + 索引 + 输出 = 3 个参数
      // XLA Gather 是一个复杂的操作，涉及 offset_dims, collapsed_slice_dims, start_index_map 等参数
      if (acl_tensor_ptr_vector.size() < 3) {
        const std::string err_msg = absl::StrFormat(
            "Gather operation expects 3 parameters (input, index, output), but "
            "got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 解析 Gather 维度信息
      int64_t dim = 0;
      const std::string& desc = thunk_info().hloInstrDesc;

      // 尝试从 start_index_map 解析维度
      size_t pos = desc.find("start_index_map={");
      if (pos != std::string::npos) {
        size_t start = pos + strlen("start_index_map={");
        size_t end = desc.find("}", start);
        if (end != std::string::npos) {
          std::string map_str = desc.substr(start, end - start);
          try {
            dim = std::stoll(map_str);
          } catch (...) {
            dim = 0;
          }
        }
      }

      // 如果 dimensions 不为空，优先使用
      if (!thunk_info().dimensions.empty()) {
        dim = thunk_info().dimensions[0];
      }

      NPU_LOG_FORMAT_PRINT(DEBUG, "Gather using dim=%d, hloDesc: %s", dim,
                           desc.c_str());

      // 获取输入和索引的形状信息
      const auto& input_shape = param_shapes[0];
      const auto& index_shape = param_shapes[1];
      const auto& output_shape = param_shapes[2];

      // GatherV2 输出维度 = input_dims + index_dims - 1
      // XLA 期望的输出维度可能会压缩某些维度
      int64_t gather_output_dims =
          input_shape.dimensions().size() + index_shape.dimensions().size() - 1;
      int64_t xla_output_dims = output_shape.dimensions().size();

      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "Gather shapes: input=%s, index=%s, XLA output=%s, "
                           "GatherV2 output dims=%d, XLA output dims=%d",
                           NpuTool::GetXlaShapeString(input_shape).c_str(),
                           NpuTool::GetXlaShapeString(index_shape).c_str(),
                           NpuTool::GetXlaShapeString(output_shape).c_str(),
                           gather_output_dims, xla_output_dims);

      aclTensor* output_tensor = acl_tensor_ptr_vector[2];
      aclTensor* expanded_output_tensor = nullptr;
      bool need_cleanup = false;

      if (gather_output_dims > xla_output_dims) {
        // 需要创建扩展形状的输出tensor
        // GatherV2的输出形状 = index_shape + input_shape[dim+1:]
        // 例如：index=[1,8,1], input=[151936,896], dim=0
        // GatherV2输出 = [1,8,1,896]
        // XLA输出 = [1,8,896]（压缩了index的最后一个维度）

        // 构建GatherV2期望的输出形状
        std::vector<int64_t> expanded_dims;

        // 首先添加索引的所有维度
        for (int64_t dim_val : index_shape.dimensions()) {
          expanded_dims.push_back(dim_val);
        }

        // 然后添加输入除dim维度外的其他维度
        for (int64_t i = 0; i < input_shape.dimensions().size(); ++i) {
          if (i != dim) {
            expanded_dims.push_back(input_shape.dimensions()[i]);
          }
        }

        // 计算stride（连续内存布局）
        std::vector<int64_t> expanded_strides(expanded_dims.size());
        int64_t stride = 1;
        for (int i = expanded_dims.size() - 1; i >= 0; --i) {
          expanded_strides[i] = stride;
          stride *= expanded_dims[i];
        }

        // 获取输出数据类型
        aclDataType data_type =
            NpuTool::ConvertXlaType2AclDataType(output_shape.element_type());

        // 获取输出缓冲区地址
        auto output_addr = allocations->GetDeviceAddress(args_.back());
        void* device_addr = output_addr.opaque();

        // 创建扩展形状的tensor，指向相同的内存
        // 注意：storageDims应该与viewDims相同，因为数据是连续的
        expanded_output_tensor = dep::aclCreateTensor(
            expanded_dims.data(), expanded_dims.size(), data_type,
            expanded_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            expanded_dims.data(), expanded_dims.size(), device_addr);

        if (expanded_output_tensor == nullptr) {
          const std::string err_msg =
              "Failed to create expanded output tensor for Gather";
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }

        output_tensor = expanded_output_tensor;
        need_cleanup = true;

        std::string expanded_shape_str = "[";
        for (size_t i = 0; i < expanded_dims.size(); ++i) {
          if (i > 0) expanded_shape_str += ", ";
          expanded_shape_str += std::to_string(expanded_dims[i]);
        }
        expanded_shape_str += "]";
        NPU_LOG_FORMAT_PRINT(
            DEBUG, "Created expanded output tensor with shape %s for GatherV2",
            expanded_shape_str.c_str());
      }

      // 使用 GatherV2
      ret = dep::aclnnGatherV2GetWorkspaceSize(
          acl_tensor_ptr_vector[0], dim, acl_tensor_ptr_vector[1],
          output_tensor, &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnGatherV2GetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        if (need_cleanup && expanded_output_tensor) {
          dep::aclDestroyTensor(expanded_output_tensor);
        }
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnGatherV2);

      // 清理创建的tensor
      if (need_cleanup && expanded_output_tensor) {
        dep::aclDestroyTensor(expanded_output_tensor);
      }

      break;
    }

    case HloOpcode::kCompare: {
      // Compare 操作：输入 1 + 输入 2 + 输出 = 3 个参数
      if (acl_tensor_ptr_vector.size() < 3) {
        const std::string err_msg = absl::StrFormat(
            "Compare operation expects 3 parameters (input1, input2, output), "
            "but got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 根据 hloInstrDesc 判断比较方向
      // 格式：comparison_direction=EQ/NE/LT/LE/GT/GE
      const std::string& hlo_desc = thunk_info().hloInstrDesc;

      aclnnStatus compare_ret = ACL_SUCCESS;

      if (hlo_desc.find("direction=EQ") != std::string::npos ||
          hlo_desc.find("comparison_direction=EQ") != std::string::npos) {
        // Equal
        compare_ret = dep::aclnnEqTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnEqTensor);
        }
      } else if (hlo_desc.find("direction=NE") != std::string::npos ||
                 hlo_desc.find("comparison_direction=NE") !=
                     std::string::npos) {
        // Not Equal
        compare_ret = dep::aclnnNeTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnNeTensor);
        }
      } else if (hlo_desc.find("direction=LT") != std::string::npos ||
                 hlo_desc.find("comparison_direction=LT") !=
                     std::string::npos) {
        // Less Than
        compare_ret = dep::aclnnLtTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnLtTensor);
        }
      } else if (hlo_desc.find("direction=LE") != std::string::npos ||
                 hlo_desc.find("comparison_direction=LE") !=
                     std::string::npos) {
        // Less Equal
        compare_ret = dep::aclnnLeTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnLeTensor);
        }
      } else if (hlo_desc.find("direction=GT") != std::string::npos ||
                 hlo_desc.find("comparison_direction=GT") !=
                     std::string::npos) {
        // Greater Than
        compare_ret = dep::aclnnGtTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnGtTensor);
        }
      } else if (hlo_desc.find("direction=GE") != std::string::npos ||
                 hlo_desc.find("comparison_direction=GE") !=
                     std::string::npos) {
        // Greater Equal
        compare_ret = dep::aclnnGeTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnGeTensor);
        }
      } else {
        const std::string err_msg = absl::StrFormat(
            "Unknown comparison direction in: %s", hlo_desc.c_str());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      if (compare_ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Compare GetWorkspaceSize failed. Error code: %d", compare_ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }
      break;
    }

    case HloOpcode::kSelect: {
      // Select (Where) 操作：condition + self + other + output = 4 个参数
      // 根据 condition 选择 self 或 other 中的元素
      if (acl_tensor_ptr_vector.size() < 4) {
        const std::string err_msg = absl::StrFormat(
            "Select operation expects 4 parameters (condition, self, other, "
            "output), but got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // acl_tensors[0] = condition
      // acl_tensors[1] = self
      // acl_tensors[2] = other
      // acl_tensors[3] = output
      ret = dep::aclnnSWhereGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector[2], acl_tensor_ptr_vector[3], &workspaceSize,
          &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSWhereGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSWhere);
      break;
    }

    case HloOpcode::kAnd: {
      // And (BitwiseAnd/LogicalAnd) 操作：输入 1 + 输入 2 + 输出 = 3 个参数
      // 支持 BOOL 类型的逻辑与和整数类型的按位与
      if (acl_tensor_ptr_vector.size() < 3) {
        const std::string err_msg = absl::StrFormat(
            "And operation expects 3 parameters (input1, input2, output), but "
            "got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // acl_tensors[0] = self
      // acl_tensors[1] = other
      // acl_tensors[2] = output
      ret = dep::aclnnBitwiseAndTensorGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnBitwiseAndTensorGetWorkspaceSize failed. Error code: %d",
            ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnBitwiseAndTensor);
      break;
    }

    default: {
      const std::string err_msg =
          absl::StrFormat("Op code(%v) does not support execute now",
                          HloOpcodeString(thunk_info().op_code));
      NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
      return absl::InternalError(err_msg);
    }
  }

  // 使用统一的流同步函数
  TF_RETURN_IF_ERROR(SyncStreamIfEnabled(params.stream));

  // 根据环境变量控制是否打印输入输出值(包含d2h操作,影响性能)
  // 打印输出值需要先同步流,确保数据已从设备复制到主机
  if (NPU_CONFIG_GET_BOOL(NpuConfig::Key::ENABLE_PRINT_IO_VALUES)) {
    // 先同步流,确保算子执行完成
    TF_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
    int device_ordinal = params.stream->parent()->device_ordinal();
    PrintOutputTensor(allocations, param_shapes, device_ordinal);
  }

  return exec_ret;
}

absl::StatusOr<std::vector<aclTensor*>>
NpuAclnnThunk::ExtractAclInputOutTensorPtr(
    const NpuBufferAllocations* allocations,
    const std::vector<Shape>& param_shapes) {
  std::vector<aclTensor*> acl_tensor_ptr_vector;

  // 先获取所有参数的设备地址
  std::vector<void*> param_addrs;
  param_addrs.reserve(args_.size());
  for (int i = 0; i < args_.size(); ++i) {
    auto addr = allocations->GetDeviceAddress(args_[i]);
    param_addrs.push_back(addr.opaque());
  }

  // 打印所有参数的地址用于诊断
  std::string addr_debug = "Param addrs: ";
  for (size_t i = 0; i < param_addrs.size(); ++i) {
    addr_debug += absl::StrFormat("[%d]=%p; ", i, param_addrs[i]);
  }
  NPU_LOG_FORMAT_PRINT(DEBUG, addr_debug);

  // 对于 kDot 算子，检测不对称的 batch 维度
  bool dot_lhs_expand = false;
  bool dot_rhs_expand = false;
  // 检查是否有 batch_dims（通过 HLO 描述判断）
  const std::string& hlo_desc = thunk_info().hloInstrDesc;
  bool has_batch_dims =
      (hlo_desc.find("lhs_batch_dims=") != std::string::npos &&
       hlo_desc.find("lhs_batch_dims={}") == std::string::npos);

  if (thunk_info().op_code == HloOpcode::kDot && args_.size() == 3) {
    int lhs_dims = param_shapes[0].dimensions_size();
    int rhs_dims = param_shapes[1].dimensions_size();
    int out_dims = param_shapes[2].dimensions_size();

    // 检测不对称情况：lhs 3D, rhs 4D -> lhs 需要扩展
    if (lhs_dims == 3 && rhs_dims == 4) {
      dot_lhs_expand = true;
      NPU_LOG_FORMAT_PRINT(INFO, "Dot: lhs 3D, rhs 4D, will expand lhs to 4D");
    }
    // 检测不对称情况：lhs 4D, rhs 3D -> rhs 需要扩展
    if (lhs_dims == 4 && rhs_dims == 3) {
      dot_rhs_expand = true;
      NPU_LOG_FORMAT_PRINT(INFO, "Dot: lhs 4D, rhs 3D, will expand rhs to 4D");
    }
  }

  for (int i = 0; i < args_.size(); ++i) {
    void* device_addr = param_addrs[i];

    if ((thunk_info().op_code == HloOpcode::kSlice ||
         thunk_info().op_code == HloOpcode::kDynamicSlice) &&
        i == args_.size() - 1) {
      auto tensor_ptr = NpuTool::CreateAclTensorPtr(
          param_shapes[i], param_shapes[i].element_type(), device_addr);
      acl_tensor_ptr_vector.push_back(tensor_ptr);
    } else {
      auto tensor_ptr = NpuTool::CreateAclTensorPtr(
          param_shapes[i], param_shapes[i].element_type(), device_addr);
      acl_tensor_ptr_vector.push_back(tensor_ptr);
    }
  }

  return acl_tensor_ptr_vector;
}

absl::Status NpuAclnnThunk::ExecuteAclnn(aclrtStream& stream,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         AclnnExecFunc aclnnExecFunc) {
  aclnnStatus ret = 0;

  // 从 Workspace 内存池获取内存,替代动态申请
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    NPU_LOG_FORMAT_PRINT(DEBUG,
                         "Additional aclnn WorkspaceSize is %d, get from pool",
                         workspaceSize);

    auto* pool = NpuWorkspaceMemoryPool::GetInstance();
    if (pool == nullptr) {
      const std::string err_msg = "Workspace memory pool not available";
      NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
      return absl::InternalError(err_msg);
    }

    auto alloc_result = pool->AllocateForStream(stream, workspaceSize);
    if (!alloc_result.ok()) {
      const std::string err_msg =
          absl::StrFormat("Failed to allocate workspace from pool: %s",
                          alloc_result.status().ToString().c_str());
      NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
      return absl::InternalError(err_msg);
    }
    workspaceAddr = alloc_result.value();
  } else {
    NPU_LOG_FORMAT_PRINT(
        DEBUG, "Additional aclnn WorkspaceSize is 0, no need to apply");
  }

  // 调用 aclnnExecFunc 第二段接口
  ret = aclnnExecFunc(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    const std::string err_msg =
        absl::StrFormat("Execute aclnnExecFunc failed. Error code: %d", ret);
    NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
    return absl::InternalError(err_msg);
  }

  return absl::OkStatus();
}

absl::Status NpuAclnnThunk::PrintOutputTensor(
    const NpuBufferAllocations* allocations,
    const std::vector<Shape>& param_shapes, int device_ordinal) {
  // 对于当前aclnn处理的简单算子来说,只有最后一个参数为输出参数
  const auto& output_arg = args_.back();
  const auto& output_shape = param_shapes.back();

  auto addr = allocations->GetDeviceAddress(output_arg);
  void* device_addr = addr.opaque();

  // 使用统一的打印函数
  PrintParamInfo(0, "output", device_ordinal, output_arg, output_shape,
                 device_addr);

  return absl::OkStatus();
}

}  // namespace xla
