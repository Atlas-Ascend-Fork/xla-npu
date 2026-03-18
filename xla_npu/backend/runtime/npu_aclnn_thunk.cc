/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_op_interface.h"
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
      // Concatenate 是变长参数算子,至少需要 2 个输入 + 1 个输出 = 至少 3 个参数
      if (param_shapes.size() < 3) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "Concatenate param shape size %d is less than 3",
                             param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kReduce:
      // Reduce 操作: 输入 + 初始值 + 输出 = 3 个参数
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Reduce param shape size %d does not meet target value 3",
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
    NPU_LOG_FORMAT_PRINT(
        ERROR, "Stream need to be created or init before execute npu thunk.");
    return absl::InternalError("");
  }
  auto stream_handle = params.stream->platform_specific_handle();
  if (stream_handle.stream == nullptr) {
    NPU_LOG_FORMAT_PRINT(
        ERROR, "Stream need to be created or init before execute npu thunk.");
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

  TF_ASSIGN_OR_RETURN(auto acl_tensor_ptr_vector,
                      ExtractAclInputOutTensorPtr(allocations, param_shapes));

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
      ret = npu::aclrtMemcpy(output_device_addr, input_size, input_device_addr,
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

        input_tensor = npu::aclCreateTensor(
            expanded_shape.data(), expanded_shape.size(), data_type,
            expanded_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            expanded_shape.data(), expanded_shape.size(), device_addr);

        if (input_tensor == nullptr) {
          const std::string err_msg = "Failed to create expanded view tensor";
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }
      }

      aclIntArray* output_param_size = npu::aclCreateIntArray(
          output_param_size_data.data(), output_param_size_data.size());

      // 调用 aclnnExpand 第一段接口
      ret = npu::aclnnExpandGetWorkspaceSize(input_tensor, output_param_size,
                                             acl_tensor_ptr_vector.back(),
                                             &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnExpandGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnExpand);

      // Clean up the output_param_size array
      npu::aclDestroyIntArray(output_param_size);
      break;
    }

    case HloOpcode::kAdd: {
      float alphaValue = 1.0;
      aclScalar* alpha =
          npu::aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

      // 调用aclnnAdd第一段接口
      ret = npu::aclnnAddGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1], alpha,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnAddGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnAdd);
      break;
    }

    case HloOpcode::kTanh: {
      ret = npu::aclnnTanhGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                           acl_tensor_ptr_vector.back(),
                                           &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnTanhGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnTanh);
      break;
    }

    case HloOpcode::kMultiply: {
      ret = npu::aclnnMulGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnMulGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnMul);
      break;
    }

    case HloOpcode::kDot: {
      int8_t cubeMathType = 0;
      ret = npu::aclnnMatmulGetWorkspaceSize(
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
          ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnMatmul);
      break;
    }

    case HloOpcode::kDivide: {
      ret = npu::aclnnDivGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnDivGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnDiv);
      break;
    }

    case HloOpcode::kSubtract: {
      float alphaValue = 1.0;
      aclScalar* alpha =
          npu::aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

      ret = npu::aclnnSubGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1], alpha,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSubGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnSub);
      break;
    }

    case HloOpcode::kNegate: {
      ret = npu::aclnnNegGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnNegGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnNeg);
      break;
    }

    case HloOpcode::kExp: {
      ret = npu::aclnnExpGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnExpGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnExp);
      break;
    }

    case HloOpcode::kMaximum: {
      ret = npu::aclnnMaximumGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnMaximumGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnMaximum);
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
          npu::aclCreateTensorList(input_tensors.data(), input_tensors.size());
      if (tensor_list == nullptr) {
        const std::string err_msg =
            "Failed to create aclTensorList for concatenate";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 调用 aclnnCatGetWorkspaceSize
      ret = npu::aclnnCatGetWorkspaceSize(tensor_list, dim,
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnCatGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnCat);

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
      aclIntArray* dims = npu::aclCreateIntArray(
          thunk_info().dimensions.data(), thunk_info().dimensions.size());
      if (dims == nullptr) {
        const std::string err_msg =
            "Failed to create aclIntArray for reduce dimensions";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 获取输出数据类型
      aclDataType out_dtype = NpuTool::ConvertXlaType2AclDataType(
          param_shapes.back().element_type());

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
        npu::aclDestroyIntArray(dims);
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
        ret = npu::aclnnAmaxGetWorkspaceSize(acl_tensor_ptr_vector[0], dims,
                                             false, acl_tensor_ptr_vector[2],
                                             &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          const std::string err_msg = absl::StrFormat(
              "Call aclnnAmaxGetWorkspaceSize failed. Error code: %d", ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          npu::aclDestroyIntArray(dims);
          return absl::InternalError(err_msg);
        }

        exec_ret =
            ExecuteAclnn(acl_stream, workspaceSize, executor, npu::aclnnAmax);
        if (!exec_ret.ok()) {
          NPU_LOG_FORMAT_PRINT(ERROR, "ExecuteAclnn for aclnnAmax failed: %s",
                               exec_ret.ToString().c_str());
          npu::aclDestroyIntArray(dims);
          return exec_ret;
        }
      } else {
        // 默认使用 ReduceSum
        // 调用 aclnnReduceSum 第一段接口
        // aclnnReduceSum 接口参数: input, dims, keepDims, dtype, output,
        // workspaceSize, executor
        ret = npu::aclnnReduceSumGetWorkspaceSize(
            acl_tensor_ptr_vector[0], dims, false, out_dtype,
            acl_tensor_ptr_vector[2], &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          const std::string err_msg = absl::StrFormat(
              "Call aclnnReduceSumGetWorkspaceSize failed. Error code: %d",
              ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          npu::aclDestroyIntArray(dims);
          return absl::InternalError(err_msg);
        }

        exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                npu::aclnnReduceSum);
      }

      // 清理维度数组
      npu::aclDestroyIntArray(dims);
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

  PrintOutputTensor(allocations, param_shapes);

  return exec_ret;
}

absl::StatusOr<std::vector<aclTensor*>>
NpuAclnnThunk::ExtractAclInputOutTensorPtr(
    const NpuBufferAllocations* allocations,
    const std::vector<Shape>& param_shapes) {
  std::vector<aclTensor*> acl_tensor_ptr_vector;

  for (int i = 0; i < args_.size(); ++i) {
    auto addr = allocations->GetDeviceAddress(args_[i]);
    void* device_addr = addr.opaque();
    auto tensor_ptr = NpuTool::CreateAclTensorPtr(
        param_shapes[i], param_shapes[i].element_type(), device_addr);
    acl_tensor_ptr_vector.push_back(tensor_ptr);

    if (i < args_.size() - 1) {
      // 只能执行前打印,因为算子执行会复用内存,执行结果会覆盖到入参地址上
      NPU_LOG_FORMAT_PRINT(DEBUG, "---- Param(input) %d info: ----", i);
      NPU_LOG_FORMAT_PRINT(DEBUG, "  Buffer allocation: %v",
                           args_[i].ToString());
      NPU_LOG_FORMAT_PRINT(DEBUG, "  Tensor Shape: %v",
                           param_shapes[i].ToString(true));
      NPU_LOG_FORMAT_PRINT(DEBUG, "  Device addr: %p", device_addr);
      NpuTool::PrintNpuDeviceMemoryValue<float>(
          "  Value on device:", NpuTool::GetXlaShapeSize(param_shapes[i]),
          device_addr);
    }
    // 最后一个追加的tensor_ptr为出参tensor
  }

  return acl_tensor_ptr_vector;
}

absl::Status NpuAclnnThunk::ExecuteAclnn(aclrtStream& stream,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         AclnnExecFunc aclnnExecFunc) {
  aclnnStatus ret = 0;

  // 根据第一段接口计算出的 workspaceSize 申请 device 内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    NPU_LOG_FORMAT_PRINT(DEBUG,
                         "Additional aclnn WorkspaceSize is %d, need to apply",
                         workspaceSize);
    ret = npu::aclrtMalloc(&workspaceAddr, workspaceSize,
                           ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      const std::string err_msg =
          absl::StrFormat("Allocate workspace failed. Error code: %d", ret);
      NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
      return absl::InternalError(err_msg);
    }
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
    const std::vector<Shape>& param_shapes) {
  // 对于当前aclnn处理的简单算子来说,只有最后一个参数为输出参数
  const auto& output_arg = args_.back();
  const auto& outout_arg_param = param_shapes.back();

  auto addr = allocations->GetDeviceAddress(output_arg);
  void* device_addr = addr.opaque();
  NPU_LOG_FORMAT_PRINT(DEBUG, "---- Param(output) 0 info: ----");
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Buffer allocation: %v", output_arg.ToString());
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Tensor Shape: %v",
                       outout_arg_param.ToString(true));
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Device addr: %p", device_addr);
  NpuTool::PrintNpuDeviceMemoryValueByShape(
      "  Value on device:", outout_arg_param, device_addr);

  return absl::OkStatus();
}

}  // namespace xla
