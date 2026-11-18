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
    case HloOpcode::kSqrt:
    case HloOpcode::kSin:
    case HloOpcode::kCos:
    case HloOpcode::kLogistic:
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
    case HloOpcode::kPower:
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
    case HloOpcode::kCustomCall:
      if (param_shapes.size() != 2) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "ArgMax custom call expects 2 parameters, got %d",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kIota:
      // Iota only writes its output buffer.
      if (param_shapes.size() != 1) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Iota param shape size %d does not meet target value 1",
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
    case HloOpcode::kDynamicUpdateSlice:
      // operand + update + start_indices... + output
      if (param_shapes.size() < 4) {
        NPU_LOG_FORMAT_PRINT(
            ERROR,
            "DynamicUpdateSlice param shape size %d is less than 4",
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
    case HloOpcode::kOr:
      // Or (BitwiseOr/LogicalOr) 操作：input1 + input2 + output = 3 个参数
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Or param shape size %d does not meet target value 3",
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
    case HloOpcode::kConvert:
      // Convert operation: input + output = 2 parameters
      if (param_shapes.size() != 2) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "Convert param shape size %d does not meet target "
                             "value 2",
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
                         "Stream needs to be created or initialized before "
                         "executing npu thunk.");
    return absl::InternalError("");
  }
  auto stream_handle = params.stream->platform_specific_handle();
  if (stream_handle.stream == nullptr) {
    NPU_LOG_FORMAT_PRINT(ERROR,
                         "Stream needs to be created or initialized before "
                         "executing npu thunk.");
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
    case HloOpcode::kCustomCall: {
      const std::string& desc = thunk_info().hloInstrDesc;
      if (desc.find("npu$copy") != std::string::npos) {
        const int64_t bytes =
            NpuTool::GetXlaShapeSize(param_shapes[0]) *
            ShapeUtil::ByteSizeOfPrimitiveType(param_shapes[0].element_type());
        void* src = allocations->GetDeviceAddress(args_[0]).opaque();
        void* dst = allocations->GetDeviceAddress(args_.back()).opaque();
        ret = dep::aclrtMemcpy(dst, bytes, src, bytes,
                               ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
          return absl::InternalError(absl::StrFormat(
              "npu$copy memcpy failed. Error code: %d", ret));
        }
        exec_ret = absl::OkStatus();
        break;
      }
      int64_t dim = 0;
      const size_t pos = desc.find("axis=");
      if (pos != std::string::npos) {
        dim = std::stoll(desc.substr(pos + strlen("axis=")));
      }
      ret = dep::aclnnArgMaxGetWorkspaceSize(
          acl_tensor_ptr_vector[0], dim, true,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        return absl::InternalError(absl::StrFormat(
            "Call aclnnArgMaxGetWorkspaceSize failed. Error code: %d", ret));
      }
      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnArgMax);
      break;
    }

    case HloOpcode::kIota: {
      const Shape& output_shape = param_shapes.back();
      const std::vector<int64_t> dims(output_shape.dimensions().begin(),
                                      output_shape.dimensions().end());
      int64_t iota_dim = 0;
      const std::string& iota_desc = thunk_info().hloInstrDesc;
      const size_t pos = iota_desc.find("iota_dimension=");
      if (pos != std::string::npos) {
        iota_dim = std::stoll(iota_desc.substr(pos + strlen("iota_dimension=")));
      }
      if (iota_dim < 0 || iota_dim >= static_cast<int64_t>(dims.size())) {
        return absl::InternalError("Iota dimension is out of range");
      }
      std::vector<int64_t> strides(dims.size(), 1);
      int64_t stride = 1;
      for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= dims[i];
      }
      const int64_t count = NpuTool::GetXlaShapeSize(output_shape);
      const size_t elem_bytes =
          ShapeUtil::ByteSizeOfPrimitiveType(output_shape.element_type());
      std::vector<char> host(static_cast<size_t>(count) * elem_bytes, 0);
      for (int64_t linear = 0; linear < count; ++linear) {
        const int64_t value = (linear / strides[iota_dim]) % dims[iota_dim];
        char* dst = host.data() + static_cast<size_t>(linear) * elem_bytes;
        switch (output_shape.element_type()) {
          case PrimitiveType::S8:
            *reinterpret_cast<int8_t*>(dst) = static_cast<int8_t>(value);
            break;
          case PrimitiveType::U8:
            *reinterpret_cast<uint8_t*>(dst) = static_cast<uint8_t>(value);
            break;
          case PrimitiveType::S16:
            *reinterpret_cast<int16_t*>(dst) = static_cast<int16_t>(value);
            break;
          case PrimitiveType::U16:
            *reinterpret_cast<uint16_t*>(dst) = static_cast<uint16_t>(value);
            break;
          case PrimitiveType::S32:
            *reinterpret_cast<int32_t*>(dst) = static_cast<int32_t>(value);
            break;
          case PrimitiveType::U32:
            *reinterpret_cast<uint32_t*>(dst) = static_cast<uint32_t>(value);
            break;
          case PrimitiveType::S64:
            *reinterpret_cast<int64_t*>(dst) = value;
            break;
          case PrimitiveType::U64:
            *reinterpret_cast<uint64_t*>(dst) = static_cast<uint64_t>(value);
            break;
          case PrimitiveType::F32:
            *reinterpret_cast<float*>(dst) = static_cast<float>(value);
            break;
          case PrimitiveType::F64:
            *reinterpret_cast<double*>(dst) = static_cast<double>(value);
            break;
          default:
            return absl::UnimplementedError(
                "Iota currently supports integer and f32/f64 element types");
        }
      }
      void* output_addr = allocations->GetDeviceAddress(args_.back()).opaque();
      ret = dep::aclrtMemcpy(output_addr, host.size(), host.data(), host.size(),
                             ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_SUCCESS) {
        return absl::InternalError(
            absl::StrFormat("Failed to upload iota buffer. Error code: %d",
                            ret));
      }
      exec_ret = absl::OkStatus();
      break;
    }

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
        NPU_LOG_PRINT(ERROR, err_msg);
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

      // Scalar broadcasts cannot be expressed as a stride-0 view for ACLNN
      // Expand. Materialize them on the host for modest sizes.
      if (NpuTool::GetXlaShapeSize(param_shapes[0]) == 1) {
        const Shape& input_shape = param_shapes[0];
        const size_t elem_bytes =
            ShapeUtil::ByteSizeOfPrimitiveType(input_shape.element_type());
        const int64_t count = NpuTool::GetXlaShapeSize(param_shapes.back());
        constexpr int64_t kMaxHostBroadcastElements = 16 * 1024 * 1024;
        if (count > 0 && count <= kMaxHostBroadcastElements) {
          std::vector<char> scalar_bytes(elem_bytes);
          auto input_addr = allocations->GetDeviceAddress(args_[0]).opaque();
          ret = dep::aclrtMemcpy(scalar_bytes.data(), elem_bytes, input_addr,
                                 elem_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
          if (ret != ACL_SUCCESS) {
            return absl::InternalError(absl::StrFormat(
                "Failed to read broadcast scalar. Error code: %d", ret));
          }
          std::vector<char> host(static_cast<size_t>(count) * elem_bytes);
          for (int64_t i = 0; i < count; ++i) {
            std::memcpy(host.data() + static_cast<size_t>(i) * elem_bytes,
                        scalar_bytes.data(), elem_bytes);
          }
          void* output_addr =
              allocations->GetDeviceAddress(args_.back()).opaque();
          ret = dep::aclrtMemcpy(output_addr, host.size(), host.data(),
                                 host.size(), ACL_MEMCPY_HOST_TO_DEVICE);
          if (ret != ACL_SUCCESS) {
            return absl::InternalError(absl::StrFormat(
                "Failed to fill broadcast scalar. Error code: %d", ret));
          }
          exec_ret = absl::OkStatus();
          break;
        }
      }

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
          NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnExpand);

      // Clean up the output_param_size array
      dep::aclDestroyIntArray(output_param_size);
      break;
    }

    case HloOpcode::kAdd: {
      // ACLNN Add has no UINT32 kernel; the values used here (indices,
      // positions) fit in INT32, so reinterpret the buffers as INT32 views.
      const bool add_is_u32 =
          param_shapes[0].element_type() == PrimitiveType::U32;
      aclTensor* add_lhs = acl_tensor_ptr_vector[0];
      aclTensor* add_rhs = acl_tensor_ptr_vector[1];
      aclTensor* add_out = acl_tensor_ptr_vector.back();
      if (add_is_u32) {
        add_lhs = NpuTool::CreateAclTensorPtr(
            param_shapes[0], PrimitiveType::S32,
            allocations->GetDeviceAddress(args_[0]).opaque());
        add_rhs = NpuTool::CreateAclTensorPtr(
            param_shapes[1], PrimitiveType::S32,
            allocations->GetDeviceAddress(args_[1]).opaque());
        add_out = NpuTool::CreateAclTensorPtr(
            param_shapes.back(), PrimitiveType::S32,
            allocations->GetDeviceAddress(args_.back()).opaque());
        if (add_lhs == nullptr || add_rhs == nullptr || add_out == nullptr) {
          if (add_lhs != nullptr) dep::aclDestroyTensor(add_lhs);
          if (add_rhs != nullptr) dep::aclDestroyTensor(add_rhs);
          if (add_out != nullptr) dep::aclDestroyTensor(add_out);
          return absl::InternalError("Failed to create U32 add views");
        }
      }

      // 优化：对于 RMSNorm 中的 variance + epsilon 场景
      // 检测输入类型，如果是 FP16/BF16，尝试使用 FP32 计算以减少精度损失
      aclDataType input_type =
          add_is_u32
              ? ACL_INT32
              : NpuTool::ConvertXlaType2AclDataType(
                    param_shapes[0].element_type());

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
          add_lhs, add_rhs, alpha, add_out, &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnAddGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_PRINT(ERROR, err_msg);
        dep::aclDestroyScalar(alpha);
        if (add_is_u32) {
          dep::aclDestroyTensor(add_lhs);
          dep::aclDestroyTensor(add_rhs);
          dep::aclDestroyTensor(add_out);
        }
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnAdd);

      dep::aclDestroyScalar(alpha);
      if (add_is_u32) {
        dep::aclDestroyTensor(add_lhs);
        dep::aclDestroyTensor(add_rhs);
        dep::aclDestroyTensor(add_out);
      }
      break;
    }

    case HloOpcode::kPower: {
      ret = dep::aclnnPowTensorTensorGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnPowTensorTensorGetWorkspaceSize failed. Error code: %d",
            ret);
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnPowTensorTensor);
      break;
    }

    case HloOpcode::kLogistic: {
      ret = dep::aclnnSigmoidGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                              acl_tensor_ptr_vector.back(),
                                              &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSigmoidGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnSigmoid);
      break;
    }

    case HloOpcode::kTanh: {
      ret = dep::aclnnTanhGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                           acl_tensor_ptr_vector.back(),
                                           &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnTanhGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnMul);
      break;
    }

    case HloOpcode::kDot: {
      const std::string& dot_desc = thunk_info().hloInstrDesc;
      auto parse_dim_list = [&dot_desc](const char* key) {
        std::vector<int64_t> dims;
        const size_t key_pos = dot_desc.find(key);
        if (key_pos == std::string::npos) return dims;
        const size_t open = dot_desc.find('{', key_pos);
        if (open == std::string::npos) return dims;
        const size_t close = dot_desc.find('}', open);
        if (close == std::string::npos) return dims;
        const std::string body = dot_desc.substr(open + 1, close - open - 1);
        size_t pos = 0;
        while (pos < body.size()) {
          const size_t comma = body.find(',', pos);
          const std::string part =
              body.substr(pos, comma == std::string::npos ? std::string::npos
                                                          : comma - pos);
          try {
            dims.push_back(std::stoll(part));
          } catch (...) {
          }
          if (comma == std::string::npos) break;
          pos = comma + 1;
        }
        return dims;
      };
      const std::vector<int64_t> lhs_batch = parse_dim_list("lhs_batch_dims");
      const std::vector<int64_t> rhs_batch = parse_dim_list("rhs_batch_dims");
      const std::vector<int64_t> lhs_contracting =
          parse_dim_list("lhs_contracting_dims");
      const std::vector<int64_t> rhs_contracting =
          parse_dim_list("rhs_contracting_dims");
      const Shape& lhs_shape = param_shapes[0];
      const Shape& rhs_shape = param_shapes[1];
      const Shape& out_shape = param_shapes.back();
      const int lhs_rank = lhs_shape.dimensions().size();
      const int rhs_rank = rhs_shape.dimensions().size();
      const int out_rank = out_shape.dimensions().size();

      const bool is_batched = !lhs_batch.empty() || !rhs_batch.empty() ||
                              lhs_rank > 2 || rhs_rank > 2;
      if (!is_batched) {
        // Fast path for 2D dots: pass a strided view instead of copying when
        // the contracting dimension is not the matmul innermost one.
        auto make_transposed_view = [&](int arg_index) -> aclTensor* {
          const Shape& shape = param_shapes[arg_index];
          if (!shape.has_layout() || shape.layout().minor_to_major(0) != 1 ||
              shape.layout().minor_to_major(1) != 0) {
            return nullptr;
          }
          const int64_t d0 = shape.dimensions(0);
          const int64_t d1 = shape.dimensions(1);
          const std::vector<int64_t> view_dims = {d1, d0};
          const std::vector<int64_t> view_strides = {1, d1};
          const std::vector<int64_t> storage_dims = {d0, d1};
          void* addr = allocations->GetDeviceAddress(args_[arg_index]).opaque();
          return dep::aclCreateTensor(
              view_dims.data(), view_dims.size(),
              NpuTool::ConvertXlaType2AclDataType(shape.element_type()),
              view_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
              storage_dims.data(), storage_dims.size(), addr);
        };
        aclTensor* lhs = acl_tensor_ptr_vector[0];
        aclTensor* rhs = acl_tensor_ptr_vector[1];
        aclTensor* lhs_view = nullptr;
        aclTensor* rhs_view = nullptr;
        const int64_t lhs_contract =
            lhs_contracting.empty() ? -1 : lhs_contracting[0];
        const int64_t rhs_contract =
            rhs_contracting.empty() ? -1 : rhs_contracting[0];
        bool unsupported_dot = false;
        if (lhs_contract == 0) {
          lhs_view = make_transposed_view(0);
          if (lhs_view == nullptr) unsupported_dot = true;
          else lhs = lhs_view;
        }
        if (rhs_contract == 1) {
          rhs_view = make_transposed_view(1);
          if (rhs_view == nullptr) unsupported_dot = true;
          else rhs = rhs_view;
        }
        if (unsupported_dot || (lhs_contract != 0 && lhs_contract != 1) ||
            (rhs_contract != 0 && rhs_contract != 1)) {
          if (lhs_view != nullptr) dep::aclDestroyTensor(lhs_view);
          if (rhs_view != nullptr) dep::aclDestroyTensor(rhs_view);
          return absl::UnimplementedError(absl::StrFormat(
              "Unsupported Dot contracting dims: %s", dot_desc.c_str()));
        }
        int8_t cubeMathType = 0;
        ret = dep::aclnnMatmulGetWorkspaceSize(
            lhs, rhs, acl_tensor_ptr_vector.back(), cubeMathType,
            &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          if (lhs_view != nullptr) dep::aclDestroyTensor(lhs_view);
          if (rhs_view != nullptr) dep::aclDestroyTensor(rhs_view);
          const std::string err_msg = absl::StrFormat(
              "Call aclnnMatmulGetWorkspaceSize failed. Error code: %d", ret);
          NPU_LOG_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }
        exec_ret =
            ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnMatmul);
        if (lhs_view != nullptr) dep::aclDestroyTensor(lhs_view);
        if (rhs_view != nullptr) dep::aclDestroyTensor(rhs_view);
        break;
      }

      // Batched dot: permute each operand into [B, M, K] / [B, K, N] views,
      // run a batched aclnnMatmul, and write directly into the XLA output
      // buffer (whose row-major flattening is exactly [B, M, N]).
      if (lhs_contracting.size() != 1 || rhs_contracting.size() != 1 ||
          lhs_batch.size() != rhs_batch.size()) {
        return absl::UnimplementedError(absl::StrFormat(
            "Unsupported batched Dot: %s", dot_desc.c_str()));
      }
      const int64_t lhs_contract = lhs_contracting[0];
      const int64_t rhs_contract = rhs_contracting[0];
      int64_t batch = 1;
      std::vector<int64_t> expected;
      auto is_in = [](const std::vector<int64_t>& list, int64_t v) {
        for (int64_t x : list) {
          if (x == v) return true;
        }
        return false;
      };
      for (size_t i = 0; i < lhs_batch.size(); ++i) {
        const int64_t ld = lhs_batch[i];
        const int64_t rd = rhs_batch[i];
        if (lhs_shape.dimensions(ld) != rhs_shape.dimensions(rd)) {
          return absl::UnimplementedError(
              "Batched Dot with mismatched batch dimensions");
        }
        batch *= lhs_shape.dimensions(ld);
        expected.push_back(lhs_shape.dimensions(ld));
      }
      const int64_t K = lhs_shape.dimensions(lhs_contract);
      if (rhs_shape.dimensions(rhs_contract) != K) {
        return absl::UnimplementedError(
            "Batched Dot with mismatched contracting dimensions");
      }
      std::vector<int64_t> lhs_free;
      for (int d = 0; d < lhs_rank; ++d) {
        if (!is_in(lhs_batch, d) && d != lhs_contract) lhs_free.push_back(d);
      }
      std::vector<int64_t> rhs_free;
      for (int d = 0; d < rhs_rank; ++d) {
        if (!is_in(rhs_batch, d) && d != rhs_contract) rhs_free.push_back(d);
      }
      int64_t M = 1;
      for (int64_t d : lhs_free) {
        M *= lhs_shape.dimensions(d);
        expected.push_back(lhs_shape.dimensions(d));
      }
      int64_t N = 1;
      for (int64_t d : rhs_free) {
        N *= rhs_shape.dimensions(d);
        expected.push_back(rhs_shape.dimensions(d));
      }
      if (static_cast<int>(expected.size()) != out_rank) {
        return absl::UnimplementedError(
            "Batched Dot output rank is not supported");
      }
      for (int i = 0; i < out_rank; ++i) {
        if (expected[i] != out_shape.dimensions(i)) {
          return absl::UnimplementedError(
              "Batched Dot output shape is not supported");
        }
      }
      if (out_shape.has_layout()) {
        for (int i = 0; i < out_rank; ++i) {
          if (out_shape.layout().minor_to_major(i) != out_rank - 1 - i) {
            return absl::UnimplementedError(
                "Batched Dot requires a row-major output layout");
          }
        }
      }

      std::vector<int64_t> lhs_perm = lhs_batch;
      lhs_perm.insert(lhs_perm.end(), lhs_free.begin(), lhs_free.end());
      lhs_perm.push_back(lhs_contract);
      std::vector<int64_t> rhs_perm = rhs_batch;
      rhs_perm.push_back(rhs_contract);
      rhs_perm.insert(rhs_perm.end(), rhs_free.begin(), rhs_free.end());

      auto permute_operand = [&](int arg_index, const Shape& shape,
                                 const std::vector<int64_t>& perm,
                                 aclTensor** out_tensor,
                                 void** out_buffer) -> absl::Status {
        *out_tensor = nullptr;
        *out_buffer = nullptr;
        const std::vector<int64_t> dims =
            NpuTool::ConvertXlaShape2VectorShape(shape);
        std::vector<int64_t> permuted_dims;
        for (int64_t d : perm) permuted_dims.push_back(dims[d]);
        std::vector<int64_t> strides(permuted_dims.size(), 1);
        int64_t stride = 1;
        for (int i = static_cast<int>(permuted_dims.size()) - 1; i >= 0; --i) {
          strides[i] = stride;
          stride *= permuted_dims[i];
        }
        const int64_t count = NpuTool::GetXlaShapeSize(shape);
        const size_t bytes =
            count * ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
        void* buffer = nullptr;
        aclnnStatus alloc_ret =
            dep::aclrtMalloc(&buffer, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (alloc_ret != ACL_SUCCESS) {
          return absl::InternalError(
              absl::StrFormat("Failed to allocate Dot permute buffer. Error "
                              "code: %d",
                              alloc_ret));
        }
        aclTensor* tensor = dep::aclCreateTensor(
            permuted_dims.data(), permuted_dims.size(),
            NpuTool::ConvertXlaType2AclDataType(shape.element_type()),
            strides.data(), 0, aclFormat::ACL_FORMAT_ND, permuted_dims.data(),
            permuted_dims.size(), buffer);
        if (tensor == nullptr) {
          dep::aclrtFree(buffer);
          return absl::InternalError("Failed to create Dot permute tensor");
        }
        aclIntArray* perm_array =
            dep::aclCreateIntArray(perm.data(), perm.size());
        if (perm_array == nullptr) {
          dep::aclDestroyTensor(tensor);
          dep::aclrtFree(buffer);
          return absl::InternalError("Failed to create Dot permute array");
        }
        uint64_t permute_workspace = 0;
        aclOpExecutor* permute_executor = nullptr;
        aclnnStatus permute_ret = dep::aclnnPermuteGetWorkspaceSize(
            acl_tensor_ptr_vector[arg_index], perm_array, tensor,
            &permute_workspace, &permute_executor);
        absl::Status status =
            permute_ret == ACL_SUCCESS
                ? ExecuteAclnn(acl_stream, permute_workspace, permute_executor,
                               dep::aclnnPermute)
                : absl::InternalError(absl::StrFormat(
                      "aclnnPermuteGetWorkspaceSize failed. Error code: %d",
                      permute_ret));
        dep::aclDestroyIntArray(perm_array);
        if (!status.ok()) {
          dep::aclDestroyTensor(tensor);
          dep::aclrtFree(buffer);
          return status;
        }
        *out_tensor = tensor;
        *out_buffer = buffer;
        return absl::OkStatus();
      };

      aclTensor* lhs_permuted = nullptr;
      aclTensor* rhs_permuted = nullptr;
      void* lhs_buffer = nullptr;
      void* rhs_buffer = nullptr;
      absl::Status permute_status =
          permute_operand(0, lhs_shape, lhs_perm, &lhs_permuted, &lhs_buffer);
      if (permute_status.ok()) {
        permute_status = permute_operand(1, rhs_shape, rhs_perm, &rhs_permuted,
                                         &rhs_buffer);
      }
      if (!permute_status.ok()) {
        if (lhs_permuted != nullptr) dep::aclDestroyTensor(lhs_permuted);
        if (rhs_permuted != nullptr) dep::aclDestroyTensor(rhs_permuted);
        if (lhs_buffer != nullptr) dep::aclrtFree(lhs_buffer);
        if (rhs_buffer != nullptr) dep::aclrtFree(rhs_buffer);
        return permute_status;
      }

      auto make_view = [&](const std::vector<int64_t>& dims, void* buffer,
                           PrimitiveType type) -> aclTensor* {
        std::vector<int64_t> strides(dims.size(), 1);
        int64_t stride = 1;
        for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
          strides[i] = stride;
          stride *= dims[i];
        }
        return dep::aclCreateTensor(
            dims.data(), dims.size(), NpuTool::ConvertXlaType2AclDataType(type),
            strides.data(), 0, aclFormat::ACL_FORMAT_ND, dims.data(),
            dims.size(), buffer);
      };
      const std::vector<int64_t> lhs_view_dims = {batch, M, K};
      const std::vector<int64_t> rhs_view_dims = {batch, K, N};
      const std::vector<int64_t> out_view_dims = {batch, M, N};
      aclTensor* lhs_view =
          make_view(lhs_view_dims, lhs_buffer, lhs_shape.element_type());
      aclTensor* rhs_view =
          make_view(rhs_view_dims, rhs_buffer, rhs_shape.element_type());
      void* output_buffer =
          allocations->GetDeviceAddress(args_.back()).opaque();
      aclTensor* out_view =
          make_view(out_view_dims, output_buffer, out_shape.element_type());
      if (lhs_view == nullptr || rhs_view == nullptr || out_view == nullptr) {
        if (lhs_view != nullptr) dep::aclDestroyTensor(lhs_view);
        if (rhs_view != nullptr) dep::aclDestroyTensor(rhs_view);
        if (out_view != nullptr) dep::aclDestroyTensor(out_view);
        dep::aclDestroyTensor(lhs_permuted);
        dep::aclDestroyTensor(rhs_permuted);
        dep::aclrtFree(lhs_buffer);
        dep::aclrtFree(rhs_buffer);
        return absl::InternalError("Failed to create batched Dot views");
      }

      int8_t batched_cube_math_type = 0;
      ret = dep::aclnnMatmulGetWorkspaceSize(
          lhs_view, rhs_view, out_view, batched_cube_math_type, &workspaceSize,
          &executor);
      absl::Status matmul_status =
          ret == ACL_SUCCESS
              ? absl::OkStatus()
              : absl::InternalError(absl::StrFormat(
                    "Call aclnnMatmulGetWorkspaceSize (batched) failed. Error "
                    "code: %d",
                    ret));
      if (matmul_status.ok()) {
        exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                dep::aclnnMatmul);
        matmul_status = exec_ret;
      }
      dep::aclrtSynchronizeStream(acl_stream);
      dep::aclDestroyTensor(lhs_view);
      dep::aclDestroyTensor(rhs_view);
      dep::aclDestroyTensor(out_view);
      dep::aclDestroyTensor(lhs_permuted);
      dep::aclDestroyTensor(rhs_permuted);
      dep::aclrtFree(lhs_buffer);
      dep::aclrtFree(rhs_buffer);
      if (!matmul_status.ok()) return matmul_status;
      break;
    }

    case HloOpcode::kDivide: {
      ret = dep::aclnnDivGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnDivGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnExp);
      break;
    }

    case HloOpcode::kSqrt: {
      ret = dep::aclnnSqrtGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                           acl_tensor_ptr_vector.back(),
                                           &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSqrtGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSqrt);
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 调用 aclnnCatGetWorkspaceSize
      ret = dep::aclnnCatGetWorkspaceSize(tensor_list, dim,
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnCatGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 创建维度数组
      aclIntArray* dims = dep::aclCreateIntArray(
          thunk_info().dimensions.data(), thunk_info().dimensions.size());
      if (dims == nullptr) {
        const std::string err_msg =
            "Failed to create aclIntArray for reduce dimensions";
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 获取输入数据类型
      aclDataType input_dtype =
          NpuTool::ConvertXlaType2AclDataType(param_shapes[0].element_type());

      // aclnnReduceSum's dtype argument must match the output tensor dtype.
      aclDataType accumulate_dtype = NpuTool::ConvertXlaType2AclDataType(
          param_shapes[2].element_type());

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
        NPU_LOG_PRINT(ERROR, err_msg);
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
          NPU_LOG_PRINT(ERROR, err_msg);
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
          NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 创建 aclIntArray 作为 permute 维度
      aclIntArray* perm = dep::aclCreateIntArray(
          thunk_info().dimensions.data(), thunk_info().dimensions.size());
      if (perm == nullptr) {
        const std::string err_msg = "Failed to create perm array for transpose";
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 调用 aclnnPermuteGetWorkspaceSize
      ret = dep::aclnnPermuteGetWorkspaceSize(acl_tensor_ptr_vector[0], perm,
                                              acl_tensor_ptr_vector.back(),
                                              &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnPermuteGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
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

        // XLA clamps dynamic slice start indices into the valid range.
        if (i < input_shape.dimensions().size()) {
          const int64_t dim = input_shape.dimensions()[i];
          int64_t max_start = dim - slice_sizes_logical[i];
          if (max_start < 0) max_start = 0;
          if (start_val < 0) start_val = 0;
          if (start_val > max_start) start_val = max_start;
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSliceV2);
      break;
    }

    case HloOpcode::kDynamicUpdateSlice: {
      // XLA DUS writes a rectangular update into a dynamically positioned
      // slice of operand. Lower it to a copy of operand into output followed
      // by layout-aware device-to-device block copies of update.
      if (param_shapes.size() < 4) {
        return absl::InternalError(
            "DynamicUpdateSlice expects at least 4 parameters");
      }
      const Shape& operand_shape = param_shapes[0];
      const Shape& update_shape = param_shapes[1];
      const Shape& output_shape = param_shapes.back();
      const int rank = operand_shape.dimensions().size();
      const int k_starts = param_shapes.size() - 3;
      if (!operand_shape.IsArray() || !update_shape.IsArray() ||
          !output_shape.IsArray() || rank == 0 || k_starts != rank ||
          update_shape.dimensions().size() != rank ||
          operand_shape.dimensions() != output_shape.dimensions()) {
        return absl::UnimplementedError(
            "DynamicUpdateSlice currently supports array operands and updates "
            "with the same non-zero rank");
      }

      // Make sure pending producers of the start indices have completed
      // before reading them back to the host.
      ret = dep::aclrtSynchronizeStream(acl_stream);
      if (ret != ACL_SUCCESS) {
        return absl::InternalError(
            absl::StrFormat("aclrtSynchronizeStream failed. Error code: %d",
                            ret));
      }

      auto strides_of = [](const Shape& shape) {
        const int r = shape.dimensions().size();
        std::vector<int64_t> strides(r, 1);
        int64_t stride = 1;
        if (shape.has_layout()) {
          for (int i = 0; i < r; ++i) {
            const int d = shape.layout().minor_to_major(i);
            strides[d] = stride;
            stride *= shape.dimensions(d);
          }
        } else {
          for (int d = r - 1; d >= 0; --d) {
            strides[d] = stride;
            stride *= shape.dimensions(d);
          }
        }
        return strides;
      };
      const auto operand_strides = strides_of(operand_shape);
      const auto update_strides = strides_of(update_shape);
      const int64_t elem_bytes =
          ShapeUtil::ByteSizeOfPrimitiveType(operand_shape.element_type());

      std::vector<int64_t> starts(rank, 0);
      for (int d = 0; d < rank; ++d) {
        const int arg_index = 2 + d;
        const Shape& start_shape = param_shapes[arg_index];
        auto start_addr = allocations->GetDeviceAddress(args_[arg_index]);
        const size_t bytes =
            ShapeUtil::ByteSizeOfPrimitiveType(start_shape.element_type());
        if (bytes > sizeof(int64_t)) {
          return absl::UnimplementedError(
              "DynamicUpdateSlice start index wider than 64 bits");
        }
        int64_t raw = 0;
        ret = dep::aclrtMemcpy(&raw, sizeof(raw), start_addr.opaque(), bytes,
                               ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
          return absl::InternalError(absl::StrFormat(
              "Failed to read DynamicUpdateSlice start index. Error code: %d",
              ret));
        }
        int64_t value = 0;
        switch (start_shape.element_type()) {
          case PrimitiveType::U8:
            value = static_cast<uint8_t>(raw);
            break;
          case PrimitiveType::S8:
            value = static_cast<int8_t>(raw);
            break;
          case PrimitiveType::U16:
            value = static_cast<uint16_t>(raw);
            break;
          case PrimitiveType::S16:
            value = static_cast<int16_t>(raw);
            break;
          case PrimitiveType::U32:
            value = static_cast<uint32_t>(raw);
            break;
          case PrimitiveType::S32:
            value = static_cast<int32_t>(raw);
            break;
          case PrimitiveType::U64:
            value = static_cast<int64_t>(static_cast<uint64_t>(raw));
            break;
          case PrimitiveType::S64:
            value = raw;
            break;
          default:
            return absl::UnimplementedError(
                "DynamicUpdateSlice start index must be an integer");
        }
        int64_t max_start =
            operand_shape.dimensions(d) - update_shape.dimensions(d);
        if (max_start < 0) max_start = 0;
        if (value < 0) value = 0;
        if (value > max_start) value = max_start;
        starts[d] = value;
      }


      void* operand_addr = allocations->GetDeviceAddress(args_[0]).opaque();
      void* update_addr = allocations->GetDeviceAddress(args_[1]).opaque();
      void* output_addr = allocations->GetDeviceAddress(args_.back()).opaque();
      const int64_t total_bytes =
          NpuTool::GetXlaShapeSize(operand_shape) * elem_bytes;
      if (output_addr != operand_addr) {
        ret = dep::aclrtMemcpy(output_addr, total_bytes, operand_addr,
                               total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
          return absl::InternalError(absl::StrFormat(
              "aclrtMemcpy for DynamicUpdateSlice operand copy failed. Error "
              "code: %d",
              ret));
        }
      }

      // Coalesce the trailing dimensions whose physical strides agree.
      int split = rank;
      for (int d = rank - 1; d >= 0; --d) {
        if (operand_strides[d] == update_strides[d]) {
          split = d;
        } else {
          break;
        }
      }
      int64_t inner_elems = 1;
      for (int d = rank - 1; d >= split; --d) {
        if (update_strides[d] != inner_elems) {
          return absl::UnimplementedError(
              "DynamicUpdateSlice update is not contiguous in its trailing "
              "dimensions");
        }
        inner_elems *= update_shape.dimensions(d);
      }
      int64_t outer_elems = 1;
      for (int d = 0; d < split; ++d) {
        outer_elems *= update_shape.dimensions(d);
      }
      int64_t inner_dst_base = 0;
      for (int d = split; d < rank; ++d) {
        inner_dst_base += starts[d] * operand_strides[d];
      }
      const int64_t block_bytes = inner_elems * elem_bytes;
      for (int64_t outer = 0; outer < outer_elems; ++outer) {
        int64_t remaining = outer;
        int64_t dst = inner_dst_base;
        for (int d = split - 1; d >= 0; --d) {
          const int64_t idx = remaining % update_shape.dimensions(d);
          remaining /= update_shape.dimensions(d);
          dst += (starts[d] + idx) * operand_strides[d];
        }
        const int64_t src = outer * inner_elems;
        ret = dep::aclrtMemcpy(
            static_cast<char*>(output_addr) + dst * elem_bytes, block_bytes,
            static_cast<const char*>(update_addr) + src * elem_bytes, block_bytes,
            ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
          return absl::InternalError(absl::StrFormat(
              "aclrtMemcpy for DynamicUpdateSlice block failed. Error code: %d",
              ret));
        }
      }
      exec_ret = absl::OkStatus();
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
          NPU_LOG_PRINT(ERROR, err_msg);
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

      // A gather whose slice covers the whole input always clamps its start
      // offset to zero; it is an identity copy of the input.
      auto parse_slice_sizes = [&desc]() {
        std::vector<int64_t> sizes;
        const size_t key_pos = desc.find("slice_sizes={");
        if (key_pos == std::string::npos) return sizes;
        const size_t open = desc.find('{', key_pos);
        const size_t close = desc.find('}', open);
        if (open == std::string::npos || close == std::string::npos) return sizes;
        const std::string body = desc.substr(open + 1, close - open - 1);
        size_t pos = 0;
        while (pos < body.size()) {
          const size_t comma = body.find(',', pos);
          const std::string part =
              body.substr(pos, comma == std::string::npos ? std::string::npos
                                                          : comma - pos);
          try {
            sizes.push_back(std::stoll(part));
          } catch (...) {
          }
          if (comma == std::string::npos) break;
          pos = comma + 1;
        }
        return sizes;
      };
      const std::vector<int64_t> slice_sizes = parse_slice_sizes();
      bool full_slice = !slice_sizes.empty() &&
                        slice_sizes.size() == input_shape.dimensions().size();
      if (full_slice) {
        for (size_t i = 0; i < slice_sizes.size(); ++i) {
          if (slice_sizes[i] != input_shape.dimensions(i)) {
            full_slice = false;
            break;
          }
        }
      }
      if (full_slice &&
          NpuTool::GetXlaShapeSize(input_shape) ==
              NpuTool::GetXlaShapeSize(output_shape)) {
        void* input_addr = allocations->GetDeviceAddress(args_[0]).opaque();
        void* output_addr = allocations->GetDeviceAddress(args_.back()).opaque();
        const size_t bytes =
            NpuTool::GetXlaShapeSize(input_shape) *
            ShapeUtil::ByteSizeOfPrimitiveType(input_shape.element_type());
        ret = dep::aclrtMemcpy(output_addr, bytes, input_addr, bytes,
                               ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
          return absl::InternalError(absl::StrFormat(
              "Failed to copy full-slice gather. Error code: %d", ret));
        }
        exec_ret = absl::OkStatus();
        if (need_cleanup && expanded_output_tensor) {
          dep::aclDestroyTensor(expanded_output_tensor);
        }
        break;
      }

      // ACLNN GatherV2 accepts integer index tensors but not unsigned 32-bit
      // indices, which is what XLA frontends commonly emit for token ids.
      aclTensor* index_tensor = acl_tensor_ptr_vector[1];
      void* cast_index_buf = nullptr;
      aclTensor* cast_index_tensor = nullptr;
      if (index_shape.element_type() == PrimitiveType::U32) {
        std::vector<int64_t> index_dims(index_shape.dimensions().begin(),
                                        index_shape.dimensions().end());
        std::vector<int64_t> index_strides(index_dims.size(), 1);
        int64_t stride = 1;
        for (int i = static_cast<int>(index_dims.size()) - 1; i >= 0; --i) {
          index_strides[i] = stride;
          stride *= index_dims[i];
        }
        const int64_t index_count = NpuTool::GetXlaShapeSize(index_shape);
        const size_t cast_bytes = index_count * sizeof(int32_t);
        ret = dep::aclrtMalloc(&cast_index_buf, cast_bytes,
                               ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
          if (need_cleanup && expanded_output_tensor) {
            dep::aclDestroyTensor(expanded_output_tensor);
          }
          return absl::InternalError(
              absl::StrFormat("Failed to allocate gather index cast buffer. "
                              "Error code: %d",
                              ret));
        }
        cast_index_tensor = dep::aclCreateTensor(
            index_dims.data(), index_dims.size(), ACL_INT32,
            index_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            index_dims.data(), index_dims.size(), cast_index_buf);
        if (cast_index_tensor == nullptr) {
          dep::aclrtFree(cast_index_buf);
          if (need_cleanup && expanded_output_tensor) {
            dep::aclDestroyTensor(expanded_output_tensor);
          }
          return absl::InternalError(
              "Failed to create gather index cast tensor");
        }
        uint64_t cast_workspace_size = 0;
        aclOpExecutor* cast_executor = nullptr;
        ret = dep::aclnnCastGetWorkspaceSize(
            acl_tensor_ptr_vector[1], ACL_INT32, cast_index_tensor,
            &cast_workspace_size, &cast_executor);
        if (ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, cast_workspace_size, cast_executor,
                                  dep::aclnnCast);
        }
        if (ret != ACL_SUCCESS || !exec_ret.ok()) {
          dep::aclDestroyTensor(cast_index_tensor);
          dep::aclrtFree(cast_index_buf);
          if (need_cleanup && expanded_output_tensor) {
            dep::aclDestroyTensor(expanded_output_tensor);
          }
          return absl::InternalError(absl::StrFormat(
              "Failed to cast gather index from U32. Error code: %d", ret));
        }
        index_tensor = cast_index_tensor;
      }

      // 使用 GatherV2
      ret = dep::aclnnGatherV2GetWorkspaceSize(
          acl_tensor_ptr_vector[0], dim, index_tensor, output_tensor,
          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnGatherV2GetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_PRINT(ERROR, err_msg);
        if (cast_index_tensor != nullptr) {
          dep::aclrtSynchronizeStream(acl_stream);
          dep::aclDestroyTensor(cast_index_tensor);
          dep::aclrtFree(cast_index_buf);
        }
        if (need_cleanup && expanded_output_tensor) {
          dep::aclDestroyTensor(expanded_output_tensor);
        }
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnGatherV2);

      // 清理创建的tensor
      if (cast_index_tensor != nullptr) {
        dep::aclrtSynchronizeStream(acl_stream);
        dep::aclDestroyTensor(cast_index_tensor);
        dep::aclrtFree(cast_index_buf);
      }
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      if (compare_ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Compare GetWorkspaceSize failed. Error code: %d", compare_ret);
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSWhere);
      break;
    }

    case HloOpcode::kOr: {
      if (acl_tensor_ptr_vector.size() < 3) {
        return absl::InternalError(
            "Or operation expects 3 parameters (input1, input2, output)");
      }
      const bool is_bool =
          param_shapes[0].element_type() == PrimitiveType::PRED;
      if (is_bool) {
        ret = dep::aclnnLogicalOrGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          return absl::InternalError(absl::StrFormat(
              "Call aclnnLogicalOrGetWorkspaceSize failed. Error code: %d",
              ret));
        }
        exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                dep::aclnnLogicalOr);
      } else {
        ret = dep::aclnnBitwiseOrTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          return absl::InternalError(absl::StrFormat(
              "Call aclnnBitwiseOrTensorGetWorkspaceSize failed. Error code: "
              "%d",
              ret));
        }
        exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                dep::aclnnBitwiseOrTensor);
      }
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
        NPU_LOG_PRINT(ERROR, err_msg);
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
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnBitwiseAndTensor);
      break;
    }

    case HloOpcode::kConvert: {
      if (acl_tensor_ptr_vector.size() < 2) {
        const std::string err_msg = absl::StrFormat(
            "Convert operation expects 2 parameters (input, output), but got "
            "%d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      aclDataType output_dtype = NpuTool::ConvertXlaType2AclDataType(
          param_shapes.back().element_type());
      ret = dep::aclnnCastGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                           output_dtype,
                                           acl_tensor_ptr_vector.back(),
                                           &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnCastGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnCast);
      break;
    }

    default: {
      const std::string err_msg =
          absl::StrFormat("Op code(%v) does not support execute now",
                          HloOpcodeString(thunk_info().op_code));
      NPU_LOG_PRINT(ERROR, err_msg);
      return absl::InternalError(err_msg);
    }
  }

  // ACLNN launches asynchronously on the ACL stream, but the PJRT completion
  // events do not reliably observe that stream. Block here so buffers are safe
  // to read/rewrite once the thunk returns.
  TF_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());

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
  NPU_LOG_PRINT(DEBUG, addr_debug);

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
      NPU_LOG_PRINT(ERROR, err_msg);
      return absl::InternalError(err_msg);
    }

    auto alloc_result = pool->AllocateForStream(stream, workspaceSize);
    if (!alloc_result.ok()) {
      const std::string err_msg =
          absl::StrFormat("Failed to allocate workspace from pool: %s",
                          alloc_result.status().ToString().c_str());
      NPU_LOG_PRINT(ERROR, err_msg);
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
    NPU_LOG_PRINT(ERROR, err_msg);
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
