/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
 */

#ifndef XLA_XLA_NPU_COMMON_NPU_TOOL_H_
#define XLA_XLA_NPU_COMMON_NPU_TOOL_H_

#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_interface.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

// npu常用工具函数类
class NpuTool {
 public:
  static HloPrintOptions GetDetailedPrintOptions() {
    HloPrintOptions opts = HloPrintOptions::Default();

    opts
        // 打印大常量（默认关闭，大常量会被截断）
        .set_print_large_constants(true)
        // 打印子计算的完整内容（默认仅打印子计算名称）HloPrintOptions::PrintSubcomputationMode::kFullBodies
        .set_print_subcomputation_mode(
            HloPrintOptions::PrintSubcomputationMode::kNameOnly)
        // 打印原始值
        .set_print_original_value(true)
        // 打印元数据（如位置、注释等）-- 不打印
        .set_print_metadata(false)
        // 只打印元数据中的op name
        .set_print_metadata_only_op_name(true)
        // 打印后端配置（如 GPU/CPU 特定的优化配置）
        .set_print_backend_config(true)
        // 排序后端配置的 JSON 输出（便于阅读）
        .set_sort_backend_config(true)
        // 打印操作数的形状（默认关闭）
        .set_print_operand_shape(true)
        // 打印操作数的索引注释（每 N 个操作数标注一次，0 表示全部标注）
        .set_print_operand_index_annotation_interval(0)
        // 打印程序形状（输入输出整体形状）
        .set_print_program_shape(true)
        // 打印 % 前缀（默认开启，保持语义完整）
        .set_print_percent(true)
        // 打印控制依赖（默认开启，保留执行顺序信息）
        .set_print_control_dependencies(true)
        // 打印 infeed/outfeed 配置
        .set_print_infeed_outfeed_config(true)
        // 不压缩操作数（默认关闭，确保所有操作数都显示）
        .set_compact_operands(false)
        // 打印操作数名称（默认开启）
        .set_print_operand_names(true)
        // 打印结果形状（默认开启）
        .set_print_result_shape(true)
        // 打印额外属性（如 fusion 类型、replica group 等）
        .set_print_extra_attributes(true)
        // 打印完整的 replica group 列表（避免压缩）
        .set_print_full_replica_group_list(true)
        // 打印参数编号和通道 ID
        .set_print_parameter_number(true)
        .set_print_channel_id(true);

    return opts;
  }

  static void PrintHloModule(const HloModule* hlo_module,
                             const std::string& title) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
    NPU_LOG_FORMAT_PRINT(DEBUG, "**** %s: ****", title);
    HloPrintOptions hloPrintOptions = GetDetailedPrintOptions();
    NPU_LOG_FORMAT_PRINT(DEBUG, hlo_module->ToString(hloPrintOptions));
    NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
  }

  static void PrintModuleOp(mlir::ModuleOp module, const std::string& title) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
    std::string result;
    llvm::raw_string_ostream os(result);
    module.print(os);
    NPU_LOG_FORMAT_PRINT(DEBUG, "**** %s: ****", title);
    NPU_LOG_FORMAT_PRINT(DEBUG, os.str());
    NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
  }

  template <typename printDataType>
  static void PrintNpuDeviceMemoryValue(const std::string& title,
                                        const int64_t& size,
                                        const void* deviceAddr,
                                        int64_t max_print_count = -1) {
    NPU_LOG_FORMAT_PRINT(DEBUG, title);

    // 获取配置的最大打印数量
    if (max_print_count <= 0) {
      max_print_count = NPU_CONFIG_GET_INT(xla::NpuConfig::Key::XLA_NPU_MAX_PRINT_COUNT);
    }

    int64_t print_count = std::min(size, max_print_count);
    std::vector<printDataType> resultData(print_count, 0);
    aclError acl_ret = npu::aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(printDataType),
        deviceAddr, print_count * sizeof(printDataType),
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (acl_ret != ACL_SUCCESS) {
      NPU_LOG_FORMAT_PRINT(
          DEBUG, "    Copy result from device to host failed. ERROR: %d", acl_ret);
      return;
    }

    for (int64_t i = 0; i < print_count; i++) {
      NPU_LOG_FORMAT_PRINT(DEBUG, "    Value[%ld] on device is: %f", i,
                           static_cast<double>(resultData[i]));
    }

    if (size > max_print_count) {
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "    ...(total %ld elements, showing first %ld)",
                           size, max_print_count);
    }
  }

  // 根据 Shape 的元素类型自动选择打印类型的接口
  static void PrintNpuDeviceMemoryValueByShape(const std::string& title,
                                               const Shape& xla_shape,
                                               const void* device_addr,
                                               int64_t max_print_count = -1);

  // 计算xla shape表示的元素个数
  static int64_t GetXlaShapeSize(const Shape& xla_data_shape) {
    int64_t size = 1;
    for (int i = 0; i < xla_data_shape.dimensions_size(); ++i) {
      const int64_t& xla_data_shape_dim = xla_data_shape.dimensions(i);
      size *= xla_data_shape_dim;
    }

    return size;
  }

  static std::string GetShapeString(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
      return "empty";
    }

    std::stringstream ss;
    ss << "[";
    for (const auto& shape_dim : shape) {
      ss << shape_dim << ", ";
    }
    ss << "]";

    return ss.str();
  }

  static std::vector<int64_t> ConvertXlaShape2VectorShape(
      const Shape& xla_data_shape) {
    std::vector<int64_t> vector_shape;
    for (int i = 0; i < xla_data_shape.dimensions_size(); ++i) {
      const int64_t& xla_data_shape_dim = xla_data_shape.dimensions(i);
      vector_shape.emplace_back(xla_data_shape_dim);
    }

    return vector_shape;
  }

  static std::string GetXlaShapeString(const Shape& xla_data_shape) {
    if (xla_data_shape.IsTuple()) {
      std::stringstream ss;
      ss << "(";
      const auto& tuple_shapes = xla_data_shape.tuple_shapes();
      for (size_t i = 0; i < tuple_shapes.size(); ++i) {
        ss << GetXlaShapeString(tuple_shapes[i]);
        if (i < tuple_shapes.size() - 1) {
          ss << ", ";
        }
      }
      ss << ")";
      return ss.str();
    } else {
      const auto& vector_shape = ConvertXlaShape2VectorShape(xla_data_shape);
      return GetShapeString(vector_shape);
    }
  }

  static void PrintXlaShape(const Shape& xla_data_shape,
                            std::string desc = "Xla tensor shape: ") {
    const auto& shape_str = GetXlaShapeString(xla_data_shape);
    NPU_LOG_FORMAT_PRINT(DEBUG, "%s%s", desc, shape_str);
  }

  static aclDataType ConvertXlaType2AclDataType(const PrimitiveType& type) {
    switch (type) {
      default:
        return aclDataType::ACL_FLOAT;
    };
  }

  static aclTensor* CreateAclTensorPtr(const Shape& xla_data_shape,
                                       const PrimitiveType& xla_data_type,
                                       void* device_addr) {
    const auto& data_shape = ConvertXlaShape2VectorShape(xla_data_shape);
    const auto& data_type = ConvertXlaType2AclDataType(xla_data_type);

    // 从 XLA shape 的 layout 中获取正确的 strides
    // 如果 shape 没有 layout 信息或 layout 是默认的，则使用连续布局的 strides
    std::vector<int64_t> strides;
    if (xla_data_shape.has_layout()) {
      // 从 layout 中获取 strides
      const auto& layout = xla_data_shape.layout();
      const auto& minor_to_major = layout.minor_to_major();

      // 根据 minor_to_major 顺序计算 strides
      // minor_to_major[0] 是最 minor 的维度（stride=1）
      strides.resize(data_shape.size());
      int64_t stride = 1;
      for (size_t i = 0; i < minor_to_major.size(); ++i) {
        int64_t dim = minor_to_major[i];
        strides[dim] = stride;
        stride *= data_shape[dim];
      }
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "CreateAclTensorPtr: Using layout-based strides");
    } else {
      // 使用默认的连续 tensor strides (从后往前累乘)
      strides.resize(data_shape.size(), 1);
      for (int64_t i = data_shape.size() - 2; i >= 0; i--) {
        strides[i] = data_shape[i + 1] * strides[i + 1];
      }
    }

    // Debug: print strides
    std::string strides_str = "[";
    for (size_t i = 0; i < strides.size(); ++i) {
      if (i > 0) strides_str += ", ";
      strides_str += std::to_string(strides[i]);
    }
    strides_str += "]";
    NPU_LOG_FORMAT_PRINT(DEBUG, "CreateAclTensorPtr: shape=%s, strides=%s",
                         GetXlaShapeString(xla_data_shape).c_str(),
                         strides_str.c_str());

    // 调用 aclCreateTensor 接口创建 aclTensor
    auto tensor =
        npu::aclCreateTensor(data_shape.data(), data_shape.size(), data_type,
                             strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                             data_shape.data(), data_shape.size(), device_addr);

    return tensor;
  }

  // 生成不重复文件夹名（Abseil 实现）
  static std::string generate_unique_folder_name_absl();

  // 解析 reduce 操作的维度参数
  // 格式: "dims=[dim1,dim2,...],keepDims=true/false"
  static void ParseReduceDims(const std::string& desc,
                              std::vector<int64_t>& dims, bool& keepDims) {
    dims.clear();
    keepDims = false;

    size_t dims_pos = desc.find("dims=[");
    if (dims_pos == std::string::npos) {
      return;
    }

    size_t dims_end = desc.find("]", dims_pos);
    if (dims_end == std::string::npos) {
      return;
    }

    std::string dims_str = desc.substr(dims_pos + 6, dims_end - dims_pos - 6);
    std::stringstream ss(dims_str);
    std::string dim_str;
    while (std::getline(ss, dim_str, ',')) {
      dims.push_back(std::stoll(dim_str));
    }

    size_t keep_pos = desc.find("keepDims=");
    if (keep_pos != std::string::npos) {
      keepDims = (desc.substr(keep_pos + 9, 5) == "true");
    }
  }
};

}  // namespace xla

#endif  // XLA_XLA_NPU_COMMON_NPU_TOOL_H_