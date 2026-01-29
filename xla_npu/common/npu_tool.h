#ifndef COMMON_NPU_TOOL_H_
#define COMMON_NPU_TOOL_H_

#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "xla/xla_npu/common/npu_log.h"
#include "xla/xla_npu/third_party/register/aclInterface.h"
#include "xla/xla_npu/third_party/register/aclnnInterface.h"

namespace xla {

class NpuTool {
   public:
    static HloPrintOptions GetDetailedPrintOptions() {
        HloPrintOptions opts = HloPrintOptions::Default();

        opts
            // 打印大常量（默认关闭，大常量会被截断）
            .set_print_large_constants(true)
            // 打印子计算的完整内容（默认仅打印子计算名称）HloPrintOptions::PrintSubcomputationMode::kFullBodies
            .set_print_subcomputation_mode(HloPrintOptions::PrintSubcomputationMode::kNameOnly)
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

    static void PrintHloModule(const HloModule* hlo_module, const std::string& title) {
        NpuLog::Print(NpuLog::Level::DEBUG, "========================================");
        NpuLog::Print(NpuLog::Level::DEBUG, "**** %s: ****", title);
        HloPrintOptions hloPrintOptions = GetDetailedPrintOptions();
        NpuLog::Print(NpuLog::Level::DEBUG, hlo_module->ToString(hloPrintOptions));
        NpuLog::Print(NpuLog::Level::DEBUG, "========================================");
    }

    static void PrintModuleOp(mlir::ModuleOp module, const std::string& title) {
        NpuLog::Print(NpuLog::Level::DEBUG, "========================================");
        std::string result;
        llvm::raw_string_ostream os(result);
        module.print(os);
        NpuLog::Print(NpuLog::Level::DEBUG, "**** %s: ****", title);
        NpuLog::Print(NpuLog::Level::DEBUG, os.str());
        NpuLog::Print(NpuLog::Level::DEBUG, "========================================");
    }

    template <typename printDataType>
    static void PrintNpuDeviceMemoryValue(const std::string& title, const int64_t& size, const void* deviceAddr) {
        NpuLog::Print(NpuLog::Level::DEBUG, title);

        std::vector<printDataType> resultData(size, 0);
        aclError ret = npu::aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr,
                                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);

        if (ret != ACL_SUCCESS) {
            NpuLog::Print(NpuLog::Level::DEBUG, "    Copy result from device to host failed. ERROR: %d", ret);
            return;
        }
        for (int64_t i = 0; i < size; i++) {
            NpuLog::Print(NpuLog::Level::DEBUG, "    Value[%ld] on device is: %f", i, resultData[i]);
        }
    }

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

    static std::vector<int64_t> ConvertXlaShape2VectorShape(const Shape& xla_data_shape) {
        std::vector<int64_t> vector_shape;
        for (int i = 0; i < xla_data_shape.dimensions_size(); ++i) {
            const int64_t& xla_data_shape_dim = xla_data_shape.dimensions(i);
            vector_shape.emplace_back(xla_data_shape_dim);
        }

        return vector_shape;
    }

    static void PrintXlaShape(const Shape& xla_data_shape, std::string desc = "Xla tensor shape: ") {
        const auto& vector_shape = ConvertXlaShape2VectorShape(xla_data_shape);
        const auto& vector_shape_str = GetShapeString(vector_shape);
        NpuLog::Print(NpuLog::Level::DEBUG, "%s%s", desc, vector_shape_str);
    }

    static aclDataType ConvertXlaType2AclDataType(const PrimitiveType& type) {
        switch (type) {
            default:
                return aclDataType::ACL_FLOAT;
        };
    }

    static aclTensor* CreateAclTensorPtr(const Shape& xla_data_shape, const PrimitiveType& xla_data_type,
                                         void* device_addr) {
        const auto& data_shape = ConvertXlaShape2VectorShape(xla_data_shape);
        const auto& data_type = ConvertXlaType2AclDataType(xla_data_type);

        // 计算连续tensor的strides
        std::vector<int64_t> strides(data_shape.size(), 1);
        for (int64_t i = data_shape.size() - 2; i >= 0; i--) {
            strides[i] = data_shape[i + 1] * strides[i + 1];
        }

        // 调用aclCreateTensor接口创建aclTensor
        auto tensor = npu::aclCreateTensor(data_shape.data(), data_shape.size(), data_type, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, data_shape.data(), data_shape.size(), device_addr);

        return tensor;
    }
};

}  // namespace xla

#endif  // COMMON_NPU_TOOL_H_