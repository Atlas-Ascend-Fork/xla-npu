#ifndef PJRT_NPU_NPU_TOOL_H_
#define PJRT_NPU_NPU_TOOL_H_

#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"

// #include "xla/shape.h"
// #include "xla/shape_util.h"
// #include "xla/xla_data.pb.h"

// #include "xla/xla_npu/third_party/metadef/external/graph/tensor.h"

namespace xla {

class NpuTool {
public:
    static HloPrintOptions GetDetailedPrintOptions() {
        HloPrintOptions opts = HloPrintOptions::Default();

        opts
        // 打印大常量（默认关闭，大常量会被截断）
        .set_print_large_constants(true)
        // 打印子计算的完整内容（默认仅打印子计算名称）
        .set_print_subcomputation_mode(HloPrintOptions::PrintSubcomputationMode::kFullBodies)
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
        std::cout << "========================================" << std::endl;
        std::cout << "**** " << title << ": ****" << std::endl;
        std::cout << std::endl;
        HloPrintOptions hloPrintOptions = GetDetailedPrintOptions();
        std::cout << hlo_module->ToString(hloPrintOptions) << std::endl;
        std::cout << "========================================" << std::endl;
    }

    static void PrintDebugLog(const std::string& logStr) {
        std::cout << "DEBUG: " << logStr << std::endl;
    }

    static void PrintModuleOp(mlir::ModuleOp module, const std::string& title) {
        std::cout << "========================================" << std::endl;
        std::string result;
        llvm::raw_string_ostream os(result);
        module.print(os);
        std::cout << "**** " << title << ": ****" << std::endl;
        std::cout << os.str() << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // static std::string GetShapeString(const ge::Shape &shape) {
    //     std::stringstream ss;
    //     auto dims = shape.GetDims();
    //     if (dims.empty()) {
    //         return "empty";
    //     }
    //     ss << "[";
    //     size_t index = 0U;
    //     for (; index < (dims.size() - 1U); index++) {
    //     ss << dims[index] << ", ";
    //     }
    //     ss << dims[index] << "]";
        
    //     return ss.str(); 
    // }

    // static ge::DataType ConvertXlaType2GeType(const PrimitiveType& type)
    // {
    //     switch (type) {
    //         default:
    //             return ge::DataType::DT_FLOAT;
    //     };
    // }

    // static PrimitiveType ConvertGeType2XlaType(const ge::DataType& type)
    // {
    //     switch (type) {
    //         default:
    //             return PrimitiveType::F32;
    //     };
    // }

    // static ge::Shape ConvertXlaShape2GeShape(const Shape& xla_data_shape)
    // {
    //     std::vector<int64_t> ge_shape_vector;
    //     for (int i = 0; i < xla_data_shape.dimensions_size(); ++i) {
    //         const int64_t& xla_data_shape_dim = xla_data_shape.dimensions(i);
    //         ge_shape_vector.emplace_back(xla_data_shape_dim);
    //         // ge_data_shape.SetDim(i, xla_data_shape_dim);
    //     }

    //     ge::Shape ge_data_shape(ge_shape_vector);
    //     std::cout << "convert xla shape to ge shape,"
    //         << " input xla shape is " << xla_data_shape
    //         << ", output ge shape is " << GetShapeString(ge_data_shape)
    //         << std::endl;

    //     return ge_data_shape;
    // }

    // static Shape ConvertGeShape2XlaShape(const ge::Shape& ge_data_shape, const ge::DataType& ge_data_type)
    // {
    //     const auto& xla_data_dims = ge_data_shape.GetDims();
    //     const auto& xla_data_type = ConvertGeType2XlaType(ge_data_type);
    //     const Shape& xla_data_shape = ShapeUtil::MakeShape(xla_data_type, xla_data_dims);

    //     std::cout << "convert ge shape to xla shape,"
    //         << " input ge shape is " << GetShapeString(ge_data_shape)
    //         << " which type is " << xla_data_type
    //         << ", output xla shape is " << xla_data_shape
    //         << std::endl;

    //     return xla_data_shape;
    // }
};

}
#endif  //PJRT_NPU_PJRT_COMMON_H_