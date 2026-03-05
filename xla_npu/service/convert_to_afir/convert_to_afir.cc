/* Copyright 2025 The OpenXLA Authors

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

#include "xla/xla_npu/service/convert_to_afir/convert_to_afir.h"

#include <algorithm>
#include <functional>
#include <set>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/util.h"

// Ascend-MLIR AFIR Dialect
#include "Dialect/AFIR/AFIR.h"

namespace xla {

namespace {

// Helper class to manage the conversion state
class HloToAfirConverter {
   public:
    HloToAfirConverter(mlir::ModuleOp module, mlir::MLIRContext* context, const HloToAfirOptions& options)
        : module_op_ptr_(module.getOperation()), context_(context), builder_(context), options_(options) {}

    mlir::ModuleOp getModule() const { return mlir::cast<mlir::ModuleOp>(module_op_ptr_); }

    // Convert the entire HLO module
    absl::Status Convert(const HloModule& hlo_module);

    // Convert a single HLO computation
    absl::StatusOr<mlir::func::FuncOp> ConvertComputation(const HloComputation& computation, bool is_entry);

    // Set module-level graph attributes
    void SetModuleGraphAttr(const HloModule& hlo_module);
    void SetModuleGraphAttr(const HloComputation& computation);

   private:
    // Convert a single HLO instruction
    absl::StatusOr<mlir::Value> ConvertInstruction(const HloInstruction* instruction);

    // Instruction-specific converters
    absl::StatusOr<mlir::Value> ConvertParameterOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertConstantOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertAddOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertSubtractOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertMultiplyOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertDivideOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertMaximumOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertBroadcastOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertTanhOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertNegateOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertExponentialOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertConvertOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertReshapeOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertReduceOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertConcatenateOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertFusionOp(const HloInstruction* instruction);
    absl::StatusOr<mlir::Value> ConvertCallOp(const HloInstruction* instruction);

    // Type conversion
    mlir::Type ConvertShape(const Shape& shape);
    mlir::Type ConvertPrimitiveType(PrimitiveType prim_type);

    // Location conversion
    mlir::Location ConvertLocation(const HloInstruction* instr);
    mlir::Location ConvertLocation(const HloComputation* comp);

    // AFIR-specific helpers
    mlir::AffineMap GetIdentityAffineMap(int64_t rank);
    mlir::ArrayAttr CreateIndexingMaps(llvm::ArrayRef<mlir::AffineMap> maps);
    mlir::afir::AscTensorGroupsAttr CreateDefaultOutputAttr(mlir::Type tensor_type);

    // Helper: Check if reshape only adds/removes dimensions of size 1
    // Returns true if it's a no-op reshape (only 1-dim changes), and the effective type to use
    struct ReshapeTypeInfo {
        bool is_one_dim_reshape;
        mlir::Type effective_type;
    };
    ReshapeTypeInfo GetReshapeEffectiveType(const HloInstruction* reshape_instr);

    // Module state
    mlir::Operation* module_op_ptr_;
    mlir::MLIRContext* context_;
    mlir::OpBuilder builder_;

    const HloToAfirOptions& options_;

    // Value mapping for SSA conversion
    absl::flat_hash_map<const HloInstruction*, mlir::Value> value_map_;

    // Function mapping
    absl::flat_hash_map<const HloComputation*, mlir::func::FuncOp> function_map_;
};

absl::Status HloToAfirConverter::Convert(const HloModule& hlo_module) {
    const HloComputation* entry = hlo_module.entry_computation();
    if (!entry) {
        return absl::InvalidArgumentError("Module has no entry computation");
    }

    // Set module-level asc_graph_attr if emit_afir_dialect is enabled
    if (options_.emit_afir_dialect) {
        SetModuleGraphAttr(hlo_module);
    }

    // Convert all embedded computations FIRST (before entry computation)
    // This ensures that fused/called computations are available when
    // the entry computation references them via fusion or call instructions
    for (const HloComputation* computation : hlo_module.computations()) {
        if (computation == entry) continue;

        TF_ASSIGN_OR_RETURN(auto func, ConvertComputation(*computation, /*is_entry=*/false));
    }

    // Convert entry computation LAST (after all embedded computations)
    TF_ASSIGN_OR_RETURN(auto entry_func, ConvertComputation(*entry, /*is_entry=*/true));

    return absl::OkStatus();
}

absl::StatusOr<mlir::func::FuncOp> HloToAfirConverter::ConvertComputation(const HloComputation& computation,
                                                                          bool is_entry) {
    // Collect parameter types
    llvm::SmallVector<mlir::Type> param_types;
    for (const HloInstruction* param : computation.parameter_instructions()) {
        param_types.push_back(ConvertShape(param->shape()));
    }

    // Determine return type
    // If root instruction is a reshape that only adds/removes dimensions of size 1,
    // use the input type of reshape (since reshape is a no-op in AFIR)
    mlir::Type result_type;
    const HloInstruction* root = computation.root_instruction();

    if (options_.emit_afir_dialect && root->opcode() == HloOpcode::kReshape) {
        auto reshape_info = GetReshapeEffectiveType(root);
        result_type = reshape_info.effective_type;
    } else {
        result_type = ConvertShape(root->shape());
    }

    // Create function type
    mlir::TypeRange param_types_range(param_types);
    auto func_type = builder_.getFunctionType(param_types_range, result_type);

    // Create function location
    mlir::Location loc = ConvertLocation(&computation);

    // Set insertion point to module body before creating function
    builder_.setInsertionPointToEnd(getModule().getBody());

    // Replace all '.' with '_' in computation name for funcOp
    std::string func_name = std::string(computation.name());
    size_t pos = 0;
    while ((pos = func_name.find('.', pos)) != std::string::npos) {
        func_name.replace(pos, 1, "_");
        pos++;
    }

    // Create function
    mlir::func::FuncOp func = builder_.create<mlir::func::FuncOp>(loc, func_name, func_type);

    function_map_[&computation] = func;

    // Create function body
    mlir::Block* block = func.addEntryBlock();
    builder_.setInsertionPointToEnd(block);

    // Map parameters and create load nodes when emit_afir_dialect is true
    for (size_t i = 0; i < computation.num_parameters(); ++i) {
        const HloInstruction* param = computation.parameter_instruction(i);
        mlir::Value arg = block->getArgument(i);

        if (options_.preserve_names && !param->name().empty()) {
            arg.setLoc(mlir::NameLoc::get(builder_.getStringAttr(param->name())));
        }

        // Create load node for AFIR dialect
        if (options_.emit_afir_dialect) {
            mlir::Type param_type = ConvertShape(param->shape());
            int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(param_type).getRank();
            auto identity_map = GetIdentityAffineMap(rank);

            // Create load output attribute
            auto output_attr = CreateDefaultOutputAttr(param_type);
            llvm::SmallVector<mlir::Attribute> output_attrs;
            output_attrs.push_back(output_attr);
            auto outputs_array = builder_.getArrayAttr(output_attrs);

            // Create indexing_maps for load (single map for input)
            llvm::SmallVector<mlir::AffineMap> load_maps;
            load_maps.push_back(identity_map);
            auto indexing_maps_attr = CreateIndexingMaps(load_maps);

            // Create load operation
            auto load_op =
                builder_.create<mlir::afir::LoadOp>(ConvertLocation(param), param_type, arg, indexing_maps_attr,
                                                    nullptr,                         // loop_axis
                                                    builder_.getDictionaryAttr({}),  // ir_attr_def
                                                    nullptr,                         // tmp_buffers
                                                    outputs_array);

            value_map_[param] = load_op.getResult();
        } else {
            value_map_[param] = arg;
        }
    }

    // Convert instructions in post-order (topological sort)
    std::vector<HloInstruction*> instructions = computation.MakeInstructionPostOrder();

    for (const HloInstruction* instr : instructions) {
        if (instr->opcode() == HloOpcode::kParameter) continue;

        TF_ASSIGN_OR_RETURN(mlir::Value value, ConvertInstruction(instr));
        value_map_[instr] = value;
    }

    // Set return value with store node when emit_afir_dialect is true
    mlir::Value result = value_map_[computation.root_instruction()];

    if (options_.emit_afir_dialect) {
        // Determine the actual result type for store operation
        // If root instruction is a reshape that only adds/removes dimensions of size 1,
        // use the input type of reshape (since reshape is a no-op in AFIR)
        const HloInstruction* root = computation.root_instruction();
        mlir::Type result_type;

        if (root->opcode() == HloOpcode::kReshape) {
            auto reshape_info = GetReshapeEffectiveType(root);
            result_type = reshape_info.effective_type;
        } else {
            result_type = ConvertShape(root->shape());
        }

        int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
        auto identity_map = GetIdentityAffineMap(rank);

        // Create store output attribute
        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        // Create indexing_maps for store (single map for output)
        llvm::SmallVector<mlir::AffineMap> store_maps;
        store_maps.push_back(identity_map);
        auto indexing_maps_attr = CreateIndexingMaps(store_maps);

        // Create store operation
        auto store_op = builder_.create<mlir::afir::StoreOp>(ConvertLocation(computation.root_instruction()),
                                                             result_type, result, indexing_maps_attr,
                                                             nullptr,                         // loop_axis
                                                             builder_.getDictionaryAttr({}),  // ir_attr_def
                                                             nullptr,                         // tmp_buffers
                                                             outputs_array);

        // Return the stored value
        builder_.create<mlir::func::ReturnOp>(ConvertLocation(computation.root_instruction()), store_op.getResult());
    } else {
        builder_.create<mlir::func::ReturnOp>(ConvertLocation(computation.root_instruction()), result);
    }

    return func;
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertInstruction(const HloInstruction* instruction) {
    switch (instruction->opcode()) {
        case HloOpcode::kParameter:
            return ConvertParameterOp(instruction);

        case HloOpcode::kConstant:
            return ConvertConstantOp(instruction);

        case HloOpcode::kAdd:
            return ConvertAddOp(instruction);

        case HloOpcode::kSubtract:
            return ConvertSubtractOp(instruction);

        case HloOpcode::kMultiply:
            return ConvertMultiplyOp(instruction);

        case HloOpcode::kDivide:
            return ConvertDivideOp(instruction);

        case HloOpcode::kBroadcast:
            return ConvertBroadcastOp(instruction);

        case HloOpcode::kMaximum:
            return ConvertMaximumOp(instruction);

        case HloOpcode::kTanh:
            return ConvertTanhOp(instruction);

        case HloOpcode::kNegate:
            return ConvertNegateOp(instruction);

        case HloOpcode::kExp:
            return ConvertExponentialOp(instruction);

        case HloOpcode::kConvert:
            return ConvertConvertOp(instruction);

        case HloOpcode::kReshape:
            return ConvertReshapeOp(instruction);

        case HloOpcode::kReduce:
            return ConvertReduceOp(instruction);

        case HloOpcode::kConcatenate:
            return ConvertConcatenateOp(instruction);

        case HloOpcode::kFusion:
            return ConvertFusionOp(instruction);

        case HloOpcode::kCall:
            return ConvertCallOp(instruction);

        default:
            return absl::UnimplementedError(absl::StrCat("Unsupported opcode: ", HloOpcodeString(instruction->opcode()),
                                                         " (instruction: ", instruction->name(), ")"));
    }
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertParameterOp(const HloInstruction* instruction) {
    // Parameters are handled in computation conversion
    auto it = value_map_.find(instruction);
    if (it == value_map_.end()) {
        return absl::InternalError(absl::StrCat("Parameter not found in value map: ", instruction->name()));
    }
    return it->second;
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertConstantOp(const HloInstruction* instruction) {
    mlir::Location loc = ConvertLocation(instruction);
    const Literal& literal = instruction->literal();

    // Handle different element types
    mlir::Type result_type = ConvertShape(instruction->shape());

    // For now, only support scalar constants
    if (literal.shape().dimensions().size() != 0) {
        return absl::UnimplementedError("Only scalar constants are currently supported");
    }

    // Create constant based on element type
    switch (literal.shape().element_type()) {
        case F32: {
            float value = literal.GetFirstElement<float>();
            auto tensor_type = mlir::cast<mlir::RankedTensorType>(result_type);

            if (options_.emit_afir_dialect) {
                // Create identity indexing map
                int64_t rank = tensor_type.getRank();
                auto identity_map = GetIdentityAffineMap(rank);
                auto indexing_maps_attr = CreateIndexingMaps({identity_map});

                // Create output attribute
                auto output_attr = CreateDefaultOutputAttr(result_type);
                llvm::SmallVector<mlir::Attribute> output_attrs;
                output_attrs.push_back(output_attr);
                auto outputs_array = builder_.getArrayAttr(output_attrs);

                // Create AFIR scalar constant with value attribute
                auto value_attr = mlir::FloatAttr::get(tensor_type.getElementType(), value);
                auto scalar_op = builder_.create<mlir::afir::ScalarOp>(loc, result_type, value_attr, indexing_maps_attr,
                                                                       nullptr,                         // loop_axis
                                                                       builder_.getDictionaryAttr({}),  // ir_attr_def
                                                                       nullptr,                         // tmp_buffers
                                                                       outputs_array);
                return scalar_op.getResult();
            } else {
                // Use DenseElementsAttr to create a tensor constant (non-AFIR mode)
                auto attr = mlir::DenseElementsAttr::get(tensor_type, value);
                auto op = builder_.create<mlir::arith::ConstantOp>(loc, result_type, attr);
                return op.getResult();
            }
        }
        case S32: {
            int32_t value = literal.GetFirstElement<int32_t>();
            auto tensor_type = mlir::cast<mlir::RankedTensorType>(result_type);

            if (options_.emit_afir_dialect) {
                // Create identity indexing map
                int64_t rank = tensor_type.getRank();
                auto identity_map = GetIdentityAffineMap(rank);
                auto indexing_maps_attr = CreateIndexingMaps({identity_map});

                // Create output attribute
                auto output_attr = CreateDefaultOutputAttr(result_type);
                llvm::SmallVector<mlir::Attribute> output_attrs;
                output_attrs.push_back(output_attr);
                auto outputs_array = builder_.getArrayAttr(output_attrs);

                // Create AFIR scalar constant with value attribute
                auto value_attr = mlir::IntegerAttr::get(tensor_type.getElementType(), value);
                auto scalar_op = builder_.create<mlir::afir::ScalarOp>(loc, result_type, value_attr, indexing_maps_attr,
                                                                       nullptr,                         // loop_axis
                                                                       builder_.getDictionaryAttr({}),  // ir_attr_def
                                                                       nullptr,                         // tmp_buffers
                                                                       outputs_array);
                return scalar_op.getResult();
            } else {
                // Use DenseElementsAttr to create a tensor constant (non-AFIR mode)
                auto attr = mlir::DenseElementsAttr::get(tensor_type, value);
                auto op = builder_.create<mlir::arith::ConstantOp>(loc, result_type, attr);
                return op.getResult();
            }
        }
        default:
            return absl::UnimplementedError(absl::StrCat("Unsupported constant element type: ",
                                                         PrimitiveType_Name(literal.shape().element_type())));
    }
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertAddOp(const HloInstruction* instruction) {
    if (instruction->operand_count() != 2) {
        return absl::InvalidArgumentError(
            absl::StrCat("Add operation must have 2 operands, got ", instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Value lhs = value_map_[instruction->operand(0)];
    mlir::Value rhs = value_map_[instruction->operand(1)];
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Use afir.add when emit_afir_dialect is true
    if (options_.emit_afir_dialect) {
        int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
        auto identity_map = GetIdentityAffineMap(rank);

        // Build outputs attribute array
        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        // Build indexing_maps array (3 maps: lhs, rhs, result)
        llvm::SmallVector<mlir::AffineMap> maps;
        maps.push_back(identity_map);  // lhs
        maps.push_back(identity_map);  // rhs
        maps.push_back(identity_map);  // result
        auto indexing_maps_attr = CreateIndexingMaps(maps);

        // Create AddOp with all required attributes
        auto add_op = builder_.create<mlir::afir::AddOp>(loc, result_type, lhs, rhs, indexing_maps_attr,
                                                         nullptr,                         // loop_axis (use default)
                                                         builder_.getDictionaryAttr({}),  // ir_attr_def (use default)
                                                         nullptr,                         // tmp_buffers (use default)
                                                         outputs_array);
        return add_op.getResult();
    }

    // Use arith.add as fallback
    auto add_op = builder_.create<mlir::arith::AddIOp>(loc, lhs, rhs);

    return add_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertSubtractOp(const HloInstruction* instruction) {
    if (instruction->operand_count() != 2) {
        return absl::InvalidArgumentError(
            absl::StrCat("Subtract operation must have 2 operands, got ", instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Value lhs = value_map_[instruction->operand(0)];
    mlir::Value rhs = value_map_[instruction->operand(1)];
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Use afir.sub when emit_afir_dialect is true
    if (options_.emit_afir_dialect) {
        int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
        auto identity_map = GetIdentityAffineMap(rank);

        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        llvm::SmallVector<mlir::AffineMap> maps;
        maps.push_back(identity_map);
        maps.push_back(identity_map);
        maps.push_back(identity_map);
        auto indexing_maps_attr = CreateIndexingMaps(maps);

        auto sub_op = builder_.create<mlir::afir::SubOp>(loc, result_type, lhs, rhs, indexing_maps_attr, nullptr,
                                                         builder_.getDictionaryAttr({}), nullptr, outputs_array);
        return sub_op.getResult();
    }

    // Use arith.sub as fallback
    auto sub_op = builder_.create<mlir::arith::SubIOp>(loc, lhs, rhs);
    return sub_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertMultiplyOp(const HloInstruction* instruction) {
    if (instruction->operand_count() != 2) {
        return absl::InvalidArgumentError(
            absl::StrCat("Multiply operation must have 2 operands, got ", instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Value lhs = value_map_[instruction->operand(0)];
    mlir::Value rhs = value_map_[instruction->operand(1)];
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Use afir.mul when emit_afir_dialect is true
    if (options_.emit_afir_dialect) {
        int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
        auto identity_map = GetIdentityAffineMap(rank);

        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        llvm::SmallVector<mlir::AffineMap> maps;
        maps.push_back(identity_map);
        maps.push_back(identity_map);
        maps.push_back(identity_map);
        auto indexing_maps_attr = CreateIndexingMaps(maps);

        auto mul_op = builder_.create<mlir::afir::MulOp>(loc, result_type, lhs, rhs, indexing_maps_attr, nullptr,
                                                         builder_.getDictionaryAttr({}), nullptr, outputs_array);
        return mul_op.getResult();
    }

    // Use arith.mul as fallback
    auto mul_op = builder_.create<mlir::arith::MulIOp>(loc, lhs, rhs);
    return mul_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertDivideOp(const HloInstruction* instruction) {
    if (instruction->operand_count() != 2) {
        return absl::InvalidArgumentError(
            absl::StrCat("Divide operation must have 2 operands, got ", instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Value lhs = value_map_[instruction->operand(0)];
    mlir::Value rhs = value_map_[instruction->operand(1)];
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Use afir.div when emit_afir_dialect is true
    if (options_.emit_afir_dialect) {
        int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
        auto identity_map = GetIdentityAffineMap(rank);

        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        llvm::SmallVector<mlir::AffineMap> maps;
        maps.push_back(identity_map);
        maps.push_back(identity_map);
        maps.push_back(identity_map);
        auto indexing_maps_attr = CreateIndexingMaps(maps);

        auto div_op = builder_.create<mlir::afir::DivOp>(loc, result_type, lhs, rhs, indexing_maps_attr, nullptr,
                                                         builder_.getDictionaryAttr({}), nullptr, outputs_array);
        return div_op.getResult();
    }

    // Use arith.div as fallback
    auto div_op = builder_.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
    return div_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertMaximumOp(const HloInstruction* instruction) {
    if (instruction->operand_count() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Maximum operation must have 2 operands, got ", instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Value lhs = value_map_[instruction->operand(0)];
    mlir::Value rhs = value_map_[instruction->operand(1)];
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Use afir.MaximumOp when emit_afir_dialect is true
    if (options_.emit_afir_dialect) {
    int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
    auto identity_map = GetIdentityAffineMap(rank);

    auto output_attr = CreateDefaultOutputAttr(result_type);
    llvm::SmallVector<mlir::Attribute> output_attrs;
    output_attrs.push_back(output_attr);
    auto outputs_array = builder_.getArrayAttr(output_attrs);

    llvm::SmallVector<mlir::AffineMap> maps;
    maps.push_back(identity_map);
    maps.push_back(identity_map);
    maps.push_back(identity_map);
    auto indexing_maps_attr = CreateIndexingMaps(maps);

    auto max_op = builder_.create<mlir::afir::MaximumOp>(
        loc, result_type, lhs, rhs,
        indexing_maps_attr, nullptr, builder_.getDictionaryAttr({}), nullptr, outputs_array);
    return max_op.getResult();
    }

    return absl::UnimplementedError("Maximum operation requires emit_afir_dialect to be enabled");
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertTanhOp(const HloInstruction* instruction) {
    if (instruction->operand_count() != 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Tanh operation must have 1 operand, got ", instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Value operand = value_map_[instruction->operand(0)];
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Use afir.tanh when emit_afir_dialect is true
    if (options_.emit_afir_dialect) {
        int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
        auto identity_map = GetIdentityAffineMap(rank);

        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        // Build indexing_maps array (2 maps: operand, result)
        llvm::SmallVector<mlir::AffineMap> maps;
        maps.push_back(identity_map);  // operand
        maps.push_back(identity_map);  // result
        auto indexing_maps_attr = CreateIndexingMaps(maps);

        // Create TanhOp with all required attributes
        auto tanh_op = builder_.create<mlir::afir::TanhOp>(loc, result_type, operand, indexing_maps_attr,
                                                           nullptr,                         // loop_axis (use default)
                                                           builder_.getDictionaryAttr({}),  // ir_attr_def (use default)
                                                           nullptr,                         // tmp_buffers (use default)
                                                           outputs_array);
        return tanh_op.getResult();
    }

    // Use math.tanh as fallback
    auto tanh_op = builder_.create<mlir::math::TanhOp>(loc, operand);
    return tanh_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertNegateOp(const HloInstruction* instruction) {
    if (instruction->operand_count() != 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Negate operation must have 1 operand, got ", instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Value operand = value_map_[instruction->operand(0)];
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Use afir.NegOp when emit_afir_dialect is true
    if (options_.emit_afir_dialect) {
        int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
        auto identity_map = GetIdentityAffineMap(rank);

        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        // Build indexing_maps array (2 maps: operand, result)
        llvm::SmallVector<mlir::AffineMap> maps;
        maps.push_back(identity_map);  // operand
        maps.push_back(identity_map);  // result
        auto indexing_maps_attr = CreateIndexingMaps(maps);

        // Create NegOp with all required attributes
        auto neg_op = builder_.create<mlir::afir::NegOp>(loc, result_type, operand, indexing_maps_attr,
                                                          nullptr,                         // loop_axis (use default)
                                                          builder_.getDictionaryAttr({}),  // ir_attr_def (use default)
                                                          nullptr,                         // tmp_buffers (use default)
                                                          outputs_array);
        return neg_op.getResult();
    }

    // Use arith.negf as fallback
    auto neg_op = builder_.create<mlir::arith::NegFOp>(loc, operand);
    return neg_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertExponentialOp(const HloInstruction* instruction) {
    if (instruction->operand_count() != 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Exponential operation must have 1 operand, got ", instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Value operand = value_map_[instruction->operand(0)];
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Use afir.ExpOp when emit_afir_dialect is true
    if (options_.emit_afir_dialect) {
        int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
        auto identity_map = GetIdentityAffineMap(rank);

        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        // Build indexing_maps array (2 maps: operand, result)
        llvm::SmallVector<mlir::AffineMap> maps;
        maps.push_back(identity_map);  // operand
        maps.push_back(identity_map);  // result
        auto indexing_maps_attr = CreateIndexingMaps(maps);

        // Create ExpOp with all required attributes
        auto exp_op = builder_.create<mlir::afir::ExpOp>(loc, result_type, operand, indexing_maps_attr,
                                                          nullptr,                         // loop_axis (use default)
                                                          builder_.getDictionaryAttr({}),  // ir_attr_def (use default)
                                                          nullptr,                         // tmp_buffers (use default)
                                                          outputs_array);
        return exp_op.getResult();
    }

    // Use math.exp as fallback
    auto exp_op = builder_.create<mlir::math::ExpOp>(loc, operand);
    return exp_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertConvertOp(const HloInstruction* instruction) {
    if (instruction->operand_count() != 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Convert operation must have 1 operand, got ", instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Value operand = value_map_[instruction->operand(0)];
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Use afir.CastOp when emit_afir_dialect is true
    if (options_.emit_afir_dialect) {
        int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
        auto identity_map = GetIdentityAffineMap(rank);

        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        // Build indexing_maps array (2 maps: operand, result)
        llvm::SmallVector<mlir::AffineMap> maps;
        maps.push_back(identity_map);  // operand
        maps.push_back(identity_map);  // result
        auto indexing_maps_attr = CreateIndexingMaps(maps);

        // Create CastOp with all required attributes
        auto cast_op = builder_.create<mlir::afir::CastOp>(loc, result_type, operand, indexing_maps_attr,
                                                           nullptr,                         // loop_axis (use default)
                                                           builder_.getDictionaryAttr({}),  // ir_attr_def (use default)
                                                           nullptr,                         // tmp_buffers (use default)
                                                           outputs_array);
        return cast_op.getResult();
    }

    // Use arith.ExtFOp, arith.TruncFOp, arith.IndexCastOp, etc. as fallback
    // For simplicity, we'll use the standard arith operations based on the types
    auto operand_type = llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());
    auto result_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(result_type);

    if (!operand_type || !result_tensor_type) {
        return absl::InternalError("Convert operands must be ranked tensors");
    }

    auto operand_elem_type = operand_type.getElementType();
    auto result_elem_type = result_tensor_type.getElementType();

    // Handle float to float conversions
    if (llvm::isa<mlir::FloatType>(operand_elem_type) && llvm::isa<mlir::FloatType>(result_elem_type)) {
        auto operand_width = llvm::cast<mlir::FloatType>(operand_elem_type).getWidth();
        auto result_width = llvm::cast<mlir::FloatType>(result_elem_type).getWidth();

        if (operand_width < result_width) {
            // Extend floating point
            auto extf_op = builder_.create<mlir::arith::ExtFOp>(loc, result_type, operand);
            return extf_op.getResult();
        } else if (operand_width > result_width) {
            // Truncate floating point
            auto truncf_op = builder_.create<mlir::arith::TruncFOp>(loc, result_type, operand);
            return truncf_op.getResult();
        } else {
            // Same width, just bitcast
            auto bitcast_op = builder_.create<mlir::arith::BitcastOp>(loc, result_type, operand);
            return bitcast_op.getResult();
        }
    }

    // Handle integer to integer conversions
    if (llvm::isa<mlir::IntegerType>(operand_elem_type) && llvm::isa<mlir::IntegerType>(result_elem_type)) {
        auto operand_width = llvm::cast<mlir::IntegerType>(operand_elem_type).getWidth();
        auto result_width = llvm::cast<mlir::IntegerType>(result_elem_type).getWidth();

        if (operand_width < result_width) {
            // Extend integer (signed or unsigned)
            auto operand_signed = llvm::cast<mlir::IntegerType>(operand_elem_type).isSigned();
            if (operand_signed) {
                auto exti_op = builder_.create<mlir::arith::ExtSIOp>(loc, result_type, operand);
                return exti_op.getResult();
            } else {
                auto exti_op = builder_.create<mlir::arith::ExtUIOp>(loc, result_type, operand);
                return exti_op.getResult();
            }
        } else if (operand_width > result_width) {
            // Truncate integer
            auto trunci_op = builder_.create<mlir::arith::TruncIOp>(loc, result_type, operand);
            return trunci_op.getResult();
        } else {
            // Same width, just bitcast
            auto bitcast_op = builder_.create<mlir::arith::BitcastOp>(loc, result_type, operand);
            return bitcast_op.getResult();
        }
    }

    // Handle float to integer conversions
    if (llvm::isa<mlir::FloatType>(operand_elem_type) && llvm::isa<mlir::IntegerType>(result_elem_type)) {
        auto result_signed = llvm::cast<mlir::IntegerType>(result_elem_type).isSigned();
        if (result_signed) {
            auto fptosi_op = builder_.create<mlir::arith::FPToSIOp>(loc, result_type, operand);
            return fptosi_op.getResult();
        } else {
            auto fptoui_op = builder_.create<mlir::arith::FPToUIOp>(loc, result_type, operand);
            return fptoui_op.getResult();
        }
    }

    // Handle integer to float conversions
    if (llvm::isa<mlir::IntegerType>(operand_elem_type) && llvm::isa<mlir::FloatType>(result_elem_type)) {
        auto operand_signed = llvm::cast<mlir::IntegerType>(operand_elem_type).isSigned();
        if (operand_signed) {
            auto sitofp_op = builder_.create<mlir::arith::SIToFPOp>(loc, result_type, operand);
            return sitofp_op.getResult();
        } else {
            auto uitofp_op = builder_.create<mlir::arith::UIToFPOp>(loc, result_type, operand);
            return uitofp_op.getResult();
        }
    }

    return absl::UnimplementedError(
        absl::StrCat("Unsupported type conversion"));
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertReshapeOp(const HloInstruction* instruction) {
    if (instruction->operand_count() != 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Reshape operation must have 1 operand, got ", instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Value operand = value_map_[instruction->operand(0)];
    mlir::Type result_type = ConvertShape(instruction->shape());

    const Shape& input_shape = instruction->operand(0)->shape();
    const Shape& output_shape = instruction->shape();

    // For emit_afir_dialect mode, use AFIR operations to implement reshape
    if (options_.emit_afir_dialect) {
        // Special case: scalar -> [1] reshape
        // Implement this as a broadcast operation
        // Because some ops e.g. concat, do not support input tensor with empty axis.
        if (input_shape.dimensions().size() == 0 && output_shape.dimensions().size() == 1 &&
            output_shape.dimensions(0) == 1) {
            int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();

            // Create indexing maps for broadcast
            // Input (scalar): affine_map<() -> ()>
            // Output ([1]): affine_map<(d0) -> (d0)>
            auto scalar_map = GetIdentityAffineMap(0);
            llvm::SmallVector<mlir::AffineMap> maps;
            maps.push_back(scalar_map);  // scalar input

            auto output_map = GetIdentityAffineMap(rank);
            maps.push_back(output_map);  // [1] output

            auto indexing_maps_attr = CreateIndexingMaps(maps);

            // Create output attributes
            auto output_attr = CreateDefaultOutputAttr(result_type);
            llvm::SmallVector<mlir::Attribute> output_attrs;
            output_attrs.push_back(output_attr);
            auto outputs_array = builder_.getArrayAttr(output_attrs);

            // Create BroadcastOp to implement scalar -> [1] reshape
            auto broadcast_op = builder_.create<mlir::afir::BroadcastOp>(
                loc, result_type, operand, indexing_maps_attr,
                nullptr,  // loop_axis
                builder_.getDictionaryAttr({}),  // ir_attr_def
                nullptr,  // tmp_buffers
                outputs_array);

            return broadcast_op.getResult();
        }

        // Special case: [1] -> scalar reshape
        if (input_shape.dimensions().size() == 1 && input_shape.dimensions(0) == 1 &&
            output_shape.dimensions().size() == 0) {
            // Use broadcast with scalar indexing maps
            auto scalar_map = GetIdentityAffineMap(0);
            llvm::SmallVector<mlir::AffineMap> maps;
            maps.push_back(scalar_map);  // input: scalar map
            maps.push_back(scalar_map);  // output: scalar map

            auto indexing_maps_attr = CreateIndexingMaps(maps);

            auto output_attr = CreateDefaultOutputAttr(result_type);
            llvm::SmallVector<mlir::Attribute> output_attrs;
            output_attrs.push_back(output_attr);
            auto outputs_array = builder_.getArrayAttr(output_attrs);

            auto broadcast_op = builder_.create<mlir::afir::BroadcastOp>(
                loc, result_type, operand, indexing_maps_attr,
                nullptr,
                builder_.getDictionaryAttr({}),
                nullptr,
                outputs_array);

            return broadcast_op.getResult();
        }

        // Check if reshape only adds or removes dimensions of size 1
        // This covers cases like:
        // - [1,128] -> [128] or [128] -> [1,128]
        // - [32,3,1] -> [32,3] or [32,3] -> [32,3,1]
        // - [1,128,1] -> [128] or [128] -> [1,128,1]
        // - [1] -> [1,1] or [1,1] -> [1]
        // - [1] -> [1,1,1] or [1,1,1] -> [1]
        bool is_one_dim_reshape = true;
        int64_t input_idx = 0;
        int64_t output_idx = 0;

        while (input_idx < input_shape.dimensions().size() ||
               output_idx < output_shape.dimensions().size()) {
            // Skip dimensions of size 1 in input
            while (input_idx < input_shape.dimensions().size() &&
                   input_shape.dimensions(input_idx) == 1) {
                input_idx++;
            }
            // Skip dimensions of size 1 in output
            while (output_idx < output_shape.dimensions().size() &&
                   output_shape.dimensions(output_idx) == 1) {
                output_idx++;
            }

            // If both indices are still valid, check if dimensions match
            if (input_idx < input_shape.dimensions().size() &&
                output_idx < output_shape.dimensions().size()) {
                if (input_shape.dimensions(input_idx) != output_shape.dimensions(output_idx)) {
                    is_one_dim_reshape = false;
                    break;
                }
                input_idx++;
                output_idx++;
            } else if (input_idx < input_shape.dimensions().size() ||
                       output_idx < output_shape.dimensions().size()) {
                // One shape still has non-1 dimensions left
                is_one_dim_reshape = false;
                break;
            }
        }

        if (is_one_dim_reshape) {
            // Reshape only adds/removes dimensions of size 1
            // The underlying data layout is the same, so we can just return the operand
            return operand;
        }

        // For more complex reshapes, we currently don't support them
        return absl::UnimplementedError(
            absl::StrCat("Unsupported reshape: ",
                         ShapeUtil::HumanString(input_shape), " -> ",
                         ShapeUtil::HumanString(output_shape),
                         ". Only reshapes that add/remove dimensions of size 1 are currently supported."));
    }

    // For non-afir mode, just return the operand (reshape is no-op in MLIR)
    return operand;
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertReduceOp(const HloInstruction* instruction) {
    // HLO reduce instruction has N data operands followed by N init operands
    // For example, reduce(arg0, arg1, ..., init0, init1, ...)
    // The actual reduction is applied to (arg0, init0), (arg1, init1), etc.

    // Get the number of data operands (total operands / 2)
    int64_t num_data_operands = instruction->operand_count() / 2;
    if (num_data_operands < 1) {
        return absl::InvalidArgumentError("Reduce operation must have at least one data operand");
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Get the reduce computation
    if (!instruction->has_to_apply()) {
        return absl::InvalidArgumentError("Reduce operation must have a to_apply computation");
    }
    const HloComputation* reduce_computation = instruction->to_apply();

    // For simplicity, we'll handle single-operand reduce here
    // Multi-operand reduce would need more complex handling
    if (num_data_operands != 1) {
        return absl::UnimplementedError(
            absl::StrCat("Reduce with ", num_data_operands, " operands is not yet supported"));
    }

    mlir::Value data_operand = value_map_[instruction->operand(0)];

    // Get dimensions to reduce - AFIR reduce ops only support reducing along a single axis
    const auto& dimensions = instruction->dimensions();
    if (dimensions.size() != 1) {
        return absl::UnimplementedError(
            absl::StrCat("AFIR reduce operations only support reducing along a single dimension, got ",
                         dimensions.size(), " dimensions"));
    }
    int64_t reduce_axis = dimensions[0];

    // For AFIR dialect, use the dedicated reduce ops
    if (options_.emit_afir_dialect) {
        auto data_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(data_operand.getType());
        auto result_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(result_type);

        if (!data_tensor_type || !result_tensor_type) {
            return absl::InternalError("Reduce operands must be ranked tensors");
        }

        int64_t data_rank = data_tensor_type.getRank();

        // Build identity indexing map for the input
        auto identity_map = GetIdentityAffineMap(data_rank);

        // Build outputs attribute
        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        // Build indexing_maps array (1 map: input)
        llvm::SmallVector<mlir::AffineMap> maps;
        maps.push_back(identity_map);
        auto indexing_maps_attr = CreateIndexingMaps(maps);

        // Create axis attribute for AFIR reduce op
        auto axis_attr = builder_.getI32IntegerAttr(reduce_axis);

        // Check if the reduce computation corresponds to a known operation
        const HloInstruction* reduce_root = reduce_computation->root_instruction();

        // For common reductions like sum, max, min, we can use dedicated AFIR ops
        if (reduce_root->opcode() == HloOpcode::kAdd) {
            // Sum reduction
            auto reduce_op = builder_.create<mlir::afir::SumOp>(
                loc, result_type, data_operand, axis_attr, indexing_maps_attr,
                nullptr,                         // loop_axis
                builder_.getDictionaryAttr({}),  // ir_attr_def
                nullptr,                         // tmp_buffers
                outputs_array);
            return reduce_op.getResult();
        } else if (reduce_root->opcode() == HloOpcode::kMaximum) {
            // Max reduction
            auto reduce_op = builder_.create<mlir::afir::MaxOp>(
                loc, result_type, data_operand, axis_attr, indexing_maps_attr,
                nullptr,                         // loop_axis
                builder_.getDictionaryAttr({}),  // ir_attr_def
                nullptr,                         // tmp_buffers
                outputs_array);
            return reduce_op.getResult();
        } else if (reduce_root->opcode() == HloOpcode::kMinimum) {
            // Min reduction
            auto reduce_op = builder_.create<mlir::afir::MinOp>(
                loc, result_type, data_operand, axis_attr, indexing_maps_attr,
                nullptr,                         // loop_axis
                builder_.getDictionaryAttr({}),  // ir_attr_def
                nullptr,                         // tmp_buffers
                outputs_array);
            return reduce_op.getResult();
        } else {
            // Generic reduce - not yet supported
            return absl::UnimplementedError(
                absl::StrCat("Reduce computation type '", HloOpcodeString(reduce_root->opcode()),
                             "' is not yet supported for AFIR dialect"));
        }
    }

    // Non-AFIR mode: use MLIR standard reduce operations
    // For now, return unimplemented
    return absl::UnimplementedError("Reduce operation requires emit_afir_dialect to be enabled");
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertConcatenateOp(const HloInstruction* instruction) {
    if (instruction->operand_count() < 2) {
        return absl::InvalidArgumentError(
            absl::StrCat("Concatenate operation must have at least 2 operands, got ",
                         instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Get the concatenate dimension
    const auto& concat_dimensions = instruction->dimensions();
    if (concat_dimensions.size() != 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Concatenate must have exactly 1 dimension, got ", concat_dimensions.size()));
    }
    int64_t concat_dim = concat_dimensions[0];

    // Collect all operands
    llvm::SmallVector<mlir::Value> operands;
    for (const HloInstruction* operand : instruction->operands()) {
        auto it = value_map_.find(operand);
        if (it == value_map_.end()) {
            return absl::InternalError(absl::StrCat("Concatenate operand not found in value map: ", operand->name()));
        }
        operands.push_back(it->second);
    }

    // Use AFIR dialect
    if (options_.emit_afir_dialect) {
        auto result_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(result_type);
        if (!result_tensor_type) {
            return absl::InternalError("Concatenate result must be a ranked tensor");
        }

        int64_t result_rank = result_tensor_type.getRank();

        // Build identity indexing map for inputs (AFIR ConcatOp uses identity map)
        auto identity_map = GetIdentityAffineMap(result_rank);
        llvm::SmallVector<mlir::AffineMap> maps;
        maps.push_back(identity_map);
        auto indexing_maps_attr = CreateIndexingMaps(maps);

        // Build outputs attribute
        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        // Create concat_axis attribute for AFIR concat op
        auto concat_axis_attr = builder_.getI32IntegerAttr(concat_dim);

        // Create ConcatOp with all operands
        auto concat_op = builder_.create<mlir::afir::ConcatOp>(
            loc, result_type, operands, concat_axis_attr, indexing_maps_attr,
            nullptr,                         // loop_axis
            builder_.getDictionaryAttr({}),  // ir_attr_def
            nullptr,                         // tmp_buffers
            outputs_array);

        return concat_op.getResult();
    }

    // Non-AFIR mode: use MLIR standard operations
    // For now, return unimplemented
    return absl::UnimplementedError("Concatenate operation requires emit_afir_dialect to be enabled");
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertBroadcastOp(const HloInstruction* instruction) {
    if (instruction->operand_count() != 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Broadcast operation must have 1 operand, got ", instruction->operand_count()));
    }

    mlir::Location loc = ConvertLocation(instruction);
    mlir::Value operand = value_map_[instruction->operand(0)];
    mlir::Type result_type = ConvertShape(instruction->shape());

    // Use afir.broadcast when emit_afir_dialect is true
    if (options_.emit_afir_dialect) {
        auto result_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(result_type);
        auto operand_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());

        int64_t result_rank = result_tensor_type.getRank();
        int64_t operand_rank = operand_tensor_type.getRank();

        // Get broadcast dimensions from HLO instruction
        const auto& broadcast_dims = instruction->dimensions();

        // Build indexing maps for broadcast
        // For broadcast, we need to map from result dimensions to operand dimensions
        llvm::SmallVector<mlir::AffineExpr> operand_exprs;
        llvm::SmallVector<mlir::AffineExpr> result_exprs;

        // Result map: identity (use all dimensions)
        for (int64_t i = 0; i < result_rank; ++i) {
            result_exprs.push_back(builder_.getAffineDimExpr(i));
        }
        auto result_map = mlir::AffineMap::get(result_rank, 0, result_exprs, context_);

        // Operand map: map result dims to operand dims based on broadcast_dims
        // broadcast_dims specifies which dimensions in the result correspond to the operand's dimensions
        // For example: if broadcasting from (2,) to (3, 2) with broadcast_dims={1},
        // then operand map should be (d0, d1) -> (d1)
        llvm::SmallVector<mlir::AffineExpr> exprs;
        for (int64_t i = 0; i < operand_rank; ++i) {
            if (i < broadcast_dims.size()) {
                // This operand dimension maps to a result dimension
                int64_t result_dim = broadcast_dims[i];
                exprs.push_back(builder_.getAffineDimExpr(result_dim));
            } else {
                // This operand dimension is broadcast (size 1), use constant 0
                exprs.push_back(builder_.getAffineConstantExpr(0));
            }
        }
        auto operand_map = mlir::AffineMap::get(result_rank, 0, exprs, context_);

        // Build outputs attribute
        auto output_attr = CreateDefaultOutputAttr(result_type);
        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(output_attr);
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        // Build indexing_maps array (2 maps: operand, result)
        llvm::SmallVector<mlir::AffineMap> maps;
        maps.push_back(operand_map);  // operand (broadcasted)
        maps.push_back(result_map);   // result
        auto indexing_maps_attr = CreateIndexingMaps(maps);

        // Create BroadcastOp with all required attributes
        auto broadcast_op =
            builder_.create<mlir::afir::BroadcastOp>(loc, result_type, operand, indexing_maps_attr,
                                                     nullptr,                         // loop_axis (use default)
                                                     builder_.getDictionaryAttr({}),  // ir_attr_def (use default)
                                                     nullptr,                         // tmp_buffers (use default)
                                                     outputs_array);
        return broadcast_op.getResult();
    }

    // Fallback: Broadcast is only supported with AFIR dialect
    return absl::UnimplementedError("Broadcast operation requires emit_afir_dialect=true");
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertFusionOp(const HloInstruction* instruction) {
    // Fusion instruction is similar to call - it invokes a fused computation
    // The fused computation should already be converted to a function
    HloComputation* fused_computation = instruction->called_computations()[0];

    // Look up the converted function
    auto it = function_map_.find(fused_computation);
    if (it == function_map_.end()) {
        return absl::InternalError(
            absl::StrCat("Fused computation not found in function map: ", fused_computation->name()));
    }

    mlir::func::FuncOp callee_func = it->second;

    // Collect operands
    llvm::SmallVector<mlir::Value> operands;
    for (const HloInstruction* operand : instruction->operands()) {
        auto operand_it = value_map_.find(operand);
        if (operand_it == value_map_.end()) {
            return absl::InternalError(absl::StrCat("Fusion operand not found in value map: ", operand->name()));
        }
        operands.push_back(operand_it->second);
    }

    mlir::Location loc = ConvertLocation(instruction);

    // Create call operation
    auto call_op = builder_.create<mlir::func::CallOp>(loc, callee_func, operands);

    // Fusion produces a single result
    return call_op.getResult(0);
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertCallOp(const HloInstruction* instruction) {
    // Call instruction invokes a nested computation
    HloComputation* called_computation = instruction->called_computations()[0];

    // Look up the converted function
    auto it = function_map_.find(called_computation);
    if (it == function_map_.end()) {
        return absl::InternalError(
            absl::StrCat("Called computation not found in function map: ", called_computation->name()));
    }

    mlir::func::FuncOp callee_func = it->second;

    // Collect operands
    llvm::SmallVector<mlir::Value> operands;
    for (const HloInstruction* operand : instruction->operands()) {
        auto operand_it = value_map_.find(operand);
        if (operand_it == value_map_.end()) {
            return absl::InternalError(absl::StrCat("Call operand not found in value map: ", operand->name()));
        }
        operands.push_back(operand_it->second);
    }

    mlir::Location loc = ConvertLocation(instruction);

    // Create call operation
    auto call_op = builder_.create<mlir::func::CallOp>(loc, callee_func, operands);

    // Call produces a single result (for non-tuple results)
    return call_op.getResult(0);
}

HloToAfirConverter::ReshapeTypeInfo HloToAfirConverter::GetReshapeEffectiveType(
        const HloInstruction* reshape_instr) {
    ReshapeTypeInfo result;
    result.is_one_dim_reshape = false;

    if (reshape_instr->opcode() != HloOpcode::kReshape || reshape_instr->operand_count() != 1) {
        result.effective_type = ConvertShape(reshape_instr->shape());
        return result;
    }

    const Shape& input_shape = reshape_instr->operand(0)->shape();
    const Shape& output_shape = reshape_instr->shape();

    // Check if reshape only adds/removes dimensions of size 1
    bool is_one_dim = true;
    int64_t input_idx = 0;
    int64_t output_idx = 0;

    while (input_idx < input_shape.dimensions().size() ||
           output_idx < output_shape.dimensions().size()) {
        // Skip dimensions of size 1 in input
        while (input_idx < input_shape.dimensions().size() &&
               input_shape.dimensions(input_idx) == 1) {
            input_idx++;
        }
        // Skip dimensions of size 1 in output
        while (output_idx < output_shape.dimensions().size() &&
               output_shape.dimensions(output_idx) == 1) {
            output_idx++;
        }

        // If both indices are still valid, check if dimensions match
        if (input_idx < input_shape.dimensions().size() &&
            output_idx < output_shape.dimensions().size()) {
            if (input_shape.dimensions(input_idx) != output_shape.dimensions(output_idx)) {
                is_one_dim = false;
                break;
            }
            input_idx++;
            output_idx++;
        } else if (input_idx < input_shape.dimensions().size() ||
                   output_idx < output_shape.dimensions().size()) {
            // One shape still has non-1 dimensions left
            is_one_dim = false;
            break;
        }
    }

    result.is_one_dim_reshape = is_one_dim;
    result.effective_type = is_one_dim ? ConvertShape(input_shape) : ConvertShape(output_shape);
    return result;
}

mlir::Type HloToAfirConverter::ConvertShape(const Shape& shape) {
    if (shape.IsTuple()) {
        // Handle tuple type
        llvm::SmallVector<mlir::Type> element_types;
        for (const Shape& elem_shape : shape.tuple_shapes()) {
            element_types.push_back(ConvertShape(elem_shape));
        }
        return builder_.getTupleType(element_types);
    }

    // Handle tensor type
    mlir::Type element_type = ConvertPrimitiveType(shape.element_type());

    // Create ranked tensor type (even for scalars, use 0-dimensional tensor)
    llvm::ArrayRef<int64_t> shape_dims(shape.dimensions().begin(), shape.dimensions().end());
    return mlir::RankedTensorType::get(shape_dims, element_type);
}

mlir::Type HloToAfirConverter::ConvertPrimitiveType(PrimitiveType prim_type) {
    switch (prim_type) {
        case PRED:
            return builder_.getI1Type();
        case S8:
            return builder_.getI8Type();
        case S16:
            return builder_.getI16Type();
        case S32:
            return builder_.getI32Type();
        case S64:
            return builder_.getI64Type();
        case U8:
            return builder_.getIntegerType(8, /*isSigned=*/false);
        case U16:
            return builder_.getIntegerType(16, /*isSigned=*/false);
        case U32:
            return builder_.getIntegerType(32, /*isSigned=*/false);
        case U64:
            return builder_.getIntegerType(64, /*isSigned=*/false);
        case F16:
            return builder_.getF16Type();
        case F32:
            return builder_.getF32Type();
        case F64:
            return builder_.getF64Type();
        case BF16:
            return builder_.getBF16Type();
        default:
            // Return none type for unsupported types
            return builder_.getNoneType();
    }
}

mlir::Location HloToAfirConverter::ConvertLocation(const HloInstruction* instr) {
    const auto& metadata = instr->metadata();
    if (!metadata.source_file().empty()) {
        return mlir::FileLineColLoc::get(context_, metadata.source_file(), metadata.source_line(),
                                         metadata.source_column());
    }

    if (options_.preserve_names && !instr->name().empty()) {
        return mlir::NameLoc::get(builder_.getStringAttr(instr->name()));
    }

    return builder_.getUnknownLoc();
}

mlir::Location HloToAfirConverter::ConvertLocation(const HloComputation* comp) {
    if (options_.preserve_names && !comp->name().empty()) {
        return mlir::NameLoc::get(builder_.getStringAttr(comp->name()));
    }
    return builder_.getUnknownLoc();
}

mlir::AffineMap HloToAfirConverter::GetIdentityAffineMap(int64_t rank) {
    if (rank == 0) {
        return builder_.getEmptyAffineMap();
    }
    llvm::SmallVector<mlir::AffineExpr> exprs;
    for (int64_t i = 0; i < rank; ++i) {
        exprs.push_back(builder_.getAffineDimExpr(i));
    }
    return mlir::AffineMap::get(rank, 0, exprs, context_);
}

mlir::ArrayAttr HloToAfirConverter::CreateIndexingMaps(llvm::ArrayRef<mlir::AffineMap> maps) {
    llvm::SmallVector<mlir::Attribute> attrs;
    for (auto map : maps) {
        attrs.push_back(mlir::AffineMapAttr::get(map));
    }
    return builder_.getArrayAttr(attrs);
}

mlir::afir::AscTensorGroupsAttr HloToAfirConverter::CreateDefaultOutputAttr(mlir::Type tensor_type) {
    auto ranked_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(tensor_type);
    if (!ranked_tensor_type) {
        // Return a minimal valid attribute for non-ranked tensors
        return mlir::afir::AscTensorGroupsAttr::get(
            context_, {},  // vectorized_axis
            {},            // vectorized_strides
            -1,            // tensor_id
            -1,            // reuse_id
            mlir::afir::PositionConfigAttr::get(context_, mlir::afir::Position::GM, 0, false),
            -1  // position_id
        );
    }

    int64_t rank = ranked_tensor_type.getRank();

    // Create vectorized_axis (all dimensions)
    llvm::SmallVector<int64_t> vectorized_axis;
    for (int64_t i = 0; i < rank; ++i) {
        vectorized_axis.push_back(i);
    }

    // Create vectorized_strides (column-major strides)
    llvm::SmallVector<mlir::Attribute> vectorized_strides;
    int64_t stride = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
        vectorized_strides.push_back(builder_.getI64IntegerAttr(stride));
        stride *= ranked_tensor_type.getDimSize(i);
    }
    std::reverse(vectorized_strides.begin(), vectorized_strides.end());

    // Create position attribute (GM by default)
    auto pos_attr = mlir::afir::PositionConfigAttr::get(context_, mlir::afir::Position::GM,
                                                        0,     // depth
                                                        false  // is_double_buffer
    );

    return mlir::afir::AscTensorGroupsAttr::get(context_, vectorized_axis, vectorized_strides,
                                                -1,  // tensor_id
                                                -1,  // reuse_id
                                                pos_attr,
                                                -1  // position_id
    );
}

void HloToAfirConverter::SetModuleGraphAttr(const HloModule& hlo_module) {
    // Create axes based on the shape of the entry computation parameters
    const HloComputation* entry = hlo_module.entry_computation();
    if (!entry || entry->num_parameters() == 0) {
        return;
    }

    // Find the maximum rank across all parameters
    int64_t max_rank = 0;
    for (int64_t i = 0; i < entry->num_parameters(); ++i) {
        const Shape& param_shape = entry->parameter_instruction(i)->shape();
        int64_t param_rank = param_shape.dimensions_size();
        if (param_rank > max_rank) {
            max_rank = param_rank;
        }
    }

    // Use the first parameter's shape for axis sizes, but ensure we have axes for max_rank
    const Shape& first_param_shape = entry->parameter_instruction(0)->shape();
    int64_t first_rank = first_param_shape.dimensions_size();

    // Create axis attributes
    llvm::SmallVector<mlir::afir::AxisAttr> axes;
    for (int64_t i = 0; i < max_rank; ++i) {
        std::string axis_name = "z" + std::to_string(i);
        std::string size_str = (i < first_rank) ? std::to_string(first_param_shape.dimensions(i)) : "1";

        auto axis_attr = mlir::afir::AxisAttr::get(context_,
                                                   i,  // id
                                                   builder_.getStringAttr(axis_name), mlir::afir::AxisType::Original,
                                                   false,  // bind_block
                                                   builder_.getStringAttr(size_str),
                                                   nullptr,                   // align
                                                   llvm::ArrayRef<int64_t>()  // from
        );
        axes.push_back(axis_attr);
    }

    // Create asc_graph_attr
    auto asc_graph_attr = mlir::afir::AscGraphAttrGroupsAttr::get(context_,
                                                                  -1,  // tiling_key
                                                                  axes, mlir::afir::AscGraphType::Compute,
                                                                  llvm::ArrayRef<mlir::StringAttr>()  // size_var
    );

    // Set module attribute
    getModule()->setAttr("afir.asc_graph_attr", asc_graph_attr);
}

void HloToAfirConverter::SetModuleGraphAttr(const HloComputation& computation) {
    // Check if computation has parameters
    if (computation.num_parameters() == 0) {
        return;
    }

    // Find the maximum rank across all parameters
    int64_t max_rank = 0;
    for (int64_t i = 0; i < computation.num_parameters(); ++i) {
        const Shape& param_shape = computation.parameter_instruction(i)->shape();
        int64_t param_rank = param_shape.dimensions_size();
        if (param_rank > max_rank) {
            max_rank = param_rank;
        }
    }

    // Use the first parameter's shape for axis sizes, but ensure we have axes for max_rank
    const Shape& first_param_shape = computation.parameter_instruction(0)->shape();
    int64_t first_rank = first_param_shape.dimensions_size();

    // Create axis attributes
    llvm::SmallVector<mlir::afir::AxisAttr> axes;
    for (int64_t i = 0; i < max_rank; ++i) {
        std::string axis_name = "z" + std::to_string(i);
        std::string size_str = (i < first_rank) ? std::to_string(first_param_shape.dimensions(i)) : "1";

        auto axis_attr = mlir::afir::AxisAttr::get(context_,
                                                   i,  // id
                                                   builder_.getStringAttr(axis_name), mlir::afir::AxisType::Original,
                                                   false,  // bind_block
                                                   builder_.getStringAttr(size_str),
                                                   nullptr,                   // align
                                                   llvm::ArrayRef<int64_t>()  // from
        );
        axes.push_back(axis_attr);
    }

    // Create asc_graph_attr
    auto asc_graph_attr = mlir::afir::AscGraphAttrGroupsAttr::get(context_,
                                                                  -1,  // tiling_key
                                                                  axes, mlir::afir::AscGraphType::Compute,
                                                                  llvm::ArrayRef<mlir::StringAttr>()  // size_var
    );

    // Set module attribute
    getModule()->setAttr("afir.asc_graph_attr", asc_graph_attr);
}

}  // namespace

// ============================================================================
// Public API Implementation
// ============================================================================

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToAfir(const HloModule& module, mlir::MLIRContext* context,
                                                                   const HloToAfirOptions& options) {
    // Create MLIR module
    mlir::OwningOpRef<mlir::ModuleOp> mlir_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));

    HloToAfirConverter converter(*mlir_module, context, options);
    TF_RETURN_IF_ERROR(converter.Convert(module));

    if (options.verify_module) {
        if (failed(mlir_module->verify())) {
            return absl::InternalError("Generated AFIR module failed verification");
        }
    }

    return mlir_module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloComputationToAfir(const HloComputation& computation,
                                                                              mlir::MLIRContext* context,
                                                                              const HloToAfirOptions& options) {
    // Create MLIR module
    mlir::OwningOpRef<mlir::ModuleOp> mlir_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));

    HloToAfirConverter converter(*mlir_module, context, options);

    // Collect all computations that this computation depends on
    std::set<const HloComputation*> needed_computations;
    std::function<void(const HloComputation*)> collect_dependencies = [&](const HloComputation* comp) {
        if (needed_computations.count(comp)) return;
        needed_computations.insert(comp);

        // Recursively collect fused and called computations
        for (const HloInstruction* instruction : comp->instructions()) {
            if (instruction->opcode() == HloOpcode::kFusion) {
                collect_dependencies(instruction->fused_expression_root()->parent());
            } else if (instruction->opcode() == HloOpcode::kCall || instruction->opcode() == HloOpcode::kConditional) {
                for (const HloComputation* called_comp : instruction->called_computations()) {
                    collect_dependencies(called_comp);
                }
            }
        }
    };

    collect_dependencies(&computation);

    // Convert all needed computations (non-entry first, then entry)
    for (const HloComputation* comp : needed_computations) {
        if (comp == &computation) continue;  // Skip entry computation for now
        TF_ASSIGN_OR_RETURN(auto func, converter.ConvertComputation(*comp, /*is_entry=*/false));
    }

    // Convert entry computation last
    TF_ASSIGN_OR_RETURN(auto func, converter.ConvertComputation(computation, /*is_entry=*/true));

    // Set module-level asc_graph_attr if emit_afir_dialect is enabled
    if (options.emit_afir_dialect) {
        converter.SetModuleGraphAttr(computation);
    }

    if (options.verify_module) {
        if (failed(mlir_module->verify())) {
            return absl::InternalError("Generated AFIR module failed verification");
        }
    }

    return mlir_module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToAfir(const HloModule& module,
                                                                   const HloToAfirOptions& options) {
    // Create context on heap to ensure it lives long enough
    // Note: The caller must ensure the module is used before the context is destroyed
    // For safety, we recommend using the API that takes an explicit context pointer
    static thread_local std::unique_ptr<mlir::MLIRContext> thread_context;
    thread_context = std::make_unique<mlir::MLIRContext>();

    // Load required dialects
    thread_context->loadDialect<mlir::func::FuncDialect>();
    thread_context->loadDialect<mlir::arith::ArithDialect>();
    thread_context->loadDialect<mlir::afir::AFIRDialect>();

    TF_ASSIGN_OR_RETURN(auto mlir_module, ConvertHloToAfir(module, thread_context.get(), options));

    return mlir_module;
}

}  // namespace xla
