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

#include "xla/xla_npu/service/transform/npu_gemm_fusion.h"

#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/transform/npu_fusion_whitelist.h"

namespace xla {

namespace {

// Default block sizes for NPU GEMM
constexpr int64_t kDefaultBlockM = 128;
constexpr int64_t kDefaultBlockN = 128;
constexpr int64_t kDefaultBlockK = 32;

// Supported element types for NPU GEMM
constexpr PrimitiveType kSupportedGemmElementTypes[] = {
    PrimitiveType::F16, PrimitiveType::BF16, PrimitiveType::F32,
    PrimitiveType::S8,  PrimitiveType::U8,   PrimitiveType::S32,
};

// Check if element type is supported by NPU GEMM
bool IsSupportedGemmElementType(PrimitiveType type) {
  for (PrimitiveType supported : kSupportedGemmElementTypes) {
    if (type == supported) {
      return true;
    }
  }
  return false;
}

// Check if instruction is a scalar constant (used for bias)
bool IsScalarConstant(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kConstant &&
         ShapeUtil::IsEffectiveScalar(instr->shape());
}

// HLO visitor that identifies and fuses GEMM patterns
class GemmFusionVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmFusionVisitor(NpuGemmFusion& fusion) : fusion_(fusion) {}

  absl::Status HandleDot(HloInstruction* dot) override {
    CHECK_EQ(dot->opcode(), HloOpcode::kDot);
    auto& dot_instr = *Cast<HloDotInstruction>(dot);

    if (NpuFusionWhitelist::ShouldSkipFusion(dot)) {
      NPU_LOG_FORMAT_PRINT(WARN,
                           "GEMM instruction %s is whitelisted to skip fusion",
                           dot->name().data());
      return absl::OkStatus();
    }

    // Check if this GEMM should be handled by NPU
    if (!fusion_.ShouldNpuHandleGEMM(dot_instr)) {
      VLOG(3) << "GEMM not handled by NPU: " << dot->ToString();
      return absl::OkStatus();
    }

    // Analyze the fusion pattern
    TF_ASSIGN_OR_RETURN(NpuGemmFusionConfig config,
                        fusion_.AnalyzeGemmFusionPattern(dot_instr));

    // Create the fusion
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_fusion,
        fusion_.CreateDotFusion(dot_instr, dot->GetModule(), config));

    // Replace the original instruction chain with the fusion
    TF_RETURN_IF_ERROR(ReplaceInstruction(
        dot_instr.users().empty() ? &dot_instr : dot_instr.users()[0],
        new_fusion));

    MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  NpuGemmFusion& fusion_;
};

}  // namespace

// Type aliases for protobuf generated types
using npu_backend_config::FusionBackendConfig;
using npu_backend_config::NpuBackendConfig;
using npu_backend_config::NpuGemmConfig;

NpuGemmFusion::NpuGemmFusion(const se::DeviceDescription& device_description,
                             mlir::MLIRContext* mlir_context)
    : device_description_(device_description), mlir_context_(mlir_context) {}

bool NpuGemmFusion::ShouldNpuHandleGEMM(const HloDotInstruction& dot) {
  // Check data types
  if (!IsSupportedGemmElementType(dot.shape().element_type())) {
    return false;
  }
  for (const HloInstruction* operand : dot.operands()) {
    if (!IsSupportedGemmElementType(operand->shape().element_type())) {
      return false;
    }
  }

  // Skip very small matrices
  const Shape& shape = dot.shape();
  if (shape.dimensions_size() < 2) {
    return false;
  }
  int64_t rows = shape.dimensions(shape.dimensions_size() - 2);
  int64_t cols = shape.dimensions(shape.dimensions_size() - 1);
  if (rows * cols < 16) {
    return false;
  }

  // Check for tensor contraction that can be mapped to matmul
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();
  return dim_numbers.lhs_contracting_dimensions_size() == 1 &&
         dim_numbers.rhs_contracting_dimensions_size() == 1;
}

bool NpuGemmFusion::IsGemmSupportedElementwise(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kClamp:
      return true;
    default:
      return false;
  }
}

ActivationMode NpuGemmFusion::GetActivationMode(const HloInstruction* instr) {
  if (!instr) {
    return ActivationMode::kNone;
  }
  switch (instr->opcode()) {
    case HloOpcode::kMaximum:
      return ActivationMode::kRelu;
    case HloOpcode::kLogistic:
      return ActivationMode::kSigmoid;
    case HloOpcode::kTanh:
      return ActivationMode::kTanh;
    default:
      return ActivationMode::kNone;
  }
}

absl::StatusOr<NpuGemmFusionConfig> NpuGemmFusion::AnalyzeGemmFusionPattern(
    const HloDotInstruction& dot) {
  NpuGemmFusionConfig config;
  config.use_nz_layout = true;
  config.block_m = kDefaultBlockM;
  config.block_n = kDefaultBlockN;
  config.block_k = kDefaultBlockK;

  // Trace through the output chain to find bias and activation
  const HloInstruction* current = &dot;

  // First, look for bias addition
  if (current->user_count() == 1) {
    const HloInstruction* next = current->users()[0];
    if (next->opcode() == HloOpcode::kAdd &&
        IsScalarConstant(next->operand(1))) {
      config.has_bias = true;
      current = next;
    }
  }

  // Then, look for activation
  if (current->user_count() == 1) {
    const HloInstruction* next = current->users()[0];
    config.activation = GetActivationMode(next);
  }

  return config;
}

absl::StatusOr<bool> NpuGemmFusion::FuseDotOperand(
    HloInstruction* dot, HloInstruction* operand,
    HloComputation::Builder& builder,
    std::vector<HloInstruction*>& fusion_params) {
  // Simple implementation: check if operand is a constant or needs to be fused
  if (operand->opcode() == HloOpcode::kConstant || !operand->IsFusible()) {
    int param_index = fusion_params.size();
    fusion_params.push_back(operand);
    builder.AddInstruction(HloInstruction::CreateParameter(
        param_index, operand->shape(),
        absl::StrFormat("parameter_%d", param_index)));
    return true;
  }

  // For now, just make it a parameter
  int param_index = fusion_params.size();
  fusion_params.push_back(operand);
  builder.AddInstruction(HloInstruction::CreateParameter(
      param_index, operand->shape(),
      absl::StrFormat("parameter_%d", param_index)));
  return true;
}

absl::StatusOr<bool> NpuGemmFusion::FuseDotOutput(
    HloInstruction* dot, HloComputation::Builder& builder,
    std::vector<HloInstruction*>& fusion_params, NpuGemmFusionConfig& config) {
  const HloInstruction* current = dot;

  // Fuse bias if present
  if (current->user_count() == 1) {
    const HloInstruction* next = current->users()[0];
    if (next->opcode() == HloOpcode::kAdd &&
        IsScalarConstant(next->operand(1))) {
      int param_index = fusion_params.size();
      fusion_params.push_back(const_cast<HloInstruction*>(next->operand(1)));
      builder.AddInstruction(HloInstruction::CreateParameter(
          param_index, next->operand(1)->shape(),
          absl::StrFormat("parameter_%d", param_index)));

      std::vector<HloInstruction*> operands;
      for (int i = 0; i < current->operand_count(); ++i) {
        operands.push_back(const_cast<HloInstruction*>(current->operand(i)));
      }
      operands.push_back(builder.last_added_instruction());

      current = builder.AddInstruction(
          next->CloneWithNewOperands(next->shape(), operands));
    }
  }

  return true;
}

absl::StatusOr<HloInstruction*> NpuGemmFusion::CreateDotFusion(
    const HloDotInstruction& dot, HloModule* module,
    const NpuGemmFusionConfig& config) {
  std::string fusion_name = absl::StrCat("npu_gemm_fusion_", dot.name());
  HloComputation::Builder builder(
      absl::StrCat("npu_gemm_fusion_", dot.name(), "_computation"));

  std::vector<HloInstruction*> fusion_params;

  // Create parameters for each operand
  std::vector<HloInstruction*> new_operands;
  for (int i = 0; i < dot.operand_count(); ++i) {
    const HloInstruction* operand = dot.operand(i);
    int param_index = fusion_params.size();
    fusion_params.push_back(const_cast<HloInstruction*>(operand));
    HloInstruction* param =
        builder.AddInstruction(HloInstruction::CreateParameter(
            param_index, operand->shape(),
            absl::StrFormat("parameter_%d", param_index)));
    new_operands.push_back(param);
  }

  // Clone the dot instruction with new operands
  HloInstruction* fused_dot = builder.AddInstruction(
      dot.CloneWithNewOperands(dot.shape(), new_operands));
  fused_dot->MarkAsRoot();

  // Create the fusion computation
  HloComputation* computation = module->AddComputationAndUnifyNamesAndIds(
      builder.Build(), /*is_entry=*/false);

  // Create the fusion instruction
  // Get mutable computation from module (dot.parent() returns const)
  HloComputation* mutable_parent_computation = nullptr;
  for (auto* comp : module->computations()) {
    if (comp->name() == dot.parent()->name()) {
      mutable_parent_computation = comp;
      break;
    }
  }
  if (!mutable_parent_computation) {
    return absl::NotFoundError(
        "CreateDotFusion Cannot find parent computation");
  }
  HloInstruction* fusion =
      mutable_parent_computation->AddInstruction(HloInstruction::CreateFusion(
          computation->root_instruction()->shape(),
          HloInstruction::FusionKind::kCustom, fusion_params, computation));
  fusion->set_metadata(dot.metadata());
  fusion->GetModule()->SetAndUniquifyInstrName(fusion, fusion_name);

  // Set the backend config
  TF_ASSIGN_OR_RETURN(
      auto npu_config,
      fusion->backend_config<npu_backend_config::NpuBackendConfig>());
  npu_backend_config::FusionBackendConfig& backend_config =
      *npu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kNpuAclnnGemmFusionKind));

  // Set NPU GEMM config
  npu_backend_config::NpuGemmConfig gemm_config;
  gemm_config.set_has_bias(config.has_bias);
  gemm_config.set_activation_mode(static_cast<int32_t>(config.activation));
  gemm_config.set_block_m(config.block_m);
  gemm_config.set_block_n(config.block_n);
  gemm_config.set_block_k(config.block_k);
  *backend_config.mutable_npu_gemm_config() = gemm_config;

  TF_RETURN_IF_ERROR(fusion->set_backend_config(npu_config));

  return fusion;
}

absl::StatusOr<bool> NpuGemmFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    GemmFusionVisitor visitor(*this);
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

}  // namespace xla
