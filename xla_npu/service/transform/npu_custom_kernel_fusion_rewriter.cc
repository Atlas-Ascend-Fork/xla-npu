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

#include "xla/xla_npu/service/transform/npu_custom_kernel_fusion_rewriter.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {

// Helper to check if instruction is a reduce
bool IsReduceOp(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kReduce;
}

// Helper to check if instruction is a broadcast
bool IsBroadcastOp(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kBroadcast;
}

// Helper to check if instruction is a binary elementwise op
bool IsBinaryElementwiseOp(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    case HloOpcode::kDivide:
      return true;
    default:
      return false;
  }
}

// NPU fusion kind constants
constexpr char kNpuCustomFusionKind[] = "__npu_custom";

}  // namespace

absl::StatusOr<bool> NpuCustomKernelFusionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Initialize patterns
  RegisterPatterns();

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool computation_changed,
                        RunOnComputation(computation));
    changed |= computation_changed;
  }
  return changed;
}

void NpuCustomKernelFusionRewriter::RegisterPatterns() {
  // Add softmax pattern
  patterns_.push_back(NpuCustomKernelPattern{
      .name = "acl_softmax",
      .fusion_kind = kNpuCustomFusionKind,
      .match = MatchSoftmaxPattern,
      .rewrite = RewriteSoftmax,
  });

  // Add layer norm pattern
  patterns_.push_back(NpuCustomKernelPattern{
      .name = "acl_layer_norm",
      .fusion_kind = kNpuCustomFusionKind,
      .match = MatchLayerNormPattern,
      .rewrite = RewriteLayerNorm,
  });
}

absl::StatusOr<NpuCustomKernelPattern*>
NpuCustomKernelFusionRewriter::FindMatchingPattern(
    const HloInstruction* instr) {
  for (auto& pattern : patterns_) {
    if (pattern.match({instr})) {
      VLOG(4) << "Pattern matched: " << pattern.name;
      return &pattern;
    }
  }
  return nullptr;
}

bool NpuCustomKernelFusionRewriter::MatchSoftmaxPattern(
    const std::vector<const HloInstruction*>& instrs) {
  if (instrs.empty()) {
    return false;
  }
  const HloInstruction* instr = instrs[0];

  // Look for pattern: reduce-max -> subtract -> exp -> reduce-sum -> divide
  // This is the standard softmax implementation
  if (instr->opcode() != HloOpcode::kDivide) {
    return false;
  }

  const HloInstruction* exp = instr->operand(0);
  if (exp->opcode() != HloOpcode::kExp) {
    return false;
  }

  const HloInstruction* subtract = exp->operand(0);
  if (subtract->opcode() != HloOpcode::kSubtract) {
    return false;
  }

  const HloInstruction* logsumexp_reduce = instr->operand(1);
  if (!logsumexp_reduce || logsumexp_reduce->opcode() != HloOpcode::kReduce ||
      logsumexp_reduce->to_apply()->root_instruction()->opcode() !=
          HloOpcode::kAdd) {
    return false;
  }

  // Check it's a 1D reduction (usually on the last dimension)
  const Shape& result_shape = instr->shape();
  if (result_shape.dimensions_size() < 1) {
    return false;
  }

  return true;
}

bool NpuCustomKernelFusionRewriter::MatchLayerNormPattern(
    const std::vector<const HloInstruction*>& instrs) {
  if (instrs.empty()) {
    return false;
  }
  const HloInstruction* instr = instrs[0];

  // Look for pattern involving mean and variance computation
  // This is a simplified check; real layer norm pattern is more complex
  if (instr->opcode() != HloOpcode::kMultiply &&
      instr->opcode() != HloOpcode::kDivide) {
    return false;
  }

  return false;
}

absl::StatusOr<HloInstruction*> NpuCustomKernelFusionRewriter::RewriteSoftmax(
    const std::vector<const HloInstruction*>& instrs, HloModule* module) {
  if (instrs.empty()) {
    return nullptr;
  }
  const HloInstruction* softmax_root = instrs[0];

  VLOG(3) << "Rewriting softmax: " << softmax_root->ToString();

  HloComputation::Builder builder(
      absl::StrCat("acl_softmax_", softmax_root->name(), "_computation"));
  std::string fusion_name = "acl_softmax_" + std::string(softmax_root->name());

  // Create parameter for the input
  const HloInstruction* input = softmax_root;
  while (input->user_count() > 0 || (input->opcode() == HloOpcode::kDivide ||
                                     input->opcode() == HloOpcode::kExp ||
                                     input->opcode() == HloOpcode::kSubtract)) {
    input = input->operand(0);
  }

  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input->shape(), "input"));

  // Create a simple softmax in the fusion
  // Note: CreateSoftmax is not available in current XLA version
  // Use a simple identity as placeholder - real implementation would construct
  // exp(x - max(x)) / sum(exp(x - max(x))) manually
  HloInstruction* result = builder.AddInstruction(
      HloInstruction::CreateUnary(param->shape(), HloOpcode::kCopy, param));
  result->MarkAsRoot();

  // Create the fusion computation
  HloComputation* computation = module->AddComputationAndUnifyNamesAndIds(
      builder.Build(), /*is_entry=*/false);

  // Create the fusion instruction
  std::vector<HloInstruction*> fusion_params = {
      const_cast<HloInstruction*>(input)};
  // Get mutable computation from module
  HloComputation* mutable_parent_computation = nullptr;
  for (auto* comp : module->computations()) {
    if (comp->name() == softmax_root->parent()->name()) {
      mutable_parent_computation = comp;
      break;
    }
  }
  if (!mutable_parent_computation) {
    return absl::NotFoundError("RewriteSoftmax Cannot find parent computation");
  }
  HloInstruction* fusion =
      mutable_parent_computation->AddInstruction(HloInstruction::CreateFusion(
          computation->root_instruction()->shape(),
          HloInstruction::FusionKind::kCustom, fusion_params, computation));
  fusion->GetModule()->SetAndUniquifyInstrName(fusion, fusion_name);

  // Set the backend config
  TF_ASSIGN_OR_RETURN(
      auto npu_config,
      fusion->backend_config<npu_backend_config::NpuBackendConfig>());
  npu_backend_config::FusionBackendConfig& backend_config =
      *npu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kNpuCustomFusionKind));
  backend_config.set_custom_call_target("acl_softmax");
  TF_RETURN_IF_ERROR(fusion->set_backend_config(npu_config));

  return fusion;
}

absl::StatusOr<HloInstruction*> NpuCustomKernelFusionRewriter::RewriteLayerNorm(
    const std::vector<const HloInstruction*>& instrs, HloModule* module) {
  if (instrs.empty()) {
    return nullptr;
  }
  // Placeholder for layer norm rewriting
  return nullptr;
}

absl::StatusOr<bool> NpuCustomKernelFusionRewriter::RunOnComputation(
    HloComputation* computation) {
  bool changed = false;

  // Visit instructions in post-order
  std::vector<HloInstruction*> instrs = computation->MakeInstructionPostOrder();

  for (HloInstruction* instr : instrs) {
    // Skip if it's already a fusion or parameter
    if (instr->opcode() == HloOpcode::kFusion ||
        instr->opcode() == HloOpcode::kParameter) {
      continue;
    }

    // Try to find a matching pattern
    TF_ASSIGN_OR_RETURN(NpuCustomKernelPattern * pattern,
                        FindMatchingPattern(instr));

    if (pattern == nullptr) {
      continue;
    }

    // Rewrite the pattern
    TF_ASSIGN_OR_RETURN(HloInstruction * fusion,
                        pattern->rewrite({instr}, computation->parent()));

    if (fusion == nullptr) {
      continue;
    }

    // Replace the original instruction
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(instr, fusion));
    changed = true;
  }

  return changed;
}

}  // namespace xla
