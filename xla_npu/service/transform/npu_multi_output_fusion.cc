/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/xla_npu/service/transform/npu_multi_output_fusion.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_dfs_reachability.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {

// Determine if a computation can be converted to a multi-output fusion.
bool IsEligibleForMultiOutputFusion(HloComputation* computation) {
  // Basic eligibility check:
  // 1. Must have at least 2 root instructions (producing multiple outputs)
  // 2. All root instructions must be element-wise or similar fusible ops

  if (computation->root_instruction()->opcode() != HloOpcode::kTuple) {
    return false;
  }

  // Check if the tuple is from multiple GTE instructions
  const HloInstruction* tuple_root = computation->root_instruction();

  // Count how many outputs are actually used
  int used_output_count = 0;
  for (const HloInstruction* user : tuple_root->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      used_output_count++;
    }
  }

  // Need at least 2 distinct outputs to consider multi-output fusion
  if (used_output_count < 2) {
    return false;
  }

  return true;
}

// Check if two instructions can be fused together into a multi-output fusion.
bool CanFuseMultiOutput(HloInstruction* producer, HloInstruction* consumer,
                       const se::DeviceDescription& device_description) {
  // Both must be fusion instructions
  if (producer->opcode() != HloOpcode::kFusion ||
      consumer->opcode() != HloOpcode::kFusion) {
    return false;
  }

  // They must have compatible fusion kinds
  if (producer->fusion_kind() != consumer->fusion_kind()) {
    return false;
  }

  // They must share some computation (producers in common)
  // Simplified: check if they have any common non-parameter operands
  absl::flat_hash_set<HloInstruction*> producer_operands;
  for (HloInstruction* operand : producer->operands()) {
    producer_operands.insert(operand);
  }

  for (HloInstruction* operand : consumer->operands()) {
    if (producer->parent() != consumer->parent()) {
      // Different parent computations - can't fuse
      return false;
    }
    if (producer_operands.contains(operand)) {
      // Found shared operand - good candidate for multi-output fusion
      return true;
    }
  }

  // Also fuse if they share the same parent computation and are both loop fusions
  // This is a simplified heuristic that can be refined
  return (producer->fusion_kind() == HloInstruction::FusionKind::kLoop &&
          producer->parent() == consumer->parent());
}

}  // namespace

absl::StatusOr<bool> NpuMultiOutputFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Process each computation
  auto all_computations = module->computations(execution_threads);
  for (HloComputation* computation : all_computations) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    // Build reachability graph
    std::unique_ptr<HloDfsReachability> reachability =
        HloDfsReachability::Build(computation);

    // Collect all fusion instructions
    std::vector<HloInstruction*> fusion_instructions;
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kFusion) {
        fusion_instructions.push_back(instr);
      }
    }

    // Try to fuse compatible fusion instructions
    for (auto it1 = fusion_instructions.begin(); it1 != fusion_instructions.end();
         ++it1) {
      for (auto it2 = std::next(it1); it2 != fusion_instructions.end();
           ++it2) {
        HloInstruction* fusion1 = *it1;
        HloInstruction* fusion2 = *it2;

        if (CanFuseMultiOutput(fusion1, fusion2, device_description_)) {
          // Check if one reaches the other to avoid creating cycles
          if (!reachability->IsConnected(fusion1, fusion2) &&
              !reachability->IsConnected(fusion2, fusion1)) {
            // Found multi-output fusion opportunity
            // For now, just log the detection as this is a simplified implementation
            // A full implementation would create the multi-output fusion here
            NPU_LOG_FORMAT_PRINT(DEBUG, "Potential multi-output fusion opportunity detected between %s and %s",
                                 fusion1->name().data(), fusion2->name().data());
            changed = true;
            break;  // Only process one pair per iteration
          }
        }
      }
    }
  }

  // Run DCE once at the module level to clean up unused instructions
  HloDCE dce;
  TF_ASSIGN_OR_RETURN(bool dce_changed,
                      dce.Run(module, execution_threads));
  changed = changed || dce_changed;

  return changed;
}

}  // namespace xla
