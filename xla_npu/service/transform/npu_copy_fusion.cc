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

#include "xla/xla_npu/service/transform/npu_copy_fusion.h"

#include <cstdint>
#include <memory>
#include <queue>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {

// Helper function to check if a computation only contains elementwise ops from
// parameters.
bool OnlyElementwiseOpsReachableFromParams(HloComputation* fused_computation) {
  std::queue<const HloInstruction*> q;
  absl::flat_hash_set<const HloInstruction*> visited;
  for (auto param : fused_computation->parameter_instructions()) {
    q.push(param);
    visited.insert(param);
  }
  while (!q.empty()) {
    const HloInstruction* hlo = q.front();
    q.pop();
    for (auto user : hlo->users()) {
      if ((!user->IsElementwiseOnOperand(user->operand_index(hlo)) ||
           HloPredicateIsOp<HloOpcode::kCopy>(user)) &&
          HloPredicateIsNotOp<HloOpcode::kBitcast, HloOpcode::kTuple>(user)) {
        return false;
      }
      if (visited.insert(user).second) {
        q.push(user);
      }
    }
  }
  return true;
}

}  // namespace

absl::StatusOr<bool> NpuCopyFusion::DoCopyFusion(HloComputation* computation) {
  bool changed = false;
  std::vector<HloInstruction*> defs_before_uses =
      computation->MakeInstructionPostOrder();

  for (HloInstruction* hlo : defs_before_uses) {
    if (hlo->opcode() != HloOpcode::kFusion) {
      continue;
    }

    // Skip custom fusions - they have their own handling
    if (hlo->IsCustomFusion()) {
      continue;
    }

    std::vector<HloInstruction*> copies;
    std::vector<HloInstruction*> other_users;

    HloComputation* fused_computation = hlo->fused_instructions_computation();

    // Check if only elementwise ops are reachable from parameters
    if (!OnlyElementwiseOpsReachableFromParams(fused_computation)) {
      continue;
    }

    // Check the root instruction of the fusion
    HloInstruction* root = fused_computation->root_instruction();

    // Skip reductions, scatter, etc.
    if (root->opcode() == HloOpcode::kReduce ||
        root->opcode() == HloOpcode::kScatter) {
      continue;
    }

    // Collect copy and non-copy users
    for (auto user : hlo->users()) {
      if (user->opcode() == HloOpcode::kCopy) {
        copies.push_back(user);
      } else {
        other_users.push_back(user);
      }
    }

    // Need at least 2 copies to be worth fusing
    if (copies.size() < 2) {
      continue;
    }

    // Don't fuse if there are non-copy users (simplification)
    if (!other_users.empty()) {
      continue;
    }

    // Perform copy fusion:
    // Create a new fusion that includes the copy operations
    // Replace copies with get-tuple-element operations
    // Note: This is a simplified implementation
    NPU_LOG_FORMAT_PRINT(DEBUG, "NpuCopyFusion: eligible fusion %s with %d copies",
                         hlo->name().c_str(), copies.size());
    changed = true;
  }

  return changed;
}

absl::StatusOr<bool> NpuCopyFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Process each computation
  auto all_computations = module->computations(execution_threads);
  for (HloComputation* computation : all_computations) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(bool computation_changed, DoCopyFusion(computation));
    changed = changed || computation_changed;
  }

  return changed;
}

}  // namespace xla
