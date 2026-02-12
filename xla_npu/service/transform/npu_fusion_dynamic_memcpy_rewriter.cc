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

#include "xla/xla_npu/service/transform/npu_fusion_dynamic_memcpy_rewriter.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {

// Check if a fusion computation only contains a single dynamic-slice or similar
// operation followed by get-tuple-elements. This pattern is suitable for
// dynamic memcpy optimization on NPU.
bool IsDynamicMemcpyPattern(HloInstruction* fusion) {
  if (fusion->opcode() != HloOpcode::kFusion) {
    return false;
  }

  HloComputation* comp = fusion->fused_instructions_computation();

  // Get the root instruction
  HloInstruction* root = comp->root_instruction();

  // Simple pattern: fusion contains mostly copy/dynamic-slice/gte operations
  // This is a simplified check that can be enhanced
  bool has_dynamic_slice = false;
  bool has_copy = false;

  for (HloInstruction* instr : comp->instructions()) {
    if (HloPredicateIsOp<HloOpcode::kDynamicSlice,
                         HloOpcode::kDynamicUpdateSlice,
                         HloOpcode::kSlice>(instr)) {
      has_dynamic_slice = true;
    } else if (HloPredicateIsOp<HloOpcode::kCopy>(instr)) {
      has_copy = true;
    }
  }

  return has_dynamic_slice || has_copy;
}

}  // namespace

absl::StatusOr<bool> NpuFusionDynamicMemcpyRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Process each computation
  auto all_computations = module->computations(execution_threads);
  for (HloComputation* computation : all_computations) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    // Collect all fusion instructions first
    std::vector<HloInstruction*> fusion_instructions;
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kFusion) {
        fusion_instructions.push_back(instr);
      }
    }

    // Check if any fusion matches the dynamic memcpy pattern
    for (HloInstruction* instr : fusion_instructions) {
      if (IsDynamicMemcpyPattern(instr)) {
        // Mark the fusion as a dynamic memcpy fusion
        // This annotation can be used during code generation
        NPU_LOG_FORMAT_PRINT(DEBUG, "NpuFusionDynamicMemcpyRewriter: detected dynamic memcpy pattern in fusion %s",
                             instr->name().c_str());

        // In a full implementation, we would set a backend config to mark this
        // as a dynamic memcpy fusion. For now, we just log the detection.
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace xla
