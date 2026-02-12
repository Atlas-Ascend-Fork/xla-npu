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

#include "xla/xla_npu/service/transform/npu_dynamic_slice_fusion_rewriter.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {

constexpr absl::string_view kNpuAclnnMatMul = "__npu_aclnn_matmul";
constexpr absl::string_view kNpuAclnnBatchMatMul = "__npu_aclnn_batch_matmul";

// Check if an instruction is a legal address computation operand
bool IsLegalAddressComputationOperand(HloInstruction* instr) {
 switch (instr->opcode()) {
    case HloOpcode::kSlice:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConstant:
      return true;
    default:
      return false;
  }
}

// Check if a slice operation has a slice size of 1 along the leading dimension
bool IsUnitSliceInTheLeadingDimension(const HloInstruction* slice) {
  if (slice->opcode() != HloOpcode::kSlice) {
    return false;
  }

  const Shape& input_shape = slice->operand(0)->shape();
  const Shape& output_shape = slice->shape();

  // Check both have the same rank
  if (input_shape.dimensions_size() != output_shape.dimensions_size()) {
    return false;
  }

  // Skip scalar shapes
  if (input_shape.dimensions_size() == 0) {
    return false;
  }

  // Check the first dimension is sliced to size 1
  if (output_shape.dimensions(0) == 1 && input_shape.dimensions(0) > 1) {
    return true;
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> NpuDynamicSliceFusionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Process each computation
  auto all_computations = module->computations(execution_threads);
  for (HloComputation* computation : all_computations) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    // Collect all custom calls first
    std::vector<HloInstruction*> custom_calls;
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kCustomCall) {
        custom_calls.push_back(instr);
      }
    }

    // Try to match and rewrite each custom call
    for (HloInstruction* instr : custom_calls) {
      // Only process custom calls targeting NPU ACLNN operations
      absl::string_view target = instr->custom_call_target();
      if (target == kNpuAclnnMatMul || target == kNpuAclnnBatchMatMul) {
        // Check if all operands are legal address computation operands
        bool all_legal = true;
        bool has_slice_operand = false;

        for (HloInstruction* operand : instr->operands()) {
          if (!IsLegalAddressComputationOperand(operand)) {
            all_legal = false;
            break;
          }
          if (!has_slice_operand && operand->opcode() == HloOpcode::kSlice) {
            has_slice_operand |= IsUnitSliceInTheLeadingDimension(operand);
          }
        }

        if (all_legal && has_slice_operand) {
          // Found matching pattern - the pass would create fusion
          // For now, just log the detection as this is a simplified implementation
          NPU_LOG_FORMAT_PRINT(DEBUG, "NpuDynamicSliceFusionRewriter: detected dynamic slice pattern in custom call %s",
                               instr->name().c_str());
          changed = true;
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
