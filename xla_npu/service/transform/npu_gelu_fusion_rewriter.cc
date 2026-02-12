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

#include "xla/xla_npu/service/transform/npu_gelu_fusion_rewriter.h"

#include <cmath>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/backend_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"

namespace xla {

namespace {

// GELU fusion kind identifier for NPU backend config.
constexpr absl::string_view kGeluFusionKind = "gelu";

// Check if a float32 scalar constant has a value close to target
bool ScalarConstantMatchesF32(const HloInstruction* instr, float target, float epsilon = 1e-6f) {
    if (instr->opcode() != HloOpcode::kConstant) {
        return false;
    }
    if (!ShapeUtil::IsScalar(instr->shape())) {
        return false;
    }

    const Literal& literal = instr->literal();
    PrimitiveType type = literal.shape().element_type();

    if (type == F32) {
        float value = literal.GetFirstElement<float>();
        return std::abs(value - target) < epsilon;
    } else if (type == F16) {
        float value = static_cast<float>(literal.GetFirstElement<Eigen::half>());
        return std::abs(value - target) < epsilon;
    } else if (type == BF16) {
        float value = static_cast<float>(literal.GetFirstElement<bfloat16>());
        return std::abs(value - target) < epsilon;
    }

    return false;
}

// Collect all instructions in a subgraph in post-order
// This ensures we process dependencies before their users
void CollectInstructionsPostOrder(HloInstruction* root, absl::flat_hash_set<HloInstruction*>& visited,
                                  std::vector<HloInstruction*>& result) {
    if (!visited.insert(root).second) {
        return;  // Already visited
    }

    // Skip parameters - they will be fusion inputs
    if (root->opcode() == HloOpcode::kParameter) {
        return;
    }

    // Recursively process operands first (post-order)
    for (HloInstruction* operand : root->operands()) {
        CollectInstructionsPostOrder(operand, visited, result);
    }

    result.push_back(root);
}

// Check if instruction computes GELU CDF (cumulative distribution function)
// CDF(x) = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
bool IsGeluCdfPattern(HloInstruction* instr, absl::flat_hash_set<HloInstruction*>& cdf_instrs) {
    cdf_instrs.clear();
    absl::flat_hash_set<HloInstruction*> visited;

    // Collect all instructions in post-order
    std::vector<HloInstruction*> collected;
    CollectInstructionsPostOrder(instr, visited, collected);

    // Check if this is a CDF pattern
    bool has_tanh = false, has_0_5 = false, has_0_0447 = false, has_sqrt_2_pi = false, has_1 = false;
    float sqrt_2_pi = std::sqrt(2.0f / M_PI);

    for (HloInstruction* i : collected) {
        if (i->opcode() == HloOpcode::kTanh) {
            has_tanh = true;
        }
        if (ScalarConstantMatchesF32(i, 0.5f)) {
            has_0_5 = true;
        }
        if (ScalarConstantMatchesF32(i, 0.044715f)) {
            has_0_0447 = true;
        }
        if (ScalarConstantMatchesF32(i, sqrt_2_pi)) {
            has_sqrt_2_pi = true;
        }
        if (ScalarConstantMatchesF32(i, 1.0f)) {
            has_1 = true;
        }
    }

    // Must have tanh and 0.5, and either 0.0447, sqrt(2/pi), or 1
    if (!has_tanh || !has_0_5) {
        return false;
    }

    if (!(has_0_0447 || has_sqrt_2_pi || has_1)) {
        return false;
    }

    cdf_instrs.insert(collected.begin(), collected.end());
    return true;
}

// Identify complete GELU pattern starting from a multiply instruction
// GELU(x) = x * CDF(x) where CDF(x) = 0.5 * (1 + tanh(...))
// Returns true if this is a GELU output and fills the to_fuse vector with ALL instructions
bool IdentifyAndCollectGeluPattern(HloInstruction* multiply_instr, std::vector<HloInstruction*>& to_fuse) {
    to_fuse.clear();

    if (multiply_instr->opcode() != HloOpcode::kMultiply || multiply_instr->operand_count() != 2) {
        return false;
    }

    HloInstruction* op0 = const_cast<HloInstruction*>(multiply_instr->operand(0));
    HloInstruction* op1 = const_cast<HloInstruction*>(multiply_instr->operand(1));

    // Find which operand might be CDF (the other should be x)
    absl::flat_hash_set<HloInstruction*> cdf_instrs;
    HloInstruction* cdf_candidate = nullptr;

    // Try op0 as CDF
    if (IsGeluCdfPattern(op0, cdf_instrs)) {
        cdf_candidate = op0;
    } else if (IsGeluCdfPattern(op1, cdf_instrs)) {
        // Try op1 as CDF
        cdf_candidate = op1;
    }

    if (!cdf_candidate) {
        return false;
    }

    // Now we have confirmed this is GELU = x * CDF(x)
    // Collect CDF instructions and the outer multiplication
    absl::flat_hash_set<HloInstruction*> visited;
    CollectInstructionsPostOrder(multiply_instr, visited, to_fuse);

    NPU_LOG_FORMAT_PRINT(
        INFO, "NPU GELU fusion: identified complete GELU pattern with %d instructions (including outer multiply)",
        static_cast<int>(to_fuse.size()));
    return true;
}

}  // anonymous namespace

absl::StatusOr<bool> NpuGeluFusionRewriter::RunImpl(HloModule* module,
                                                    const absl::flat_hash_set<absl::string_view>& execution_threads) {
    bool changed = false;

    for (HloComputation* computation : module->MakeNonfusionComputations(execution_threads)) {
        // Collect all multiply instructions as potential GELU outputs
        std::vector<HloInstruction*> multiply_instructions;
        for (HloInstruction* instr : computation->instructions()) {
            if (instr->opcode() != HloOpcode::kMultiply) {
                continue;
            }
            if (instr->operand_count() != 2) {
                continue;
            }
            multiply_instructions.push_back(instr);
        }

        // Try to identify and fuse GELU patterns
        for (HloInstruction* potential_output : multiply_instructions) {
            if (potential_output->IsDead()) {
                continue;
            }

            std::vector<HloInstruction*> to_fuse;
            if (!IdentifyAndCollectGeluPattern(potential_output, to_fuse)) {
                continue;
            }

            // Check if output is root of computation or has only one user
            if (potential_output->user_count() > 1) {
                // Output is used by multiple consumers, skip
                NPU_LOG_FORMAT_PRINT(WARN, "NPU GELU fusion: output %s has %d users, skipping",
                                     potential_output->name().data(), potential_output->user_count());
                continue;
            }

            NPU_LOG_FORMAT_PRINT(INFO, "NPU GELU fusion: matched complete GELU pattern at %s",
                                 potential_output->name().data());

            // Check if this is the root instruction
            bool is_root = (computation->root_instruction() == potential_output);

            // Create fusion from the output instruction (following NpuSuperFusion pattern)
            HloInstruction* fusion_instr = computation->AddInstruction(HloInstruction::CreateFusion(
                potential_output->shape(), HloInstruction::FusionKind::kLoop, potential_output));

            // Replace uses with fusion instruction
            TF_RETURN_IF_ERROR(potential_output->ReplaceAllUsesWith(fusion_instr));

            // Now fuse all other instructions in post-order reverse
            // This follows the NpuSuperFusion pattern
            for (auto it = to_fuse.rbegin(); it != to_fuse.rend(); ++it) {
                HloInstruction* instr = *it;
                if (instr == potential_output) {
                    continue;  // Already fused as the root
                }
                fusion_instr->FuseInstruction(instr);
            }

            // If we fused the root, update root instruction
            if (is_root) {
                computation->set_root_instruction(fusion_instr);
            }

            // Set backend config using NPU protobuf to mark this as a GELU fusion.
            // Following GPU pattern from gemm_rewriter.cc with BackendConfig.
            npu_backend_config::BackendConfig backend_config;
            npu_backend_config::FusionBackendConfig* fusion_config = backend_config.mutable_fusion_backend_config();
            fusion_config->set_kind(std::string(kGeluFusionKind));
            TF_ASSIGN_OR_RETURN(std::string backend_config_str, BackendConfigToRawString(backend_config));
            fusion_instr->set_raw_backend_config_string(backend_config_str);

            NPU_LOG_FORMAT_PRINT(INFO, "NPU GELU fusion: created fusion %s with kind=kLoop and backend_config='%s'",
                                 fusion_instr->name().data(), backend_config_str.c_str());
            NPU_LOG_FORMAT_PRINT(INFO, "NPU GELU fusion: fused %d instructions into one GELU operation",
                                 static_cast<int>(to_fuse.size()));
            NPU_LOG_FORMAT_PRINT(INFO, "NPU GELU fusion: fusion instruction: %s", fusion_instr->ToString().data());

            changed = true;

            // Break after first successful fusion for this computation
            break;
        }
    }

    // Run DCE to clean up unused instructions
    HloDCE dce;
    TF_ASSIGN_OR_RETURN(bool dce_changed, dce.Run(module, execution_threads));
    changed |= dce_changed;

    return changed;
}

}  // namespace xla
