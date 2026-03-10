/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

#include "xla/xla_npu/service/transform/npu_priority_fusion.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/transform/npu_fusion_whitelist.h"

namespace xla {

namespace {

// Helper function to check if two instructions have compatible sharding.
// Simplified version that assumes most instructions are sharding-compatible.
bool IsFusibleOnSharding(const HloInstruction* consumer, const HloInstruction* producer) {
    // Simplified check: only return false if sharding is explicitly set and differs
    if (consumer->has_sharding() && producer->has_sharding()) {
        return consumer->sharding() == producer->sharding();
    }
    return true;
}

// Helper function to check basic fusibility constraints.
bool CanFuseNpu(const HloInstruction* consumer, const HloInstruction* producer) {
    // Check if consumer can be fused (XLA-level check)
    // This is required because CreateFusion() has a CHECK for fused_root->IsFusible()
    // and will crash if called with a non-fusible instruction.
    if (!consumer->IsFusible()) {
        NPU_LOG_FORMAT_PRINT(DEBUG, "Consumer %s is not fusible", consumer->name().data());
        return false;
    }

    // Check if producer can be fused (XLA-level check)
    // This is required because FuseInstruction() has a CHECK for IsFusible()
    // and will crash if called with a non-fusible instruction.
    if (!producer->IsFusible()) {
        NPU_LOG_FORMAT_PRINT(DEBUG, "Producer %s is not fusible", producer->name().data());
        return false;
    }

    if (NpuFusionWhitelist::ShouldSkipFusion(consumer)) {
        NPU_LOG_FORMAT_PRINT(WARN, "Consumer %s is whitelisted to skip fusion", consumer->name().data());
        return false;
    }
    if (NpuFusionWhitelist::ShouldSkipFusion(producer)) {
        NPU_LOG_FORMAT_PRINT(WARN, "Producer %s is whitelisted to skip fusion", producer->name().data());
        return false;
    }

    // Don't fuse if producer is a fusion instruction.
    if (producer->opcode() == HloOpcode::kFusion) {
        return false;
    }

    // Don't fuse instructions with different sharding
    if (!IsFusibleOnSharding(consumer, producer)) {
        return false;
    }

    // Only fuse if producer has exactly one user (the consumer)
    if (producer->user_count() != 1) {
        return false;
    }

    // Don't fuse if output shape is too large (simplified heuristic to avoid
    // register pressure issues)
    int64_t output_size = ShapeUtil::ByteSizeOfElements(producer->shape());
    constexpr int64_t kMaxFusionOutputSizeBytes = 128 * 1024;  // 128 KB
    if (output_size > kMaxFusionOutputSizeBytes) {
        return false;
    }

    return true;
}

}  // namespace

HloInstruction::FusionKind NpuPriorityFusion::ChooseKind(const HloInstruction* producer,
                                                         const HloInstruction* consumer) {
    // For NPU, we use kLoop fusion by default for most operations.
    // This can be refined later for specific patterns.
    return HloInstruction::FusionKind::kLoop;
}

HloInstruction* NpuPriorityFusion::Fuse(HloInstruction* producer, HloInstruction* consumer,
                                        bool use_multi_output_fusion) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "NpuPriorityFusion: fusing %s (%s) into %s (%s)", producer->name().data(),
                         HloOpcodeString(producer->opcode()), consumer->name().data(),
                         HloOpcodeString(consumer->opcode()));

    HloComputation* computation = consumer->parent();
    auto kind = ChooseKind(producer, consumer);
    HloInstruction* fusion_instruction = consumer;

    // Phase 1: Create or get fusion instruction
    if (HloPredicateIsNotOp<HloOpcode::kFusion>(consumer)) {
        fusion_instruction =
            computation->AddInstruction(HloInstruction::CreateFusion(consumer->shape(), kind, consumer));
        absl::Status replace_status = consumer->ReplaceAllUsesWith(fusion_instruction);
        if (!replace_status.ok()) {
            NPU_LOG_FORMAT_PRINT(ERROR, "Failed to replace uses of %s with fusion %s: %s", consumer->name().data(),
                                 fusion_instruction->name().data(), replace_status.ToString().data());
            return nullptr;
        }
    } else if (kind != fusion_instruction->fusion_kind()) {
        fusion_instruction->set_fusion_kind(kind);
    }

    fusion_instruction->set_called_computations_execution_thread(computation->execution_thread());

    // Phase 2: Fuse producer into the fusion instruction
    if (HloPredicateIsOp<HloOpcode::kFusion>(producer)) {
        if (use_multi_output_fusion) {
            // Merge producer fusion using multi-output approach
            fusion_instruction->MergeFusionInstructionIntoMultiOutput(producer);
        } else {
            fusion_instruction->MergeFusionInstruction(producer);
        }
    } else {
        if (use_multi_output_fusion) {
            fusion_instruction->FuseInstructionIntoMultiOutput(producer);
            if (producer->user_count() != 0) {
                NPU_LOG_FORMAT_PRINT(ERROR, "Producer %s still has users after multi-output fusion, user_count=%d",
                                     producer->name().data(), producer->user_count());
            }
            absl::Status remove_status = producer->parent()->RemoveInstruction(producer);
            if (!remove_status.ok()) {
                NPU_LOG_FORMAT_PRINT(ERROR, "Failed to remove instruction %s: %s", producer->name().data(),
                                     remove_status.ToString().data());
            }
        } else {
            fusion_instruction->FuseInstruction(producer);
        }
    }

    if (fusion_instruction != consumer) {
        NPU_LOG_FORMAT_PRINT(DEBUG, "NpuPriorityFusion: created new fusion: %s", fusion_instruction->name().data());
    }

    return fusion_instruction;
}

bool NpuPriorityFusion::ConsumeFuel(HloInstruction* producer, HloInstruction* consumer) {
    // Always allow fusion for now. Can be enhanced with fuel mechanism later.
    return true;
}

bool NpuPriorityFusion::CanFuseConstant(const HloInstruction* constant, const HloInstruction* user) {
    // Check if constant is small enough to fuse
    constexpr int64_t kMaxConstantSizeBytes = 256;  // 256 bytes threshold
    if (constant->literal().size_bytes() > kMaxConstantSizeBytes) {
        return false;  // Constant too large
    }

    // Only fuse Broadcast and constant scalar/element-wise operations
    if (user->opcode() == HloOpcode::kBroadcast || user->IsElementwise() || user->opcode() == HloOpcode::kSlice ||
        user->opcode() == HloOpcode::kReshape || user->opcode() == HloOpcode::kTranspose ||
        user->opcode() == HloOpcode::kBitcast) {
        return true;
    }

    return false;  // Constant can only be fused into elementwise-like ops
}

absl::StatusOr<bool> NpuPriorityFusion::RunImpl(HloModule* module,
                                                const absl::flat_hash_set<absl::string_view>& execution_threads) {
    bool changed = false;

    // Copy computations to a vector to avoid iterator invalidation during fusion
    std::vector<HloComputation*> all_computations_list;
    for (HloComputation* computation : module->computations(execution_threads)) {
        all_computations_list.push_back(computation);
    }

    // Process each computation from the copied list
    for (HloComputation* computation : all_computations_list) {
        if (computation == nullptr) {
            NPU_LOG_FORMAT_PRINT(WARN, "NpuPriorityFusion: skipping null computation");
            continue;
        }

        if (computation->IsFusionComputation()) {
            continue;
        }

        // Get all instructions in post-order
        std::vector<HloInstruction*> instructions = computation->MakeInstructionPostOrder();

        // Simple greedy fusion: try to fuse producers into consumers
        // Iterate in reverse post-order (from leaves to roots)
        for (auto consumer_it = instructions.rbegin(); consumer_it != instructions.rend(); ++consumer_it) {
            HloInstruction* consumer = *consumer_it;

            // Skip fusion instructions
            if (consumer->opcode() == HloOpcode::kFusion) {
                continue;
            }

            // Skip instructions with multiple users (simplification)
            if (consumer->user_count() != 0 && consumer->opcode() == HloOpcode::kGetTupleElement) {
                continue;
            }

            // Try to fuse each operand
            for (int i = 0; i < consumer->operand_count(); ++i) {
                HloInstruction* producer = consumer->mutable_operand(i);

                // Handle constant fusion
                if (producer->opcode() == HloOpcode::kConstant) {
                    if (!CanFuseConstant(producer, consumer)) {
                        continue;
                    }
                }

                // Check basic fusibility
                if (!CanFuseNpu(consumer, producer)) {
                    continue;
                }

                // Actually perform fusion
                HloInstruction* fusion = Fuse(producer, consumer, /*use_multi_output_fusion=*/false);
                if (fusion != nullptr) {
                    NPU_LOG_FORMAT_PRINT(DEBUG, "NpuPriorityFusion: successfully fused %s into %s",
                                         producer->name().data(), fusion->name().data());
                    changed = true;
                    break;  // Only process one pair per iteration
                }
            }
        }
    }

    // Run DCE once at the module level to clean up unused instructions
    HloDCE dce;
    TF_ASSIGN_OR_RETURN(bool dce_changed, dce.Run(module, execution_threads));
    changed = changed || dce_changed;

    return changed;
}

}  // namespace xla
