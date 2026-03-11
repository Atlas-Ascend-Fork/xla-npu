/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/service/transform/npu_multi_output_fusion.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_dfs_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/transform/npu_fusion_whitelist.h"

namespace xla {

namespace {

constexpr int64_t MaxOperandsAndOutputsPerFusion() { return 96; }

// Check if operand is profitable for multi-output fusion
bool IsProfitableOperand(HloInstruction* instr) {
    return !ShapeUtil::IsEffectiveScalar(instr->shape());
}

// Check if two instructions are connected (one can reach the other)
bool IsConnected(const HloInstruction* a, const HloInstruction* b,
                 const HloDfsReachability& reachability) {
    return reachability.IsConnected(a, b);
}

// Determine fusion priority for selecting fusion candidate
int FusionPriority(const HloInstruction* instr) {
    if (instr->IsMultiOutputFusion()) {
        return 2;
    }
    if (HloPredicateIsOp<HloOpcode::kFusion>(instr)) {
        return 1;
    }
    return 0;
}

// Determine the fusion kind to be used when fusing into consumer
HloInstruction::FusionKind ChooseFusionKind(const HloInstruction* producer,
                                            const HloInstruction* consumer) {
    // For NPU, we use kLoop fusion by default
    return HloInstruction::FusionKind::kLoop;
}

// Check if instruction is a valid candidate for sibling fusion
bool IsSiblingFusionCandidate(const HloInstruction* instr) {
    if (instr->users().empty()) return false;
    // Only fusion instructions can be candidates
    if (HloPredicateIsNotOp<HloOpcode::kFusion>(instr)) return false;
    // Skip custom fusions
    if (instr->IsCustomFusion()) return false;
    // Multi-output fusions should have GTE users
    if (instr->IsMultiOutputFusion()) {
        return absl::c_all_of(instr->users(), HloPredicateIsOp<HloOpcode::kGetTupleElement>);
    }
    return true;
}

// Check if two sibling instructions can be fused together
bool CanFuseSiblings(const HloInstruction& sibling1, const HloInstruction& sibling2,
                     HloInstruction& parent, const HloDfsReachability& reachability,
                     const se::DeviceDescription& device_description) {
    for (const HloInstruction* operand : sibling1.operands()) {
        if (NpuFusionWhitelist::ShouldSkipFusion(operand)) {
            NPU_LOG_FORMAT_PRINT(WARN, "Operand %s of sibling1 is whitelisted to skip fusion", operand->name().data());
            return false;
        }
    }

    for (const HloInstruction* operand : sibling2.operands()) {
        if (NpuFusionWhitelist::ShouldSkipFusion(operand)) {
            NPU_LOG_FORMAT_PRINT(WARN, "Operand %s of sibling2 is whitelisted to skip fusion", operand->name().data());
            return false;
        }
    }

    // They must not be connected (no cycle)
    if (IsConnected(&sibling1, &sibling2, reachability)) {
        return false;
    }

    // Both must be fusion instructions
    if (sibling1.opcode() != HloOpcode::kFusion ||
        sibling2.opcode() != HloOpcode::kFusion) {
        return false;
    }

    // They must have compatible fusion kinds
    if (sibling1.fusion_kind() != sibling2.fusion_kind()) {
        return false;
    }

    // Simplified check: must fit in the parameter and output budget
    int64_t total_params = sibling1.operand_count() + sibling2.operand_count();
    if (total_params > MaxOperandsAndOutputsPerFusion()) {
        return false;
    }

    return true;
}

// Check if producer-consumer fusion is valid
bool ProducerCandidateIsFusible(const HloInstruction& producer,
                                 const HloInstruction& consumer,
                                 const HloDfsReachability& reachability,
                                 const se::DeviceDescription& device_description) {
    // Consumer must be a fusion instruction
    if (consumer.opcode() != HloOpcode::kFusion) {
        return false;
    }

    // Skip custom fusions
    if (consumer.IsCustomFusion()) {
        return false;
    }

    // Producer must be fusible
    if (!producer.IsFusible()) {
        return false;
    }

    // Skip producer if it's a custom fusion
    if (producer.opcode() == HloOpcode::kFusion && producer.IsCustomFusion()) {
        return false;
    }

    // Check whitelist - skip fusion if producer is in the whitelist
    if (NpuFusionWhitelist::ShouldSkipFusion(&producer)) {
        NPU_LOG_FORMAT_PRINT(WARN, "Producer %s is whitelisted to skip fusion", producer.name().data());
        return false;
    }

    // Check that no operand of consumer is reachable from producer (avoid cycles)
    for (const auto* operand : consumer.operands()) {
        if (&producer != operand && reachability.IsReachable(&producer, operand)) {
            return false;
        }
    }

    // Check parameter budget
    int64_t total_params = consumer.operand_count() + producer.operand_count();
    if (total_params > MaxOperandsAndOutputsPerFusion()) {
        return false;
    }

    return true;
}

// Select the preferred fusion candidate from a list
HloInstruction* SelectPreferredFusionCandidate(const std::vector<HloInstruction*>& candidates) {
    if (candidates.empty()) {
        return nullptr;
    }
    return *std::max_element(candidates.begin(), candidates.end(),
                             [](const HloInstruction* a, const HloInstruction* b) {
                                 return FusionPriority(a) < FusionPriority(b);
                             });
}

}  // namespace

static HloComputation* computation_ = nullptr;
static std::unique_ptr<HloDfsReachability> reachability_;

void NpuMultiOutputFusion::RecomputeReachability() {
    reachability_ = HloDfsReachability::Build(computation_);
}

bool NpuMultiOutputFusion::FuseSiblings(HloInstruction* parent, HloCostAnalysis* cost_analysis) {
    if (!IsProfitableOperand(parent)) {
        return false;
    }

    bool changed = false;
    std::vector<HloInstruction*> siblings;

    // Only consider siblings that are fusion candidates
    absl::c_copy_if(parent->users(), std::back_inserter(siblings),
                    [](const HloInstruction* instr) {
                        return IsSiblingFusionCandidate(instr);
                    });

    // Sort siblings by fusion priority
    absl::c_stable_sort(siblings,
                        [](const HloInstruction* a, const HloInstruction* b) {
                            return FusionPriority(a) > FusionPriority(b);
                        });

    // Try to fuse siblings
    for (auto i = siblings.begin(); i != siblings.end(); ++i) {
        if ((*i)->opcode() != HloOpcode::kFusion) continue;

        for (auto j = i + 1; j != siblings.end();) {
            if ((*j)->opcode() != HloOpcode::kFusion) {
                ++j;
                continue;
            }

            if (!CanFuseSiblings(**i, **j, *parent, *reachability_, device_description_)) {
                ++j;
                continue;
            }

            // Fuse j into i
            HloInstruction* remaining = *i;
            HloInstruction* fused = *j;

            TF_CHECK_OK(cost_analysis->RemoveInstruction(remaining));
            TF_CHECK_OK(cost_analysis->RemoveInstruction(fused));

            if (HloPredicateIsOp<HloOpcode::kFusion>(fused)) {
                remaining->MergeFusionInstructionIntoMultiOutput(fused);
            } else {
                remaining->FuseInstructionIntoMultiOutput(fused);
                CHECK_EQ(0, fused->user_count());
                TF_CHECK_OK(computation_->RemoveInstruction(fused));
            }

            TF_CHECK_OK(cost_analysis->RevisitInstruction(remaining));
            changed = true;
            siblings.erase(j);
            RecomputeReachability();
        }
    }
    return changed;
}

absl::StatusOr<bool> NpuMultiOutputFusion::DoMultiOutputFusion() {
    bool changed = false;
    RecomputeReachability();

    HloCostAnalysis cost_analysis({shape_size_function_,
                                     /*per_second_rates=*/{},
                                     /*min_latencies_seconds=*/{},
                                     /*count_multiple_input_accesses=*/true});
    TF_RETURN_IF_ERROR(computation_->Accept(&cost_analysis));

    std::vector<HloInstruction*> defs_before_uses = computation_->MakeInstructionPostOrder();

    // Traverse in uses-before-defs order (reverse post-order)
    for (auto it = defs_before_uses.rbegin(); it != defs_before_uses.rend(); ++it) {
        HloInstruction* producer = *it;

        // Skip constants
        if (HloPredicateIsOp<HloOpcode::kConstant>(producer)) continue;
        if (producer->IsCustomFusion()) continue;

        // First, fuse sibling consumers
        if (FuseSiblings(/*parent=*/producer, &cost_analysis)) {
            changed = true;
        }

        // Second, perform producer-consumer multi-output fusion
        std::vector<HloInstruction*> fusion_candidates;
        for (HloInstruction* consumer : producer->users()) {
            // Only consider fusion instructions
            if (HloPredicateIsNotOp<HloOpcode::kFusion>(consumer)) continue;

            if (ProducerCandidateIsFusible(*producer, *consumer, *reachability_, device_description_)) {
                fusion_candidates.push_back(consumer);
            }
        }

        HloInstruction* consumer_for_fusion = SelectPreferredFusionCandidate(fusion_candidates);
        if (consumer_for_fusion == nullptr) continue;

        // Perform producer-consumer fusion
        changed = true;
        HloInstruction* input_fusion;

        if (HloPredicateIsOp<HloOpcode::kFusion>(consumer_for_fusion)) {
            input_fusion = consumer_for_fusion;
        } else {
            input_fusion = computation_->AddInstruction(HloInstruction::CreateFusion(
                consumer_for_fusion->shape(),
                ChooseFusionKind(producer, consumer_for_fusion),
                consumer_for_fusion));
            TF_CHECK_OK(computation_->ReplaceInstruction(consumer_for_fusion, input_fusion));
        }

        if (HloPredicateIsOp<HloOpcode::kFusion>(producer)) {
            input_fusion->MergeFusionInstructionIntoMultiOutput(producer);
        } else {
            input_fusion->FuseInstructionIntoMultiOutput(producer);
            CHECK_EQ(0, producer->user_count());
            TF_CHECK_OK(computation_->RemoveInstruction(producer));
        }

        TF_RETURN_IF_ERROR(cost_analysis.RevisitInstruction(input_fusion));
        RecomputeReachability();
    }
    return changed;
}

absl::StatusOr<bool> NpuMultiOutputFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
    bool changed = false;
    for (auto* computation : module->MakeNonfusionComputations()) {
        computation_ = computation;
        TF_ASSIGN_OR_RETURN(bool computation_changed, DoMultiOutputFusion());
        changed |= computation_changed;
    }
    return changed;
}

}  // namespace xla
