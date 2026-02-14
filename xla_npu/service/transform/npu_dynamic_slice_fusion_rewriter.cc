/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/xla_npu/service/transform/npu_dynamic_slice_fusion_rewriter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/call_graph.h"
#include "xla/shape.h"
#include "xla/util.h"

namespace xla {

namespace {

constexpr int64_t kNpuAllocatedBufferAlignBytes = 128;

// NPU specific custom call targets
constexpr absl::string_view kNpuAclnnMatMul = "__npu_aclnn_matmul";
constexpr absl::string_view kNpuAclnnBatchMatMul = "__npu_aclnn_batch_matmul";

using InstructionSet = absl::flat_hash_set<HloInstruction*>;

// Check if instruction is a NPU custom call
bool IsNpuCustomCall(const HloInstruction* hlo) {
    auto* custom_call = DynCast<HloCustomCallInstruction>(hlo);
    if (custom_call == nullptr) {
        return false;
    }

    const std::string call_target_name = custom_call->custom_call_target();
    return call_target_name == kNpuAclnnMatMul || call_target_name == kNpuAclnnBatchMatMul;
}

// Simplified version of IsAlignedSlice for NPU
bool IsAlignedSlice(const HloInstruction* slice) {
    if (!IsOpcodeAnyOf<HloOpcode::kSlice, HloOpcode::kDynamicSlice,
                       HloOpcode::kDynamicUpdateSlice>(slice)) {
        return false;
    }
    // For NPU, we allow all slice operations
    return true;
}

// Get pattern captures - instructions outside the matched pattern that are operands
absl::InlinedVector<HloInstruction*, 4> GetPatternCaptures(
    const absl::InlinedVector<HloInstruction*, 4>& matches) {
    absl::InlinedVector<HloInstruction*, 4> captures;
    InstructionSet matched_instrs(matches.begin(), matches.end());

    for (HloInstruction* instr : matches) {
        for (HloInstruction* operand : instr->operands()) {
            if (!matched_instrs.contains(operand) && absl::c_find(captures, operand) == captures.end()) {
                captures.push_back(operand);
            }
        }
    }

    return captures;
}

// Create fusion body
absl::StatusOr<HloComputation*> CreateFusionBody(
    HloModule* module,
    const absl::InlinedVector<absl::InlinedVector<HloInstruction*, 4>, 4>& flattened_sliced_operand_paths,
    const absl::InlinedVector<absl::InlinedVector<HloInstruction*, 4>, 4>& flattened_sliced_user_paths,
    const absl::InlinedVector<HloInstruction*, 4>& captures) {
    HloComputation::Builder builder("dynamic-slice-fusion");

    // A mapping from original instructions to instructions in the fusion body.
    absl::flat_hash_map<const HloInstruction*, HloInstruction*> instr_mapping;

    auto mapped_operands = [&](HloInstruction* instr) {
        absl::InlinedVector<HloInstruction*, 4> operands;
        for (HloInstruction* operand : instr->operands()) {
            operands.push_back(instr_mapping.at(operand));
        }
        return operands;
    };

    // For every captured value create a parameter instruction in the computation body
    for (const HloInstruction* capture : captures) {
        int64_t index = instr_mapping.size();
        instr_mapping[capture] = builder.AddInstruction(HloInstruction::CreateParameter(
            index, capture->shape(), absl::StrFormat("p%d", index)));
    }

    // Instructions in the pattern are already topologically sorted
    HloInstruction* hero = nullptr;
    for (const auto& operand_path : flattened_sliced_operand_paths) {
        for (HloInstruction* instr : operand_path) {
            instr_mapping[instr] = builder.AddInstruction(
                instr->CloneWithNewOperands(instr->shape(), mapped_operands(instr)));
            hero = instr;
        }
    }

    for (const auto& user_path : flattened_sliced_user_paths) {
        for (HloInstruction* instr : user_path) {
            instr_mapping[instr] = builder.AddInstruction(
                instr->CloneWithNewOperands(instr->shape(), mapped_operands(instr)));
        }
    }

    // Create a tuple root for multi-output fusions
    if (hero && hero->shape().IsTuple() && hero->shape().tuple_shapes().size() > 0) {
        std::vector<HloInstruction*> elements;
        for (size_t i = 0; i < hero->shape().tuple_shapes().size(); ++i) {
            auto* gte = builder.AddInstruction(HloInstruction::CreateGetTupleElement(instr_mapping.at(hero), i));
            if (hero->shape().tuple_shapes(i).IsTuple()) {
                gte = builder.AddInstruction(HloInstruction::CreateTuple({gte}));
            }
            elements.push_back(gte);
        }
        builder.AddInstruction(HloInstruction::CreateTuple(elements));
    }

    return module->AddComputationAndUnifyNamesAndIds(builder.Build(), false);
}

// Create fusion instruction
absl::StatusOr<HloInstruction*> CreateFusionInstruction(
    HloModule* module, HloInstruction* orig,
    const absl::InlinedVector<HloInstruction*, 4>& captures,
    HloComputation* body, bool dynamic) {
    HloComputation* parent = orig->parent();

    // Add a fusion operation calling outlined fusion computation
    HloInstruction* fusion = parent->AddInstruction(HloInstruction::CreateFusion(
        body->root_instruction()->shape(), HloInstruction::FusionKind::kLoop,
        captures, body));
    module->SetAndUniquifyInstrName(fusion, "dynamic_slice_fusion");

    return fusion;
}

// Simplified version of GetSlicedOperandPaths for NPU
absl::InlinedVector<absl::InlinedVector<HloInstruction*, 4>, 4> GetSlicedOperandPaths(
    const HloInstruction& instr, const CallGraph& call_graph) {
    absl::InlinedVector<absl::InlinedVector<HloInstruction*, 4>, 4> sliced_operand_paths;
    InstructionSet processed_instrs;

    for (const auto* operand : instr.operands()) {
        absl::InlinedVector<HloInstruction*, 4> maybe_sliced_operand_path;
        bool slice_found = false;

        std::vector<HloInstruction*> visit_stack = {const_cast<HloInstruction*>(operand)};
        while (!visit_stack.empty()) {
            HloInstruction* cur = visit_stack.back();
            visit_stack.pop_back();

            if (processed_instrs.contains(cur)) {
                continue;
            }

            maybe_sliced_operand_path.push_back(cur);

            if (IsOpcodeAnyOf<HloOpcode::kDynamicSlice, HloOpcode::kSlice,
                            HloOpcode::kDynamicUpdateSlice>(cur)) {
                if (IsAlignedSlice(cur)) {
                    slice_found = true;
                    break;
                }
            }

            if (!IsOpcodeAnyOf<HloOpcode::kBitcast, HloOpcode::kTuple, HloOpcode::kGetTupleElement>(cur)) {
                break;
            }

            // Continue traversing operands
            for (HloInstruction* op : cur->operands()) {
                if (!processed_instrs.contains(op)) {
                    visit_stack.push_back(op);
                }
            }
        }

        if (slice_found || processed_instrs.contains(maybe_sliced_operand_path.front())) {
            sliced_operand_paths.push_back(maybe_sliced_operand_path);
            processed_instrs.insert(maybe_sliced_operand_path.begin(),
                                    maybe_sliced_operand_path.end());
        }
    }

    sliced_operand_paths.push_back({const_cast<HloInstruction*>(&instr)});
    return sliced_operand_paths;
}

// Simplified version of GetSlicedUserPaths for NPU
absl::InlinedVector<absl::InlinedVector<HloInstruction*, 4>, 4> GetSlicedUserPaths(
    const HloInstruction& instr, const CallGraph& call_graph) {
    absl::InlinedVector<absl::InlinedVector<HloInstruction*, 4>, 4> sliced_user_paths;
    InstructionSet processed_instrs;

    auto traverse_hlo_and_collect = [&](HloInstruction* start) {
        absl::InlinedVector<HloInstruction*, 4> maybe_sliced_user_path;
        bool dus_found = false;

        std::vector<HloInstruction*> visit_stack = {start};
        while (!visit_stack.empty()) {
            HloInstruction* cur = visit_stack.back();
            visit_stack.pop_back();

            if (processed_instrs.contains(cur)) {
                continue;
            }

            maybe_sliced_user_path.push_back(cur);

            if (auto* slice_instr = DynCast<HloDynamicUpdateSliceInstruction>(cur)) {
                if (IsAlignedSlice(slice_instr)) {
                    dus_found = true;
                    break;
                }
            }

            if (cur->user_count() > 1 || !IsOpcodeAnyOf<HloOpcode::kBitcast,
                HloOpcode::kTuple, HloOpcode::kGetTupleElement>(cur)) {
                break;
            }

            // Continue traversing users
            for (HloInstruction* user : cur->users()) {
                if (!processed_instrs.contains(user)) {
                    visit_stack.push_back(user);
                }
            }
        }

        if (dus_found || processed_instrs.contains(maybe_sliced_user_path.front())) {
            processed_instrs.insert(maybe_sliced_user_path.begin(),
                                    maybe_sliced_user_path.end());
            sliced_user_paths.push_back(std::move(maybe_sliced_user_path));
        }
    };

    if (instr.shape().IsTuple()) {
        for (auto* user : instr.users()) {
            if (DynCast<HloGetTupleElementInstruction>(user)) {
                traverse_hlo_and_collect(user);
            }
        }
    } else {
        if (instr.user_count() == 1) {
            traverse_hlo_and_collect(instr.users().front());
        }
    }

    return sliced_user_paths;
}

}  // namespace

absl::StatusOr<bool> NpuDynamicSliceFusionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
    using SlicedOperandPaths = absl::InlinedVector<absl::InlinedVector<HloInstruction*, 4>, 4>;
    using SlicedUserPaths = absl::InlinedVector<absl::InlinedVector<HloInstruction*, 4>, 4>;

    absl::flat_hash_map<HloInstruction*, std::pair<SlicedOperandPaths, SlicedUserPaths>> matches_kv;

    std::vector<HloInstruction*> matches;
    std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

    // Collect all potential NPU custom call matches
    for (HloComputation* computation : module->computations()) {
        if (computation->IsFusionComputation()) {
            continue;
        }
        for (HloInstruction* instr : computation->instructions()) {
            if (IsNpuCustomCall(instr)) {
                SlicedOperandPaths sliced_operand_paths = GetSlicedOperandPaths(*instr, *call_graph);
                bool has_sliced_operand_paths = sliced_operand_paths.size() > 1;
                SlicedUserPaths sliced_user_paths = GetSlicedUserPaths(*instr, *call_graph);
                bool has_sliced_user_paths = absl::c_any_of(
                    sliced_user_paths, [](auto& sliced_user_path) { return !sliced_user_path.empty(); });

                if (has_sliced_operand_paths || has_sliced_user_paths) {
                    matches_kv[instr] = std::make_pair(std::move(sliced_operand_paths),
                                                       std::move(sliced_user_paths));
                    matches.push_back(instr);
                }
            }
        }
    }

    if (matches.empty()) {
        return false;
    }

    bool changed = false;
    // Process each match
    for (HloInstruction* hero : matches) {
        auto& paths = matches_kv[hero];
        auto& [sliced_operand_paths, sliced_user_paths] = paths;
        absl::InlinedVector<HloInstruction*, 4> matched_instrs;

        // Flatten the operand paths into matched_instrs
        for (const auto& operand_path : sliced_operand_paths) {
            for (HloInstruction* instr : operand_path) {
                matched_instrs.push_back(instr);
            }
        }

        // Flatten the user paths into matched_instrs
        for (const auto& user_path : sliced_user_paths) {
            for (HloInstruction* instr : user_path) {
                matched_instrs.push_back(instr);
            }
        }

        auto captures = GetPatternCaptures(matched_instrs);

        TF_ASSIGN_OR_RETURN(HloComputation* fusion_body,
            CreateFusionBody(module, sliced_operand_paths, sliced_user_paths, captures));

        bool has_dynamic_slices = absl::c_any_of(matched_instrs, [](auto* instr) {
            return DynCast<HloDynamicIndexInstruction>(instr) != nullptr;
        });
        TF_ASSIGN_OR_RETURN(HloInstruction* fusion,
            CreateFusionInstruction(module, hero, captures, fusion_body, has_dynamic_slices));

        HloComputation* parent = hero->parent();
        if (fusion->shape().IsTuple()) {
            TF_RETURN_IF_ERROR(parent->ReplaceInstructionWithDifferentShape(hero, fusion));
            for (auto& sliced_user_path : sliced_user_paths) {
                auto* old_gte = DynCast<HloGetTupleElementInstruction>(sliced_user_path.front());
                if (old_gte != nullptr) {
                    HloInstruction* gte = parent->AddInstruction(
                        HloInstruction::CreateGetTupleElement(fusion, old_gte->tuple_index()));
                    TF_RETURN_IF_ERROR(parent->ReplaceInstruction(sliced_user_path.back(), gte));
                }
            }
        } else {
            HloInstruction* instr_to_be_replaced = hero;
            if (sliced_user_paths.empty()) {
                if (hero->shape().IsTuple()) {
                    if (hero->user_count() != 1 ||
                        !DynCast<HloGetTupleElementInstruction>(hero->users().front())) {
                        return absl::InternalError(
                            "Expect a single get-tuple-element user of the original "
                            "tuple-shaped hero op when address computation fusion does "
                            "not return a tuple");
                    }
                    instr_to_be_replaced = hero->users().front();
                }
            } else {
                instr_to_be_replaced = sliced_user_paths.front().back();
            }
            TF_RETURN_IF_ERROR(parent->ReplaceInstruction(instr_to_be_replaced, fusion));
            if (hero->parent()) {
                TF_RETURN_IF_ERROR(hero->parent()->RemoveInstruction(hero));
            }
        }
        changed = true;
    }

    return changed;
}

}  // namespace xla
