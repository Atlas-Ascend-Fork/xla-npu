#include "xla/xla_npu/service/transform/npu_super_fusion.h"

#include <vector>

#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/status_macros.h"

namespace xla {

absl::StatusOr<bool> NpuSuperFusion::RunImpl(HloModule* module,
                                             const absl::flat_hash_set<absl::string_view>& execution_threads) {
    HloComputation* entry_computation = module->entry_computation();

    // If root is already a fusion, skip super fusion
    if (entry_computation->root_instruction()->opcode() == HloOpcode::kFusion) {
        return false;
    }

    // Collect all non-parameter instructions in topological order
    std::vector<HloInstruction*> to_fuse;
    for (HloInstruction* instr : entry_computation->MakeInstructionPostOrder()) {
        if (instr->opcode() != HloOpcode::kParameter) {
            to_fuse.push_back(instr);
        }
    }

    // Safety check: don't create overly large fusions that might cause issues
    if (to_fuse.size() > 50 || to_fuse.empty()) {
        LOG(INFO) << "Not creating super fusion: instruction count = " << to_fuse.size();
        return false;
    }

    LOG(INFO) << "Creating super fusion with " << to_fuse.size() << " instructions";

    // Start fusion from the root instruction (which should be the final computation)
    // This ensures we capture all computation logic but keep inputs separate
    HloInstruction* root_instr = entry_computation->root_instruction();

    // Create the fusion starting from the root (final result)
    HloInstruction* fusion_instr = entry_computation->AddInstruction(
        HloInstruction::CreateFusion(root_instr->shape(), HloInstruction::FusionKind::kLoop, root_instr));

    // Now fuse all the preceding instructions into this fusion
    // Since to_fuse is in post-order, we process them in reverse order
    // Skip the root instruction since it's already in the fusion
    for (auto it = to_fuse.rbegin(); it != to_fuse.rend(); ++it) {
        HloInstruction* instr = *it;

        // Skip if this is the root instruction we already started the fusion with
        if (instr != root_instr) {
            // Fuse this instruction into the fusion
            fusion_instr->FuseInstruction(instr);
        }
    }

    // If we created a fusion instruction and root is fuseable, make fusion the new root
    if (fusion_instr != nullptr) {
        entry_computation->set_root_instruction(fusion_instr);
    }

    // Note: We don't remove the old instructions since they create dependency issues.
    // DCE pass will clean up unreachable instructions if needed.

    // Clean up any remaining dead instructions
    TF_RETURN_IF_ERROR(HloDCE().Run(module, execution_threads).status());

    return true;
}

}  // namespace xla