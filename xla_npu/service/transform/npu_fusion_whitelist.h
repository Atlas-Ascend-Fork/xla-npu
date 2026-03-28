/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_WHITELIST_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_WHITELIST_H_

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {

// NpuFusionWhitelist manages a whitelist of HLO opcodes that should skip fusion.
// Certain operations like Dot and Reduce are excluded from fusion to preserve
// their optimized implementations.
class NpuFusionWhitelist {
   public:
    // Returns true if given instruction should skip fusion.
    static bool ShouldSkipFusion(const HloInstruction* instr) {
        if (instr == nullptr) {
            return false;
        }
        return ShouldSkipFusion(instr->opcode());
    }

    // Returns true if given opcode should skip fusion.
    static bool ShouldSkipFusion(HloOpcode opcode) {
        // Hardcoded whitelist of opcodes to skip fusion
        // Note: Exp, Add, and Subtract are added to avoid numerical instability in fused kernels
        static const absl::flat_hash_set<HloOpcode> kSkipFusionOpcodes = {
            HloOpcode::kDot,
        };
        return kSkipFusionOpcodes.contains(opcode);
    }
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_WHITELIST_H_
