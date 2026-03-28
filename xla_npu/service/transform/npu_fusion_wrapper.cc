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

#include "xla/xla_npu/service/transform/npu_fusion_wrapper.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

bool NpuFusionWrapper::MustWrapInstruction(HloOpcode opcode) {
    // For NPU, we wrap most element-wise and data movement operations
    // to ensure they can be properly emitted by the NPU backend.
    switch (opcode) {
        case HloOpcode::kXor:
        case HloOpcode::kTranspose:
        case HloOpcode::kTanh:
        case HloOpcode::kTan:
        case HloOpcode::kStochasticConvert:
        case HloOpcode::kSubtract:
        case HloOpcode::kSqrt:
        case HloOpcode::kSlice:
        case HloOpcode::kSinh:
        case HloOpcode::kSin:
        case HloOpcode::kSign:
        case HloOpcode::kShiftRightArithmetic:
        case HloOpcode::kShiftRightLogical:
        case HloOpcode::kShiftLeft:
        case HloOpcode::kSelect:
        case HloOpcode::kScatter:
        case HloOpcode::kRsqrt:
        case HloOpcode::kRoundNearestEven:
        case HloOpcode::kRoundNearestAfz:
        case HloOpcode::kReverse:
        case HloOpcode::kRemainder:
        case HloOpcode::kReduceWindow:
        case HloOpcode::kReducePrecision:
        case HloOpcode::kReduce:
        case HloOpcode::kReshape:
        case HloOpcode::kReal:
        case HloOpcode::kPower:
        case HloOpcode::kPopulationCount:
        case HloOpcode::kPad:
        case HloOpcode::kOr:
        case HloOpcode::kNot:
        case HloOpcode::kNegate:
        case HloOpcode::kMultiply:
        case HloOpcode::kMinimum:
        case HloOpcode::kMaximum:
        case HloOpcode::kMap:
        case HloOpcode::kLog1p:
        case HloOpcode::kLog:
        case HloOpcode::kIsFinite:
        case HloOpcode::kIota:
        case HloOpcode::kImag:
        case HloOpcode::kGather:
        case HloOpcode::kFloor:
        case HloOpcode::kExpm1:
        case HloOpcode::kExp:
        case HloOpcode::kErf:
        case HloOpcode::kDynamicUpdateSlice:
        case HloOpcode::kDynamicSlice:
        case HloOpcode::kDot:
        case HloOpcode::kDivide:
        case HloOpcode::kCosh:
        case HloOpcode::kCos:
        case HloOpcode::kConvert:
        case HloOpcode::kConvolution:
        case HloOpcode::kConcatenate:
        case HloOpcode::kComplex:
        case HloOpcode::kCompare:
        case HloOpcode::kClz:
        case HloOpcode::kClamp:
        case HloOpcode::kCbrt:
        case HloOpcode::kCeil:
        case HloOpcode::kBroadcast:
        case HloOpcode::kBitcastConvert:
        case HloOpcode::kAtanh:
        case HloOpcode::kAtan2:
        case HloOpcode::kAnd:
        case HloOpcode::kAdd:
        case HloOpcode::kAsinh:
        case HloOpcode::kAsin:
        case HloOpcode::kAcosh:
        case HloOpcode::kAcos:
        case HloOpcode::kAbs:
            return true;
        default:
            NPU_LOG_FORMAT_PRINT(DEBUG, "NpuFusionWrapper: not wrapping opcode %s",
                                 HloOpcodeString(opcode));
            return false;
    }
}

HloInstruction::FusionKind NpuFusionWrapper::ChooseFusionKind(
    const HloInstruction& producer, const HloInstruction& consumer) {
    // For NPU, we use kLoop fusion by default for all operations
    // This is a simplified approach compared to GPU's more complex selection
    return HloInstruction::FusionKind::kLoop;
}

}  // namespace xla
