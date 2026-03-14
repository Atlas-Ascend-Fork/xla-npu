/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_PROPOSAL_TD
#define ASC_BASIC_OP_PROPOSAL_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_ProposalConcatOp : VectorOp<"proposal_concat", "ProposalConcat", [AscFunc]> {
  let description = "Merge continuous elements into the corresponding positions within the Region Proposal";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AnyType:$repeatTime,
    AnyType:$modeNumber
  );
}

def AscendC_ProposalExtractOp : VectorOp<"proposal_extract", "ProposalExtract", [AscFunc]> {
  let description = "Extract the individual elements at the corresponding positions from the Region Proposals and then rearrange them.";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AnyType:$repeatTime,
    AnyType:$modeNumber
  );
}

#endif //ASC_BASIC_OP_PROPOSAL_TD
