/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_BLOCKSYNC_TD
#define ASC_BASIC_OP_BLOCKSYNC_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Data Sync Barrier Op
//===----------------------------------------------------------------------===//

def AscendC_DataSyncBarrierOp
    : APIOp<"data_sync_barrier", "DataSyncBarrier", [AscFunc]> {
  let description = "Synchronize data accesses by waiting for specified memory operations, "
                    "controlled by the MemDsbT parameter.";
  let arguments = (ins
    AscendC_MemDsbAttr:$memDsbType
  );
  let paramTypeLists = [3];
}

//===----------------------------------------------------------------------===//
// Synchronization operations
//===----------------------------------------------------------------------===//

def PipeBarrierOp : APIOp<"pipe_barrier", "PipeBarrier"> {
  let arguments = (ins AscendC_PipeAttr:$pipe);
  let assemblyFormat = "$pipe attr-dict";
  let hasCanonicalizeMethod = 1;
}

def SetFlagOp : APIOp<"set_flag", "SetFlag", [AscFunc]> {
  let arguments = (ins AscendC_HardEventAttr:$event, AnyType:$eventId);
  let assemblyFormat = "$event `,` $eventId attr-dict `:` type($eventId)";
  let paramTypeLists = [3, 0];
}

def WaitFlagOp : APIOp<"wait_flag", "WaitFlag"> {
  let arguments = (ins AscendC_HardEventAttr:$event, AnyType:$eventId);
  let assemblyFormat = "$event `,` $eventId attr-dict `:` type($eventId)";
}
#endif // ASC_BASIC_OP_BLOCKSYNC_TD
