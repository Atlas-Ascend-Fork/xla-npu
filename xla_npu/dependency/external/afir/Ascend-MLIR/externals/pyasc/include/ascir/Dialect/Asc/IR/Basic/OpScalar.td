/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_SCALAR_TD
#define ASC_BASIC_OP_SCALAR_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_ScalarCastOp : APIOp<"scalar_cast", "ScalarCast", []> {
  let description = "Convert the data type of a scalar value.";
  let arguments = (ins
    AnyType:$valueIn,                 
    TypeAttr:$dtype,                  
    AscendC_RoundModeAttr:$roundMode  
  );
  let results = (outs AnyType:$valueOut);
  let assemblyFormat = "$valueIn attr-dict `:` type($valueIn) `->` type($valueOut)";
}

def AscendC_ScalarGetSFFValueOp : APIOp<"scalar_get_sff_value", "ScalarGetSFFValue", [AscFunc]> {
    let description = "Get the position of the first 0 or 1 (from LSB) in a uint64_t value. Returns -1 if not found.";
    let arguments = (ins 
        UI64:$valueIn, 
        AnyInteger:$countValue                  
    );
    let results = (outs AnySignlessIntegerOrIndex:$position); 
    let assemblyFormat = "operands attr-dict `:` qualified(type(operands)) `->` type($position)";
    let paramTypeLists = [0, 4]; 
}

#endif //ASC_BASIC_OP_SCALAR_TD
