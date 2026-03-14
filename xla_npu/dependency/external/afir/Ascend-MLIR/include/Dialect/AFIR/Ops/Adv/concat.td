/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- concat.td - AFIR concat operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the concat operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONCAT_OPS
#define AFIR_CONCAT_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def AFIR_ConcatOp : AFIR_VariadicOp<"concat", [Pure, InferTensorTypeAdaptor], [I8, UI8, I16, UI16, I32, UI32, I64, F16, F32], "Concat", "Vector", 1> {
  let arguments = !con(
    (ins 
      Variadic<TensorOf<[I8, UI8, I16, UI16, I32, UI32, I64, F16, F32]>>:$inputs,
      I32Attr:$concat_axis
    ),
    commonArgs
  );
  let extraClassDeclaration = [{
    /// Returns when two result types are compatible for this op; method used by
    /// InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
  let extraClassDefinition = [{
    bool $cppClass::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      if (l.size() != r.size() || l.size() != 1) return false;
      return succeeded(verifyCompatibleShape(l[0], r[0]));
    }
  }];
}

#endif // AFIR_CONCAT_OPS