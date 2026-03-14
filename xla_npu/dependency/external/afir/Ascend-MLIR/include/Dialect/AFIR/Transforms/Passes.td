/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Passes.td - AFIR dialect passes ---------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TRANSFORMS_PASSES
#define AFIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AFIRCanonicalizePass : Pass<"afir-canonicalize", "mlir::ModuleOp"> {
  let summary = "Canonicalize AFIR operations";
  let description = [{
    This pass performs canonicalization transformations on AFIR operations,
    including constant folding and algebraic simplifications.
  }];
  let constructor = "mlir::afir::createAFIRCanonicalizePass()";
  let dependentDialects = ["mlir::afir::AFIRDialect"];
}

def AFIRShapeInferencePass : Pass<"afir-shape-inference", "mlir::ModuleOp"> {
  let summary = "Infer shapes for AFIR operations";
  let description = [{
    This pass infers and propagates shape information through AFIR operations.
  }];
  let constructor = "mlir::afir::createAFIRShapeInferencePass()";
  let dependentDialects = ["mlir::afir::AFIRDialect"];
}

def AFIRAddAxisPass : Pass<"afir-add-axis", "mlir::ModuleOp"> {
  let summary = "Add Axis for AFIR operations";
  let description = [{
    This pass add axis fpr AFIR operations.
  }];
  let constructor = "mlir::afir::createAFIRAddAxisPass()";
  let dependentDialects = ["mlir::afir::AFIRDialect"];
}


#endif // AFIR_TRANSFORMS_PASSES
