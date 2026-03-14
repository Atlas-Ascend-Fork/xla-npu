/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Passes.h - AFIR dialect passes ---------------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace afir {

// Canonicalization pass
std::unique_ptr<Pass> createAFIRCanonicalizePass();

// Shape inference pass
std::unique_ptr<Pass> createAFIRShapeInferencePass();

std::unique_ptr<Pass> createAFIRAddAxisPass();

// Generate pass registration declarations
#define GEN_PASS_REGISTRATION
#include "Dialect/AFIR/Transforms/Passes.h.inc"

}  // namespace afir
}  // namespace mlir

#endif  // MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H
