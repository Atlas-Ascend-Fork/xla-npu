/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRCanonicalize.cpp - AFIR canonicalization pass --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace afir {

#define GEN_PASS_DEF_AFIRCANONICALIZEPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

// Pattern: add(x, 0) -> x (when 0 is a constant)
// Pattern: mul(x, 1) -> x (when 1 is a constant)
// These patterns can be expanded based on needs

struct AFIRCanonicalizePass : public impl::AFIRCanonicalizePassBase<AFIRCanonicalizePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());

    // Add canonicalization patterns here
    // patterns.add<...>(patterns.getContext());

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRCanonicalizePass() {
  return std::make_unique<AFIRCanonicalizePass>();
}

}  // namespace afir
}  // namespace mlir
