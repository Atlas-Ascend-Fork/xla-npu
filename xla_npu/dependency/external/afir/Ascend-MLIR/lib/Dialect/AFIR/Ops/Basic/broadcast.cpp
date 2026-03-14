/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- broadcast.cpp - AFIR broadcast operation implementations -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

namespace {

static LogicalResult verifyUnaryOp(Operation *op) {
  if (op->getNumOperands() != 1) {
    return op->emitOpError("expected 1 operand");
  }
  if (op->getNumResults() != 1) {
    return op->emitOpError("expected 1 result");
  }

  auto inputType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!inputType || !resultType) {
    return op->emitOpError("expected tensor operand and result");
  }

  if (inputType.hasRank() && resultType.hasRank()) {
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    size_t inputRank = inputShape.size();
    size_t resultRank = resultShape.size();

    if (resultRank < inputRank) {
      return op->emitOpError("result rank must be >= input rank for broadcast");
    }

    size_t offset = resultRank - inputRank;
    for (size_t i = 0; i < inputRank; ++i) {
      int64_t inputDim = inputShape[i];
      int64_t resultDim = resultShape[i + offset];

      if (inputDim != ShapedType::kDynamic && resultDim != ShapedType::kDynamic && inputDim != 1 &&
          inputDim != resultDim) {
        return op->emitOpError("input shape must be broadcastable to result shape");
      }
    }
  }

  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Broadcast Operations
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::verify() {
  return verifyUnaryOp(getOperation());
}
