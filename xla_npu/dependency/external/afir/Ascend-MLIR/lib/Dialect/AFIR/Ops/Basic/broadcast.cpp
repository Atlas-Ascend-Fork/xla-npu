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
