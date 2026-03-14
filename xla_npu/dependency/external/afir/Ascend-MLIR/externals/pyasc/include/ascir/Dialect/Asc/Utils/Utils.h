/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASC_UTILS_UTILS_H
#define ASCIR_DIALECT_ASC_UTILS_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace ascendc {

template <typename OpT>
struct HoistOpPattern : public OpRewritePattern<OpT> {
    using OpRewritePattern<OpT>::OpRewritePattern;

    virtual bool hoistable(OpT) const
    {
        return true;
    }

    LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const override
    {
        Operation *parent = op->getParentOp();
        if (isa<func::FuncOp>(parent))
            return failure();
        if (!hoistable(op))
            return failure();
        DominanceInfo di;
        bool dominatedByOperands =
            llvm::all_of(op->getOperands(), [&](Value opnd) { return di.dominates(opnd, parent); });
        if (!dominatedByOperands)
            return failure();
        rewriter.setInsertionPoint(parent);
        rewriter.replaceOp(op, rewriter.clone(*op.getOperation())->getResults());
        return success();
    }
};

bool opPrecedes(Operation *lhs, Operation *rhs);

bool opPrecedes(Operation *lhs, Operation *rhs, DominanceInfo &di);

void registerInlinerInterfaces(DialectRegistry &registry);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_DIALECT_ASC_UTILS_UTILS_H
