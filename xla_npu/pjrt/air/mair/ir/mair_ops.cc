/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mair_ops.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_OP_CLASSES
#include "xla/xla_npu/pjrt/air/mair/ir/mair_ops.cc.inc"

namespace mlir {
namespace air {

void AirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/xla_npu/pjrt/air/mair/ir/mair_ops.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

void GraphOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {}

}  // namespace air
}  // namespace mlir
