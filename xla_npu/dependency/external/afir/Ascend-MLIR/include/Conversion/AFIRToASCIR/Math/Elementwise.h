/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Elementwise.h - AFIR Elementwise ops to ASC-IR conversion -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file declares the conversion patterns for AFIR elementwise operations
// to ASC-IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_AFIRTOASCIR_MATH_ELEMENTWISE_H
#define MLIR_CONVERSION_AFIRTOASCIR_MATH_ELEMENTWISE_H

namespace mlir {

class MLIRContext;
class RewritePatternSet;
class TypeConverter;

namespace afir {

void populateLoweringAFIRElementwiseOpToASCIRPattern(RewritePatternSet &patterns, MLIRContext *ctx,
                                                     TypeConverter &typeConverter);

}  // namespace afir
}  // namespace mlir

#endif  // MLIR_CONVERSION_AFIRTOASCIR_MATH_ELEMENTWISE_H
