/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRUtils.h - AFIR utility functions ---------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_UTILS_AFIRUTILS_H
#define MLIR_UTILS_AFIRUTILS_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace afir {
namespace utils {

/// Check if a type is a tensor type.
bool isTensorType(Type type);

/// Check if a type has a static shape.
bool hasStaticShape(Type type);

/// Get the shape of a tensor type. Returns empty vector for non-tensor types.
SmallVector<int64_t> getShape(Type type);

/// Get the element type of a shaped type. Returns the type itself for scalars.
Type getElementType(Type type);

/// Get the number of elements in a tensor. Returns -1 for dynamic shapes.
int64_t getNumElements(Type type);

}  // namespace utils
}  // namespace afir
}  // namespace mlir

#endif  // MLIR_UTILS_AFIRUTILS_H
