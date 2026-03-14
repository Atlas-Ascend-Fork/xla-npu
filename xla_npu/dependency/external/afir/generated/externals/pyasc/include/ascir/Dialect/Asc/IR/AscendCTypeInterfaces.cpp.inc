/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Returns a clone of this type with the given shape and element type.
/// 
/// If no shape is provided, the shape of this type is used. In that case, if
/// this type is unranked, so is the resulting type.
/// 
/// If a shape is provided, the resulting type is always ranked, even if this
/// type is unranked.
::mlir::ShapedType mlir::ascendc::BaseTensorType::cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const {
      return getImpl()->implShapedType->cloneWith(getImpl()->implShapedType, *this, shape, elementType);
  }

/// Returns the element type of this shaped type.
::mlir::Type mlir::ascendc::BaseTensorType::getElementType() const {
      return getImpl()->implShapedType->getElementType(getImpl()->implShapedType, *this);
  }

/// Returns if this type is ranked, i.e. it has a known number of dimensions.
bool mlir::ascendc::BaseTensorType::hasRank() const {
      return getImpl()->implShapedType->hasRank(getImpl()->implShapedType, *this);
  }

/// Returns the shape of this type if it is ranked, otherwise asserts.
::llvm::ArrayRef<int64_t> mlir::ascendc::BaseTensorType::getShape() const {
      return getImpl()->implShapedType->getShape(getImpl()->implShapedType, *this);
  }
