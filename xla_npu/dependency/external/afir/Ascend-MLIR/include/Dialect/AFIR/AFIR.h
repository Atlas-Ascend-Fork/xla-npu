/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIR.h - AFIR dialect declaration -----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_AFIR_H
#define MLIR_DIALECT_AFIR_AFIR_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

// Include dialect definition
#include "Dialect/AFIR/Dialect.h.inc"

// Include enums (must be before attributes)
#include "Dialect/AFIR/Enums.h.inc"

// Include attribute definitions
#define GET_ATTRDEF_CLASSES
#include "Dialect/AFIR/Attrs.h.inc"

// Include operation definitions
#define GET_OP_CLASSES
#include "Dialect/AFIR/Ops.h.inc"

#endif  // MLIR_DIALECT_AFIR_AFIR_H