/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRTransformOps.h - AFIR transformation ops ----------------------===//
//
// Part of Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_TRANSFORMOPS_AFIRTRANSFORMOPS_H
#define MLIR_DIALECT_AFIR_TRANSFORMOPS_AFIRTRANSFORMOPS_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Dialect/AFIR/TransformOps/AFIRTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace afir {

void registerTransformDialectExtension(DialectRegistry &registry);

} // namespace afir
} // namespace mlir

#endif // MLIR_DIALECT_AFIR_TRANSFORMOPS_AFIRTRANSFORMOPS_H
