/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIROps.td - AFIR operation definitions -------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations for the AFIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_OPS
#define AFIR_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Basic Operations
//===----------------------------------------------------------------------===//

include "Dialect/AFIR/Ops/Basic/broadcast.td"
include "Dialect/AFIR/Ops/Basic/cast.td"
include "Dialect/AFIR/Ops/Basic/condition.td"
include "Dialect/AFIR/Ops/Basic/data.td"
include "Dialect/AFIR/Ops/Basic/math.td"

include "Dialect/AFIR/Ops/Adv/concat.td"
include "Dialect/AFIR/Ops/Adv/reduce.td"

#endif // AFIR_OPS