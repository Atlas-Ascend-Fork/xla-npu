/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPUIR_BACKEND_H_
#define NPUIR_BACKEND_H_

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/xla_npu/dependency/external/npuir/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusion.h"

namespace mlir::hfusion {

class GraphGenTester {
 public:
  static void test();
};

}  // namespace mlir::hfusion

#endif NPUIR_BACKEND_H_
