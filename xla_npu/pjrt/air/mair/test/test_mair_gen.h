/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_MAIR_GEN_H_
#define TEST_MAIR_GEN_H_

#include <iostream>
#include <vector>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/CAPI/Registration.h"

#include "mlir-c/IR.h"

#include "../ir/mair_ops.h"
#include "../translations/export_graphdef.h"
#include "../translations/status.h"
#include "../translations/utils.h"

namespace mlir::air {

class GraphGenTester {
public:
    
    static Status Convert(mlir::ModuleOp& module, ge::GraphPtr &graphPtr);

    static void buildGeluModule(mlir::ModuleOp& module, mlir::MLIRContext& context);
    static void buildAddModule(mlir::ModuleOp& module, mlir::MLIRContext& context);
    static void buildAddMulModule(mlir::ModuleOp& module, mlir::MLIRContext& context);
    static void buildAEmptyModule(mlir::ModuleOp& module, mlir::MLIRContext& context);
    static void buildConstModule(mlir::ModuleOp& module, mlir::MLIRContext& context);

};

}

#endif  //TEST_MAIR_GEN_H_
