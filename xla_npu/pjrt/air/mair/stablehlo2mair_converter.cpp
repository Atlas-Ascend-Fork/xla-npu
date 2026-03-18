/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "transforms/mair_passes.h"
#include "stablehlo2mair_converter.h"

namespace xla {
namespace mair {
int Stablehlo2MairConverter(mlir::ModuleOp& module) {
  mlir::PassManager pm(module.getContext());

  // 添加Stablehlo到Mair方言的转换Pass
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::air::createLegalizeStablehloToGePass());

  if (failed(pm.run(module))) return 4;

  return 0;
}
}  // namespace mair
}  // namespace xla
