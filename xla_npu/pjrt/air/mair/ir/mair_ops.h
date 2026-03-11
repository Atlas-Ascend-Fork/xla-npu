/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAIR_OPS_H_
#define MAIR_OPS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

namespace mlir {
namespace air {

class AirDialect : public Dialect {
 public:
  explicit AirDialect(MLIRContext *context) : Dialect(getDialectNamespace(), context, TypeID::get<AirDialect>()) {
    initialize();
  }
  static StringRef getDialectNamespace() {
    return "air";
  }
  void initialize();
};

}  // end namespace air
}  // end namespace mlir

#define GET_OP_CLASSES
#include "xla/xla_npu/pjrt/air/mair/ir/mair_ops.h.inc"

#endif  //  MAIR_OPS_H_
