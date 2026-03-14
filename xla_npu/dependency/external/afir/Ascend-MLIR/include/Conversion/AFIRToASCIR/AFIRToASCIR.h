/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRToASCIR.h - AFIR to ASC-IR conversion --------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONVERSION_AFIRTOASCIR
#define AFIR_CONVERSION_AFIRTOASCIR

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace afir {

std::unique_ptr<Pass> createConvertAFIRToASCIRPass();

}  // namespace afir
}  // namespace mlir

#endif  // AFIR_CONVERSION_AFIRTOASCIR
