/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_UTILS_CONVERT_TENSOR_H_
#define AIR_UTILS_CONVERT_TENSOR_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"    // from @llvm-project

#include "xla/xla_npu/dependency/header/metadef/graph/types.h"
#include "xla/xla_npu/dependency/header/metadef/graph/tensor.h"
#include "./status.h"

using namespace ge;

namespace mlir::air {

// Converts an MLIR elements attribute to a ascend tensor.
Status ConvertToTensor(mlir::ElementsAttr attr, Tensor *output_tensor);

}  // namespace air

#endif  // AIR_UTILS_CONVERT_TENSOR_H_
