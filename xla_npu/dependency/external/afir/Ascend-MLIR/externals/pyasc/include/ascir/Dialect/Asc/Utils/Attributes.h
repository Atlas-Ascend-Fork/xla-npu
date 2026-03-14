/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASC_UTILS_ATTRIBUTES_H
#define ASCIR_DIALECT_ASC_UTILS_ATTRIBUTES_H
#define LITERAL constexpr const char *

namespace mlir {
namespace ascendc {

namespace attr {
LITERAL aicore = "ascendc.aicore";
LITERAL api = "ascendc.api";
LITERAL compile_mix = "asc.compile_mix";
LITERAL emitAsUnsigned = "ascendc.emit_as_unsigned";
LITERAL global = "ascendc.global";
LITERAL enable_debug = "asc.enable_debug";
LITERAL matmulCubeOnly = "asc.matmul_cube_only";
} // namespace attr

} // namespace ascendc
} // namespace mlir

#undef LITERAL
#endif // ASCIR_DIALECT_ASC_UTILS_ATTRIBUTES_H
