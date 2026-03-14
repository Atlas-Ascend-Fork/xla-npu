/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_BASIC_VECTOR_SELECT_INSTR_H
#define ASCIR_TARGET_ASC_BASIC_VECTOR_SELECT_INSTR_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Select operations
//===----------------------------------------------------------------------===//

template <typename SelectL2Op>
LogicalResultForT<SelectL2Op, ascendc::SelectL2Op, ascendc::SelectScalarL2Op> printOperation(CodeEmitter &emitter,
                                                                                             SelectL2Op op)
{
    auto &os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSelMask()) << ", " << emitter.getOrCreateName(op.getSrc0()) << ", "
       << emitter.getOrCreateName(op.getSrc1())
       << ", AscendC::SELMODE::" << ascendc::stringifySelMode(op.getSelMode()).upper() << ", "
       << emitter.getOrCreateName(op.getCalCount()) << ")";
    return success();
}

template <typename SelectL0Op>
LogicalResultForT<SelectL0Op, ascendc::SelectL0Op, ascendc::SelectScalarL0Op> printOperation(CodeEmitter &emitter,
                                                                                             SelectL0Op op)
{
    auto &os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSelMask()) << ", " << emitter.getOrCreateName(op.getSrc0()) << ", "
       << emitter.getOrCreateName(op.getSrc1()) << ", "
       << "AscendC::SELMODE::" << ascendc::stringifySelMode(op.getMode()).upper() << ", "
       << emitter.getOrCreateName(op.getMask()) << ", " << emitter.getOrCreateName(op.getRepeatTimes())
       << ", AscendC::BinaryRepeatParams(" << emitter.getOrCreateName(op.getDstBlkStride()) << ", "
       << emitter.getOrCreateName(op.getSrc0BlkStride()) << ", " << emitter.getOrCreateName(op.getSrc1BlkStride())
       << ", " << emitter.getOrCreateName(op.getDstRepStride()) << ", "
       << emitter.getOrCreateName(op.getSrc0RepStride()) << ", " << emitter.getOrCreateName(op.getSrc1RepStride())
       << "))";
    return success();
}

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CompareL1Op op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CompareRL1Op op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CompareScalarL1Op op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_BASIC_VECTOR_SELECT_INSTR_H
