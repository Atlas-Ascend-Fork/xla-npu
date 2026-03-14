/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_VEC_CMPSEL_TD
#define ASC_BASIC_OP_VEC_CMPSEL_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_CompareL0Op : VectorOp<"compare_l0", "Compare", [OpWithDstInterface, AscFunc]> {
    let description = "`AscendC::Compare` is a vector binary operation (L0 API).\n";
    let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1, AscendC_CMPMODEAttr:$cmpMode,
                   AnyType:$mask, AnyType:$repeatTimes, 
                   AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
    let paramTypeLists = [0, 0, 0, -1, 0, 0, 0];
}

def AscendC_CompareL1Op : VectorOp<"compare_l1", "Compare", [OpWithDstInterface]> {
    let description = "`AscendC::Compare` is a vector binary operation (L1 API).\n";
    let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1, AscendC_CMPMODEAttr:$cmpMode,
                   Variadic<UI64>:$mask, AnyType:$repeatTimes, 
                   AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
}

def AscendC_CompareL2Op : VectorOp<"compare_l2", "Compare", [OpWithDstInterface, AscFunc]> {
    let description = "`AscendC::Compare` is a vector binary operation (L2 API).\n";
    let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1, AscendC_CMPMODEAttr:$cmpMode,
                   AnyType:$calCount);
    let paramTypeLists = [0, 0, 0, -1, 0];
}

def AscendC_CompareRL0Op : VectorOp<"compare_r_l0", "Compare", [AscFunc]> {
    let description = "`AscendC::Compare` is a vector binary operation (store result in the register) (L0 API).\n";
    let arguments = (ins AnyType:$src0, AnyType:$src1, AscendC_CMPMODEAttr:$cmpMode,
                   AnyType:$mask, AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
    let paramTypeLists = [0, 0, -1, 0, 0];
}

def AscendC_CompareRL1Op : VectorOp<"compare_r_l1", "Compare"> {
    let description = "`AscendC::Compare` is a vector binary operation (store result in the register) (L1 API).\n";
    let arguments = (ins AnyType:$src0, AnyType:$src1, AscendC_CMPMODEAttr:$cmpMode,
                   Variadic<UI64>:$mask, AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
}

def AscendC_CompareScalarL0Op : VectorOp<"compare_scalar_l0", "CompareScalar", [OpWithDstInterface, AscFunc]> {
    let description = "`AscendC::CompareScalar` is a vector-scalar binary operation (L0 API).\n";
    let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1Scalar, AscendC_CMPMODEAttr:$cmpMode,
                   AnyType:$mask, AnyType:$repeatTimes, 
                   AscendC_UnaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
    let paramTypeLists = [0, 0, 0, -1, 0, 0, 0];
}

def AscendC_CompareScalarL1Op : VectorOp<"compare_scalar_l1", "CompareScalar", [OpWithDstInterface]> {
    let description = "`AscendC::CompareScalar` is a vector-scalar binary operation (L1 API).\n";
    let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1Scalar, AscendC_CMPMODEAttr:$cmpMode,
                   Variadic<UI64>:$mask, AnyType:$repeatTimes, 
                   AscendC_UnaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
}

def AscendC_CompareScalarL2Op : VectorOp<"compare_scalar_l2", "CompareScalar", [OpWithDstInterface, AscFunc]> {
    let description = "`AscendC::CompareScalar` is a vector-scalar binary operation (L2 API).\n";
    let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1Scalar, AscendC_CMPMODEAttr:$cmpMode,
                   AnyType:$calCount);
    let paramTypeLists = [0, 0, 0, -1, 0];
}

def AscendC_GetCmpMaskOp : VectorOp<"get_cmp_mask", "GetCmpMask", [OpWithDstInterface, AscFunc]> {
    let description = "get Compare(store results in register) results";
    let arguments = (ins AscendC_LocalTensor:$dst);
}

def AscendC_SelectL0Op : VectorOp<"select_l0", "Select", [OpWithDstInterface]> {
    let description = "Select elements from two source tensors (L0 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$selMask,
                         AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                         AscendC_SelModeAttr:$mode, AnyType:$mask,
                         AnyType:$repeatTimes, AnyType:$dstBlkStride,
                         AnyType:$src0BlkStride, AnyType:$src1BlkStride,
                         AnyType:$dstRepStride, AnyType:$src0RepStride,
                         AnyType:$src1RepStride);
}

def AscendC_SelectL2Op : VectorOp<"select_l2", "Select", [OpWithDstInterface]> {
    let description = "Select elements from two source tensors (L2 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$selMask,
                         AscendC_LocalTensor:$src0, AnyType:$src1,
                         AscendC_SelModeAttr:$selMode, AnyType:$calCount);
}

def AscendC_SelectScalarL0Op : VectorOp<"select_scalar_l0", "Select", [OpWithDstInterface]> {
    let description = "Select elements from source tensor and a scalar (L0 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$selMask,
                         AscendC_LocalTensor:$src0, AnyType:$src1,
                         AscendC_SelModeAttr:$mode, AnyType:$mask,
                         AnyType:$repeatTimes, AnyType:$dstBlkStride,
                         AnyType:$src0BlkStride, AnyType:$src1BlkStride,
                         AnyType:$dstRepStride, AnyType:$src0RepStride,
                         AnyType:$src1RepStride);
}

def AscendC_SelectScalarL2Op : VectorOp<"select_scalar_l2", "Select", [OpWithDstInterface]> {
    let description = "Select elements from source tensor and a scalar (L2 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$selMask,
                         AscendC_LocalTensor:$src0, AnyType:$src1,
                         AscendC_SelModeAttr:$selMode, AnyType:$calCount);
}

def AscendC_SetCmpMaskOp : VectorOp<"set_cmp_mask", "SetCmpMask", [AscFunc]> {
    let description = "set register for Select(without mask)";
    let arguments = (ins AscendC_LocalTensor:$src);
}
#endif //ASC_BASIC_OP_VEC_CMPSEL_TD
