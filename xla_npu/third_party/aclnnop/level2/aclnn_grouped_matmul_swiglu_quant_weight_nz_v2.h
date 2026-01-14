/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_V2_H
#define OP_API_INC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_V2_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulSwigluQuantWeightNzV2 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: 表示公式中的x，数据类型支持INT8数据类型，数据格式支持ND。
 * @param [in] weight:
 * 表示公式中的weight，数据类型支持INT8数据类型，数据格式支持NZ。
 * @param [in] weightScale:
 * 表示量化参数，数据类型支持FLOAT32数据类型，数据格式支持ND。 
 * @param [in] weightAssistMatrix:
 * 表示weight辅助矩阵，数据类型支持FLOAT32数据类型。 
  * @param [in] weightScale:
 * 表示偏移，数据类型支持FLOAT32数据类型，数据格式支持ND。 
 * @param [in] xScale:
 * 表示perToken量化参数，数据类型支持FLOAT8_E8M0数据类型，数据格式支持ND。
 * @param [in] smoothScale:
 * 数据类型支持FLOAT32数据类型。 
 * @param [in] groupList: 必选参数，代表输入和输出分组轴上的索引情况，数据类型支持INT64。
 * @param [in] dequantMode: 表示反量化模式。
 * @param [in] dequantDtype: 表示反量化数据格式。
 * @param [in] quantMode: 表示量化模式。
 * @param [in] quantDtype: 表示量化数据格式。
 * @param [out] quantOutput: 表示公式中的out，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2数据类型，数据格式支持ND。
 * @param [out] quantScaleOutput: 表示公式中的outQuantScale，数据类型支持FLOAT8_E8M0数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(const aclTensor *x,
        const aclTensorList *weight, const aclTensorList *weightScale,
        [[maybe_unused]] const aclTensorList *weightAssistMatrix, [[maybe_unused]] const aclTensor *bias,
        [[maybe_unused]] const aclTensor *xScale, [[maybe_unused]] const aclTensor *smoothScale,
        const aclTensor *groupList,  [[maybe_unused]] int64_t dequantMode, 
        [[maybe_unused]] int64_t dequantDtype, [[maybe_unused]] int64_t quantMode,
        int64_t groupListType,
        const aclIntArray *tuningConfigOptional, 
        aclTensor *output, aclTensor *outputScale,
        uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulSwigluQuantV2的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif