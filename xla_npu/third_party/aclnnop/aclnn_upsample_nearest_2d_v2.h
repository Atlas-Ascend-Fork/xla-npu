/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_UNAMPLE_NEAREST_2D_V2_H_
#define OP_API_INC_UNAMPLE_NEAREST_2D_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleNearest2dV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE、UIN8。
 * 数据格式支持NCHW、NHWC。支持非连续的Tensor。
 * @param [in] outputSize:  npu device侧的aclIntArray，指定输出空间大小。
 * @param [in] scalesH: float常量，表示输出out的L维度乘数。
 * @param [in] scalesW: float常量，表示输出out的W维度乘数。
 * @param [out] out: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE、UIN8，
 * 且数据类型与self的数据类型一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest2dV2GetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize,
    float scalesH, float scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnUpsampleNearest2dV2的第二段接口，用于执行计算。
 *
 * 算子功能：对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnUpsampleNearest2dV2GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的AscendCL Stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest2dV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_NEAREST_2D_V2_H_
