/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_H_
#define OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_H_

#include <cstring>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRoiAlign的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：ROIAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32，支持非连续的Tensor，数据格式支持NCHW。
 * @param [in] rois: npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT32，支持非连续的Tensor，数据格式支持ND。
 * @param [in] batchIndices: npu device侧的aclTensor，数据类型支持INT64，支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT32，和self一致，支持非连续的Tensor，数据格式支持NCHW。
 * @param [in] mode: host侧的string类型，池化模式，支持"avg"或"max"。
 * @param [in] outputHeight: host侧的int类型，ROI输出特征图的H。
 * @param [in] outputWidth: host侧的int类型，ROI输出特征图的W。
 * @param [in] samplingRatio: host侧的int类型，用于计算每个输出元素的和W上的bin数。
 * @param [in] spatialScale: host侧的float类型，缩放因子，用于将ROI坐标转换为输入特征图。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRoiAlignGetWorkspaceSize(const aclTensor* self, const aclTensor* rois,
                                                    const aclTensor* batchIndices, const char* mode, int outputHeight,
                                                    int outputWidth, int samplingRatio, float spatialScale,
                                                    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRoiAlign的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnRoiAlignGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRoiAlign(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_H_