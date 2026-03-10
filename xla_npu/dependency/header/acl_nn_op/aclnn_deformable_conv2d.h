/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/license/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_DEFORMABLE_CONV2D_H_
#define OP_API_INC_LEVEL2_ACLNN_DEFORMABLE_CONV2D_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDeformableConv2dGetWorkspaceSize
 * 算子功能：进行DeformableConv2d计算
 */
ACLNN_API aclnnStatus aclnnDeformableConv2dGetWorkspaceSize(const aclTensor* x, const aclTensor* weight,
                                                            const aclTensor* offset, const aclTensor* biasOptional,
                                                            const aclIntArray* kernelSize, const aclIntArray* stride,
                                                            const aclIntArray* padding, const aclIntArray* dilation,
                                                            int64_t groups, int64_t deformableGroups, bool modulated,
                                                            aclTensor* out, aclTensor* deformOutOptional,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor);

/*
 * @brief aclnnDeformableConv2d
 */
ACLNN_API aclnnStatus aclnnDeformableConv2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_DEFORMABLE_CONV2D_H_