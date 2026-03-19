/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OP_API_INC_SWI_GLU_QUANT_V2_H_
#define OP_API_INC_SWI_GLU_QUANT_V2_H_

 #include "aclnn_base.h"
 #include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSwiGluQuantV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：在SwiGlu激活函数后添加quant操作，实现输入x的SwiGluQuant计算
 * 动态量化计算公式：  
 * Activation = SwiGLU(x) = Swish(A)*B \\
 * Y_{tmp}^i = Activation[i:group\_index[i], :] *  smooth\_scales[i:group\_index[i], :], i \in [0, G) \cap \mathbb{Z}\\
 * scale=row\_max(abs(Y_{tmp}))/127
 * Y = Cast(Mul(Y_{tmp}, Scale))
 * 其中，A表示输入x的前半部分，B表示输入x的后半部分, G为group_index的分组数量。
 * 静态量化计算公式：  
 * Y=AscendQuant(SwiGLU(x), smooth\_scales, offsets)
 * 其中，A表示输入x的前半部分，B表示输入x的后半部分。

 * @domain aclnnop_ops_infer
 * 参数描述：
 * @param [in]   x
 * 输入aclTensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。数据格式支持ND。支持非连续Tensor。最后一维需要为2的倍数，且x的维数必须大于1维，当前仅支持输入x的最后一维长度不超过8192。
 * @param [in]   smoothScalesOptional
 * 输入aclTensor，数据类型支持FLOAT，threshold计算的边界值。数据格式支持ND。支持非连续Tensor。shape支持[G, N]，[G, ]，其中G代表groupIndex分组数量，N为计算输入x的最后一维大小的二分之一。
 * @param [in]   offsetsOptional
 * 输出aclTensor，该参数在动态量化场景下不生效，用户传入空指针即可。静态量化场景下：数据类型支持FLOAT，per_channel模式下shape支持[G, N]，per_tensor模式下shape支持[G, ]，且数据类型和shape需要与smoothScalesOptional保持一致。支持非连续的Tensor，数据格式支持ND。
 * @param [in]   groupIndexOptional
 * 数据类型支持INT32，MoE分组需要的group_index，对应公式group_index变量，Device侧的aclTensor。数据格式支持ND。支持非连续Tensor。shape支持[G, ]。
 * @param [in]   activateLeft
 * 数据类型支持bool，表示左矩阵是否参与运算，用户必须传参。
 * @param [in]   quantModeOptional
 * 数据类型支持String，表示左矩阵是否参与运算，用户必须传参。用户必须传参，"static"表示静态量化、"dynamic"表示动态量化、"dynamic_msd"表示动态MSD量化。当前仅支持"dynamic"动态量化, "static"静态量化。静态量化仅支持per_tensor量化和per_channel量化。
 * @param [in]   groupListTypeOptional
 * 数据类型支持int64_t，用户必须传参，0表示cumsum模式、1表示count模式。当前仅支持0 cumsum模式, 1 count模式。
 * @param [in]   dstType
 * 数据类型支持int64_t，用户必须传参，2表示yOut为int8量化输出、29表示yOut为int4量化输出。当前仅支持输入2和29，默认值是2。
 * @param [in]   yOut
 * 数据类型支持INT8，数据格式支持ND。支持非连续Tensor。计算输出yOut的shape最后一维大小为计算输入x最后一维的二分之一，其余维度与x保持一致。
 * @param [in]   scaleOut
 * 数据类型支持FLOAT，数据格式支持ND。支持非连续Tensor。计算输出scaleOut的shape与计算输入x相比，无最后一维，其余维度与计算输入x保持一致。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */

ACLNN_API aclnnStatus aclnnSwiGluQuantV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *smoothScalesOptional,
    const aclTensor *offsetsOptional,
    const aclTensor *groupIndexOptional,
    bool activateLeft,
    char *quantModeOptional,
    int64_t groupListTypeOptional,
    int64_t dstType,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnSwiGluQuantV2的第二段接口，用于执行计算。
 * 功能描述：在SwiGlu激活函数后添加quant操作，实现输入x的SwiGluQuant计算
 * 动态量化计算公式：  
 * Activation = SwiGLU(x) = Swish(A)*B \\
 * Y_{tmp}^i = Activation[i:group\_index[i], :] *  smooth\_scales[i:group\_index[i], :], i \in [0, G) \cap \mathbb{Z}\\
 * scale=row\_max(abs(Y_{tmp}))/127
 * Y = Cast(Mul(Y_{tmp}, Scale))
 * 其中，A表示输入x的前半部分，B表示输入x的后半部分, G为group_index的分组数量。
 * 静态量化计算公式：  
 * Y=AscendQuant(SwiGLU(x), smooth\_scales, offsets)
 * 其中，A表示输入x的前半部分，B表示输入x的后半部分。

 * @domain aclnnop_ops_infer
 * 参数描述：
 * param [in] workspace: 在npu device侧申请的workspace内存起址。
 * param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSwiGluQuantV2GetWorkspaceSize获取。
 * param [in] stream: acl stream流。
 * param [in] executor: op执行器，包含了算子计算流程。
 * return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus aclnnSwiGluQuantV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_SWI_GLU_QUANT_V2_H_