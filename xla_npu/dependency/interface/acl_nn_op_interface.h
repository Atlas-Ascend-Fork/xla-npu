/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_OP_INTERFACE_H_
#define XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_OP_INTERFACE_H_

#include "xla/xla_npu/dependency/header/acl_nn/acl_meta.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"

#define DECLARE_ACL_FUNC_IN_OPAPI(func_name, args_type, args_name) \
  DECLARE_ACL_FUNC(libopapi, aclnnStatus, func_name, args_type, args_name)

// 所有算子执行接口的形参声明是完全相同的
#define DECLARE_ACL_EXEC_FUNC_IN_OPAPI(func_name)                        \
  DECLARE_ACL_FUNC_IN_OPAPI(                                             \
      func_name,                                                         \
      (void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, \
       const aclrtStream stream),                                        \
      (workspace, workspaceSize, executor, stream))

namespace xla {
namespace dep {

DECLARE_ACL_FUNC_IN_OPAPI(aclnnExpandGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* size,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, size, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnExpand)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnAddGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           const aclScalar* alpha, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, other, alpha, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnAdd)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnTanGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnTan)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnTanhGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnTanh)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMulGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMul)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnDivGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnDiv)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnSubGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           const aclScalar* alpha, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, other, alpha, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSub)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnNegGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnNeg)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnExpGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnExp)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMaximumGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMaximum)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMatmulGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* mat2,
                           aclTensor* out, int8_t cubeMathType,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, mat2, out, cubeMathType, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMatmul)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnCatGetWorkspaceSize,
                          (const aclTensorList* tensors, int64_t dim,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (tensors, dim, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnCat)

// Reduce sum operation
DECLARE_ACL_FUNC_IN_OPAPI(aclnnReduceSumGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dims,
                           bool keepDims, aclDataType dtype, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dims, keepDims, dtype, out, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnReduceSum)

// Copy operation for reshape
DECLARE_ACL_FUNC_IN_OPAPI(aclnnInplaceCopyGetWorkspaceSize,
                          (aclTensor * selfRef, const aclTensor* src,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (selfRef, src, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnInplaceCopy)

// MaxDim operation for reduce max
DECLARE_ACL_FUNC_IN_OPAPI(aclnnMaxDimGetWorkspaceSize,
                          (const aclTensor* self, int64_t dim, bool keepdim,
                           aclTensor* out, aclTensor* indices,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, keepdim, out, indices, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMaxDim)

// Amax operation for reduce max (returns max values only, no indices)
DECLARE_ACL_FUNC_IN_OPAPI(aclnnAmaxGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dim,
                           bool keepDim, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, keepDim, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnAmax)

// ==================== 新增算子接口声明 ====================

// Rsqrt operation: 1/sqrt(x)
// Reference: acl_nn_op/aclnn_rsqrt.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnRsqrtGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnRsqrt)

// Sin operation: sin(x)
// Reference: acl_nn_op/aclnn_sin.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnSinGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSin)

// Cos operation: cos(x)
// Reference: acl_nn_op/aclnn_cos.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnCosGetWorkspaceSize,
                          (const aclTensor* input, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (input, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnCos)

// Transpose/Permute operation
// Reference: acl_nn_op/aclnn_permute.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnPermuteGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dims,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, dims, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnPermute)

// Slice operation (V2 version with array parameters)
// Reference: acl_nn_op/aclnn_slice_v2.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnSliceV2GetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* starts,
                           const aclIntArray* ends, const aclIntArray* axes,
                           const aclIntArray* steps, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, starts, ends, axes, steps, out, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSliceV2)

// Gather operation
// Reference: acl_nn_op/aclnn_gather.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnGatherGetWorkspaceSize,
                          (const aclTensor* self, int64_t dim,
                           const aclTensor* index, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, index, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnGather)

// Compare operations (Tensor-Tensor comparison)
// Reference: acl_nn_op/aclnn_eq_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnEqTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnEqTensor)

// Reference: acl_nn_op/aclnn_ne_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnNeTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnNeTensor)

// Reference: acl_nn_op/aclnn_lt_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnLtTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnLtTensor)

// Reference: acl_nn_op/aclnn_le_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnLeTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnLeTensor)

// Reference: acl_nn_op/aclnn_gt_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnGtTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnGtTensor)

// Reference: acl_nn_op/aclnn_ge_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnGeTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnGeTensor)

// SWhere (Select/Where) operation: select elements from self or other based on condition
// Reference: acl_nn_op/aclnn_s_where.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnSWhereGetWorkspaceSize,
                          (const aclTensor* condition, const aclTensor* self,
                           const aclTensor* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (condition, self, other, out, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSWhere)

// BitwiseAnd/LogicalAnd operation: element-wise logical/bitwise AND
// Reference: acl_nn_op/aclnn_bitwise_and_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnBitwiseAndTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnBitwiseAndTensor)

// IndexSelect operation: gather elements along specified dimension
// Reference: acl_nn_op/aclnn_index_select.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnIndexSelectGetWorkspaceSize,
                          (const aclTensor* self, int64_t dim,
                           const aclTensor* index, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, index, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnIndexSelect)

// GatherV2 operation: gather elements along specified dimension
// Reference: acl_nn_op/aclnn_gather_v2.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnGatherV2GetWorkspaceSize,
                          (const aclTensor* self, int64_t dim,
                           const aclTensor* index, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, index, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnGatherV2)

}  // namespace dep
}  // namespace xla

#endif  // XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_OP_INTERFACE_H_
