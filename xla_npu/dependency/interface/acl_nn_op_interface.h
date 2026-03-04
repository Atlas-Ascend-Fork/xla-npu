#ifndef XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_OP_INTERFACE_H_
#define XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_OP_INTERFACE_H_

#include "xla/xla_npu/dependency/header/acl_nn/acl_meta.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"

#define DECLARE_ACL_FUNC_IN_OPAPI(func_name, args_type, args_name) \
    DECLARE_ACL_FUNC(libopapi, aclnnStatus, func_name, args_type, args_name)

// 所有算子执行接口的形参声明是完全相同的
#define DECLARE_ACL_EXEC_FUNC_IN_OPAPI(func_name)                                                                \
    DECLARE_ACL_FUNC_IN_OPAPI(                                                                                   \
        func_name, (void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream), \
        (workspace, workspaceSize, executor, stream))

namespace npu {

DECLARE_ACL_FUNC_IN_OPAPI(aclnnExpandGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* size, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, size, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnExpand)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnAddGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, other, alpha, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnAdd)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnTanGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnTan)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnTanhGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnTanh)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMulGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMul)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnDivGetWorkspaceSize,
                            (const aclTensor* self, const aclTensor* other, aclTensor* out,
                             uint64_t* workspaceSize, aclOpExecutor** executor),
                            (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnDiv)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnSubGetWorkspaceSize,
                            (const aclTensor* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out,
                             uint64_t* workspaceSize, aclOpExecutor** executor),
                            (self, other, alpha, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSub)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnNegGetWorkspaceSize,
                            (const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                             aclOpExecutor** executor),
                            (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnNeg)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnExpGetWorkspaceSize,
                            (const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                             aclOpExecutor** executor),
                            (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnExp)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMaximumGetWorkspaceSize,
                            (const aclTensor* self, const aclTensor* other, aclTensor* out,
                             uint64_t* workspaceSize, aclOpExecutor** executor),
                            (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMaximum)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMatmulGetWorkspaceSize,
                            (const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType,
                             uint64_t* workspaceSize, aclOpExecutor** executor),
                            (self, mat2, out, cubeMathType, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMatmul)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnCatGetWorkspaceSize,
                            (const aclTensorList* tensors, int64_t dim, aclTensor* out,
                             uint64_t* workspaceSize, aclOpExecutor** executor),
                            (tensors, dim, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnCat)

// Reduce sum operation
DECLARE_ACL_FUNC_IN_OPAPI(aclnnReduceSumGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dims, bool keepDims, aclDataType dtype,
                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dims, keepDims, dtype, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnReduceSum)

// Copy operation for reshape
DECLARE_ACL_FUNC_IN_OPAPI(aclnnInplaceCopyGetWorkspaceSize,
                          (aclTensor* selfRef, const aclTensor* src, uint64_t* workspaceSize, aclOpExecutor** executor),
                          (selfRef, src, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnInplaceCopy)

// MaxDim operation for reduce max
DECLARE_ACL_FUNC_IN_OPAPI(aclnnMaxDimGetWorkspaceSize,
                          (const aclTensor* self, int64_t dim, bool keepdim, aclTensor* out, aclTensor* indices,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, keepdim, out, indices, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMaxDim)

// Amax operation for reduce max (returns max values only, no indices)
DECLARE_ACL_FUNC_IN_OPAPI(aclnnAmaxGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dim, bool keepDim, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, keepDim, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnAmax)

}  // namespace npu

#endif  // XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_OP_INTERFACE_H_
