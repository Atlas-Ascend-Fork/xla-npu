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

}  // namespace npu

#endif  // XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_OP_INTERFACE_H_
