#ifndef REGISTER_ACLNNOPINTERFACE_H_
#define REGISTER_ACLNNOPINTERFACE_H_

#include "xla/xla_npu/third_party/aclnn/acl_meta.h"
#include "xla/xla_npu/third_party/register/functionLoader.h"

#define DECLARE_ACL_FUNC_IN_OPAPI(func_name, args_type, args_name) \
    DECLARE_ACL_FUNC(libopapi, aclnnStatus, func_name, args_type, args_name)

namespace npu {

DECLARE_ACL_FUNC_IN_OPAPI(aclnnExpandGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* size, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, size, out, workspaceSize, executor))

DECLARE_ACL_FUNC_IN_OPAPI(aclnnExpand,
                          (void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream),
                          (workspace, workspaceSize, executor, stream))

DECLARE_ACL_FUNC_IN_OPAPI(aclnnAddGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, other, alpha, out, workspaceSize, executor))

DECLARE_ACL_FUNC_IN_OPAPI(aclnnAdd,
                          (void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream),
                          (workspace, workspaceSize, executor, stream))

DECLARE_ACL_FUNC_IN_OPAPI(aclnnTanGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_FUNC_IN_OPAPI(aclnnTan,
                          (void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream),
                          (workspace, workspaceSize, executor, stream))

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMulGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMul,
                          (void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream),
                          (workspace, workspaceSize, executor, stream))

}  // namespace npu

#endif  // REGISTER_ACLNNOPINTERFACE_H_
