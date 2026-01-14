#ifndef REGISTER_ACLNNOPINTERFACE_H_
#define REGISTER_ACLNNOPINTERFACE_H_

#include "xla/xla_npu/third_party/aclnn/acl_meta.h"

namespace npu {

aclnnStatus AclnnExpandGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* size, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus AclnnExpand(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

} // namespace npu

#endif // REGISTER_ACLNNOPINTERFACE_H_
