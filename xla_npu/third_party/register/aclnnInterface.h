#ifndef REGISTER_ACLNNINTERFACE_H_
#define REGISTER_ACLNNINTERFACE_H_

#include "xla/xla_npu/third_party/aclnn/acl_meta.h"

namespace npu {

aclTensor *AclCreateTensor(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
                          const int64_t *stride, int64_t offset, aclFormat format,
                          const int64_t *storageDims, uint64_t storageDimsNum,
                          void *tensorData);

} // namespace npu

#endif // REGISTER_ACLNNINTERFACE_H_
