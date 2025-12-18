#ifndef REGISTER_ACLINTERFACE_H_
#define REGISTER_ACLINTERFACE_H_

#include "xla/xla_npu/third_party/acl/acl.h"

namespace npu {

/**
  This API is used to get error msg
  */
const char *AclGetErrMsg();

} // namespace npu

#endif // REGISTER_ACLINTERFACE_H_