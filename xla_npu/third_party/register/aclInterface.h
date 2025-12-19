#ifndef REGISTER_ACLINTERFACE_H_
#define REGISTER_ACLINTERFACE_H_

#include "xla/xla_npu/third_party/acl/acl.h"

namespace npu {

/**
  This API is used to get error msg
  */
const char *AclGetErrMsg();
const aclError AclInit(const char *configPath);
const aclError AclrtGetDeviceCount(uint32_t *count);
const aclError AclrtGetDevice(uint32_t *device_id);
const aclError AclrtSetDevice(uint32_t device_id);
} // namespace npu

#endif // REGISTER_ACLINTERFACE_H_