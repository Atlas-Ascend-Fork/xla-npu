#ifndef REGISTER_ACLINTERFACE_H_
#define REGISTER_ACLINTERFACE_H_

#include "xla/xla_npu/third_party/acl/acl.h"

namespace npu {

/**
  This API is used to get error msg
  */
const char *AclGetErrMsg();
aclError AclInit(const char *configPath);
aclError AclrtGetDeviceCount(uint32_t *count);
aclError AclrtGetDevice(uint32_t *device_id);
aclError AclrtSetDevice(uint32_t device_id);
aclError AclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
aclError AclrtFree(void *devPtr);
aclError AclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);

} // namespace npu

#endif // REGISTER_ACLINTERFACE_H_
