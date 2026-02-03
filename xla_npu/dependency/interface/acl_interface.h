#ifndef REGISTER_ACLINTERFACE_H_
#define REGISTER_ACLINTERFACE_H_

#include "xla/xla_npu/dependency/header/acl/acl.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"

#define DECLARE_ACL_FUNC_IN_ASCENDCL(func_name, args_type, args_name) \
    DECLARE_ACL_FUNC(libascendcl, aclError, func_name, args_type, args_name)

namespace npu {

DECLARE_ACL_FUNC_IN_ASCENDCL(aclInit, (const char* configPath), (configPath))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclFinalize, (), ())

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtGetDeviceCount, (uint32_t* count), (count))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtGetDevice, (uint32_t* device_id), (device_id))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtSetDevice, (uint32_t device_id), (device_id))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtCreateStream, (aclrtStream * stream), (stream))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtDestroyStream, (aclrtStream stream), (stream))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtSynchronizeStream, (aclrtStream stream), (stream))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtMalloc, (void** devPtr, size_t size, aclrtMemMallocPolicy policy),
                             (devPtr, size, policy))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtFree, (void* devPtr), (devPtr))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtMemcpy,
                             (void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind),
                             (dst, destMax, src, count, kind))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtMallocHost, (void** hostPtr, size_t size), (hostPtr, size))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtFreeHost, (void* hostPtr), (hostPtr))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtLaunchKernel,
                             (aclrtFuncHandle funcHandle, uint32_t blockDim, const void* argsData, size_t argsSize,
                              aclrtStream stream),
                             (funcHandle, blockDim, argsData, argsSize, stream))

}  // namespace npu

#endif  // REGISTER_ACLINTERFACE_H_
