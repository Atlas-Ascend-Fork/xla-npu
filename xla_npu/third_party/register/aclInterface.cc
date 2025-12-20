#include "functionLoader.h"
#include "aclInterface.h"

namespace npu {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libascendcl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libascendcl, funcName)

REGISTER_LIBRARY(libascendcl)
LOAD_FUNCTION(aclGetRecentErrMsg)
LOAD_FUNCTION(aclInit)
LOAD_FUNCTION(aclrtGetDeviceCount)
LOAD_FUNCTION(aclrtGetDevice)
LOAD_FUNCTION(aclrtSetDevice)
LOAD_FUNCTION(aclrtMalloc)
LOAD_FUNCTION(aclrtFree)

const char *AclGetErrMsg()
{
    typedef const char *(*aclGetErrMsg)();
    static aclGetErrMsg func = nullptr;
    if (func == nullptr) {
        func = (aclGetErrMsg)GET_FUNC(aclGetRecentErrMsg);
    }
    if (func != nullptr) {
        return func();
    }
    return "";
}

aclError AclInit(const char *configPath)
{
    typedef aclError (*AclInit)(const char*);
    static AclInit func = nullptr;
    if (func == nullptr) {
        func = (AclInit) GET_FUNC(aclInit);
    }
    return func(configPath);
}

aclError AclrtGetDeviceCount(uint32_t *count) 
{
    typedef aclError (*AclrtGetDeviceCount)(uint32_t*);
    static AclrtGetDeviceCount func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetDeviceCount) GET_FUNC(aclrtGetDeviceCount);
    }
    return func(count);
}

aclError AclrtGetDevice(uint32_t *device_id)
{
    typedef aclError (*AclrtGetDevice)(uint32_t*);
    static AclrtGetDevice func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetDevice) GET_FUNC(aclrtGetDevice);
    }
    return func(device_id);
}

aclError AclrtSetDevice(uint32_t device_id)
{
    typedef aclError (*AclrtSetDevice)(uint32_t);
    static AclrtSetDevice func = nullptr;
    if (func == nullptr) {
        func = (AclrtSetDevice) GET_FUNC(aclrtSetDevice);
    }
    return func(device_id);
}

aclError AclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    typedef aclError (*aclrtMallocFunc)(void **, size_t, aclrtMemMallocPolicy);
    static aclrtMallocFunc func = nullptr;
    if (func == nullptr) {
        func = (aclrtMallocFunc)GET_FUNC(aclrtMalloc);
    }
    if (func != nullptr) {
        return func(devPtr, size, policy);
    }
    return ACL_ERROR_RT_INTERNAL_ERROR;
}

aclError AclrtFree(void *devPtr)
{
    typedef aclError (*aclrtFreeFunc)(void *);
    static aclrtFreeFunc func = nullptr;
    if (func == nullptr) {
        func = (aclrtFreeFunc)GET_FUNC(aclrtFree);
    }
    if (func != nullptr) {
        return func(devPtr);
    }
    return ACL_ERROR_RT_INTERNAL_ERROR;
}

} // namespace npu
