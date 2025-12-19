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

const aclError AclInit(const char *configPath)
{
    typedef aclError (*AclInit)(const char*);
    static AclInit func = nullptr;
    if (func == nullptr) {
        func = (AclInit) GET_FUNC(aclInit);
    }
    return func(configPath);
}

const aclError AclrtGetDeviceCount(uint32_t *count) 
{
    typedef aclError (*AclrtGetDeviceCount)(uint32_t*);
    static AclrtGetDeviceCount func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetDeviceCount) GET_FUNC(aclrtGetDeviceCount);
    }
    return func(count);
}

const aclError AclrtGetDevice(uint32_t *device_id)
{
    typedef aclError (*AclrtGetDevice)(uint32_t*);
    static AclrtGetDevice func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetDevice) GET_FUNC(aclrtGetDevice);
    }
    return func(device_id);
}

const aclError AclrtSetDevice(uint32_t device_id)
{
    typedef aclError (*AclrtSetDevice)(uint32_t);
    static AclrtSetDevice func = nullptr;
    if (func == nullptr) {
        func = (AclrtSetDevice) GET_FUNC(aclrtSetDevice);
    }
    return func(device_id);
}

} // namespace npu
