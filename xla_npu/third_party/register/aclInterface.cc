#include "aclInterface.h"

#include "functionLoader.h"

namespace npu {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) REGISTER_FUNCTION(libascendcl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName) GET_FUNCTION(libascendcl, funcName)

REGISTER_LIBRARY(libascendcl)
LOAD_FUNCTION(aclGetRecentErrMsg)
LOAD_FUNCTION(aclInit)
LOAD_FUNCTION(aclrtGetDeviceCount)
LOAD_FUNCTION(aclrtGetDevice)
LOAD_FUNCTION(aclrtSetDevice)
LOAD_FUNCTION(aclrtCreateStream)
LOAD_FUNCTION(aclrtSynchronizeStream)
LOAD_FUNCTION(aclrtMalloc)
LOAD_FUNCTION(aclrtFree)
LOAD_FUNCTION(aclrtMemcpy)
LOAD_FUNCTION(aclrtMallocHost)
LOAD_FUNCTION(aclrtFreeHost)

const char *AclGetErrMsg() {
    typedef const char *(*aclGetErrMsgFunc)();
    static aclGetErrMsgFunc func = nullptr;
    if (func == nullptr) {
        func = (aclGetErrMsgFunc)GET_FUNC(aclGetRecentErrMsg);
    }
    if (func != nullptr) {
        return func();
    }
    return "";
}

aclError AclInit(const char *configPath) {
    typedef aclError (*AclInitFunc)(const char*);
    static AclInitFunc func = nullptr;
    if (func == nullptr) {
        func = (AclInitFunc) GET_FUNC(aclInit);
    }
    if (func != nullptr) {
        return func(configPath);
    }
    return ACL_ERROR_RT_INTERNAL_ERROR;
}

aclError AclrtGetDeviceCount(uint32_t *count) {
    typedef aclError (*AclrtGetDeviceCountFunc)(uint32_t*);
    static AclrtGetDeviceCountFunc func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetDeviceCountFunc) GET_FUNC(aclrtGetDeviceCount);
    }
    if (func != nullptr) {
        return func(count);
    }
    return ACL_ERROR_RT_INTERNAL_ERROR;
}

aclError AclrtGetDevice(uint32_t *device_id) {
    typedef aclError (*AclrtGetDeviceFunc)(uint32_t*);
    static AclrtGetDeviceFunc func = nullptr;
    if (func == nullptr) {
        func = (AclrtGetDeviceFunc) GET_FUNC(aclrtGetDevice);
    }
    if (func != nullptr) {
        return func(device_id);
    }
    return ACL_ERROR_RT_INTERNAL_ERROR;
}

aclError AclrtSetDevice(uint32_t device_id) {
    typedef aclError (*AclrtSetDeviceFunc)(uint32_t);
    static AclrtSetDeviceFunc func = nullptr;
    if (func == nullptr) {
        func = (AclrtSetDeviceFunc) GET_FUNC(aclrtSetDevice);
    }
    if (func != nullptr) {
        return func(device_id);
    }
    return ACL_ERROR_RT_INTERNAL_ERROR;
}

aclError AclrtCreateStream(aclrtStream *stream) {
    typedef aclError (*aclrtCreateStreamFunc)(aclrtStream*);
    static aclrtCreateStreamFunc func = nullptr;
    if (func == nullptr) {
        func = (aclrtCreateStreamFunc) GET_FUNC(aclrtCreateStream);
    }
    if (func != nullptr) {
        return func(stream);
    }
    return ACL_ERROR_RT_INTERNAL_ERROR;
}

aclError AclrtSynchronizeStream(aclrtStream stream) {
    typedef aclError (*aclrtSynchronizeStreamFunc)(aclrtStream);
    static aclrtSynchronizeStreamFunc func = nullptr;
    if (func == nullptr) {
        func = (aclrtSynchronizeStreamFunc) GET_FUNC(aclrtSynchronizeStream);
    }
    if (func != nullptr) {
        return func(stream);
    }
    return ACL_ERROR_RT_INTERNAL_ERROR;
}

aclError AclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
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

aclError AclrtFree(void *devPtr) {
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

aclError AclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
    typedef aclError (*aclrtMemcpyFunc)(void *, size_t, const void *, size_t, aclrtMemcpyKind);
    static aclrtMemcpyFunc func = nullptr;
    if (func == nullptr) {
        func = (aclrtMemcpyFunc)GET_FUNC(aclrtMemcpy);
    }
    if (func != nullptr) {
        return func(dst, destMax, src, count, kind);
    }
    return ACL_ERROR_RT_INTERNAL_ERROR;
}

aclError AclrtMallocHost(void **hostPtr, size_t size) {
    typedef aclError (*aclrtMallocHostFunc)(void **hostPtr, size_t size);
    static aclrtMallocHostFunc func = nullptr;
    if (func == nullptr) {
        func = (aclrtMallocHostFunc)GET_FUNC(aclrtMallocHost);
    }
    if (func != nullptr) {
        return func(hostPtr, size);
    }
    return ACL_ERROR_RT_INTERNAL_ERROR;
}

aclError AclrtFreeHost(void *hostPtr) {
    typedef aclError (*aclrtFreeHostFunc)(void *hostPtr);
    static aclrtFreeHostFunc func = nullptr;
    if (func == nullptr) {
        func = (aclrtFreeHostFunc)GET_FUNC(aclrtFreeHost);
    }
    if (func != nullptr) {
        return func(hostPtr);
    }
    return ACL_ERROR_RT_INTERNAL_ERROR;
}

} // namespace npu
