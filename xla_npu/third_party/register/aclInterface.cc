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

} // namespace npu
