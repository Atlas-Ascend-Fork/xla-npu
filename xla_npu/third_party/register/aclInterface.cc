#include "xla/xla_npu/third_party/register/aclInterface.h"

#include "xla/xla_npu/third_party/register/functionLoader.h"

namespace npu {

REGISTER_LIBRARY(libascendcl)
REGISTER_FUNCTION(libascendcl, aclInit)
REGISTER_FUNCTION(libascendcl, aclFinalize)
REGISTER_FUNCTION(libascendcl, aclrtGetDeviceCount)
REGISTER_FUNCTION(libascendcl, aclrtGetDevice)
REGISTER_FUNCTION(libascendcl, aclrtSetDevice)
REGISTER_FUNCTION(libascendcl, aclrtCreateStream)
REGISTER_FUNCTION(libascendcl, aclrtDestroyStream)
REGISTER_FUNCTION(libascendcl, aclrtSynchronizeStream)
REGISTER_FUNCTION(libascendcl, aclrtMalloc)
REGISTER_FUNCTION(libascendcl, aclrtFree)
REGISTER_FUNCTION(libascendcl, aclrtMemcpy)
REGISTER_FUNCTION(libascendcl, aclrtMallocHost)
REGISTER_FUNCTION(libascendcl, aclrtFreeHost)

}  // namespace npu
