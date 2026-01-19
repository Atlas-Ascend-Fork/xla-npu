#include "xla/xla_npu/third_party/register/aclnnInterface.h"

#include "xla/xla_npu/third_party/register/functionLoader.h"

namespace npu {

REGISTER_LIBRARY(libnnopbase)
REGISTER_FUNCTION(libnnopbase, aclCreateTensor)
REGISTER_FUNCTION(libnnopbase, aclCreateScalar)
REGISTER_FUNCTION(libnnopbase, aclCreateIntArray)

}  // namespace npu
