#include "xla/xla_npu/dependency/interface/acl_nn_interface.h"

#include "xla/xla_npu/dependency/tool/function_loader.h"

namespace npu {

REGISTER_LIBRARY(libnnopbase)
REGISTER_FUNCTION(libnnopbase, aclCreateTensor)
REGISTER_FUNCTION(libnnopbase, aclCreateScalar)
REGISTER_FUNCTION(libnnopbase, aclCreateIntArray)
REGISTER_FUNCTION(libnnopbase, aclCreateTensorList)
REGISTER_FUNCTION(libnnopbase, aclDestroyTensorList)

}  // namespace npu
