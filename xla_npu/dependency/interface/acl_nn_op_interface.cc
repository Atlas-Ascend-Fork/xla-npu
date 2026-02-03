#include "xla/xla_npu/dependency/interface/acl_nn_op_interface.h"

#include "xla/xla_npu/dependency/tool/function_loader.h"

namespace npu {

REGISTER_LIBRARY(libopapi)
REGISTER_FUNCTION(libopapi, aclnnExpandGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnExpand)
REGISTER_FUNCTION(libopapi, aclnnAddGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnAdd)
REGISTER_FUNCTION(libopapi, aclnnTanGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnTan)
REGISTER_FUNCTION(libopapi, aclnnTanhGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnTanh)
REGISTER_FUNCTION(libopapi, aclnnMulGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnMul)

}  // namespace npu
