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
REGISTER_FUNCTION(libopapi, aclnnMatmulGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnMatmul)
REGISTER_FUNCTION(libopapi, aclnnDivGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnDiv)
REGISTER_FUNCTION(libopapi, aclnnSubGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnSub)
REGISTER_FUNCTION(libopapi, aclnnNegGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnNeg)
REGISTER_FUNCTION(libopapi, aclnnExpGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnExp)
REGISTER_FUNCTION(libopapi, aclnnMaximumGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnMaximum)
REGISTER_FUNCTION(libopapi, aclnnCatGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnCat)

}  // namespace npu
