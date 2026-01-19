#include "xla/xla_npu/third_party/register/aclnnopInterface.h"

#include "xla/xla_npu/third_party/register/functionLoader.h"

namespace npu {

REGISTER_LIBRARY(libopapi)
REGISTER_FUNCTION(libopapi, aclnnExpandGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnExpand)
REGISTER_FUNCTION(libopapi, aclnnAddGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnAdd)

}  // namespace npu
