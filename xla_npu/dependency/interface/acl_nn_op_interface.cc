/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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
REGISTER_FUNCTION(libopapi, aclnnReduceSumGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnReduceSum)
REGISTER_FUNCTION(libopapi, aclnnInplaceCopyGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnInplaceCopy)
REGISTER_FUNCTION(libopapi, aclnnAmaxGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnAmax)
REGISTER_FUNCTION(libopapi, aclnnMaxDimGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnMaxDim)

}  // namespace npu
