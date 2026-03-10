/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/dependency/interface/acl_interface.h"

#include "xla/xla_npu/dependency/tool/function_loader.h"

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
REGISTER_FUNCTION(libascendcl, aclrtMemset)
REGISTER_FUNCTION(libascendcl, aclrtMallocHost)
REGISTER_FUNCTION(libascendcl, aclrtFreeHost)
REGISTER_FUNCTION(libascendcl, aclrtBinaryLoadFromFile)
REGISTER_FUNCTION(libascendcl, aclrtBinaryLoadFromData)
REGISTER_FUNCTION(libascendcl, aclrtBinaryGetFunction)
REGISTER_FUNCTION(libascendcl, aclrtKernelArgsInit)
REGISTER_FUNCTION(libascendcl, aclrtKernelArgsAppend)
REGISTER_FUNCTION(libascendcl, aclrtKernelArgsAppendPlaceHolder)
REGISTER_FUNCTION(libascendcl, aclrtKernelArgsGetPlaceHolderBuffer)
REGISTER_FUNCTION(libascendcl, aclrtKernelArgsFinalize)
REGISTER_FUNCTION(libascendcl, aclrtLaunchKernel)
REGISTER_FUNCTION(libascendcl, aclrtLaunchKernelWithConfig)

}  // namespace npu
