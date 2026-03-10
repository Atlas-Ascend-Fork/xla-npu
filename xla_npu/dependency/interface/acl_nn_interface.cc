/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/dependency/interface/acl_nn_interface.h"

#include "xla/xla_npu/dependency/tool/function_loader.h"

namespace npu {

REGISTER_LIBRARY(libnnopbase)
REGISTER_FUNCTION(libnnopbase, aclCreateTensor)
REGISTER_FUNCTION(libnnopbase, aclCreateScalar)
REGISTER_FUNCTION(libnnopbase, aclCreateIntArray)
REGISTER_FUNCTION(libnnopbase, aclCreateTensorList)
REGISTER_FUNCTION(libnnopbase, aclDestroyTensorList)
REGISTER_FUNCTION(libnnopbase, aclDestroyIntArray)

}  // namespace npu
