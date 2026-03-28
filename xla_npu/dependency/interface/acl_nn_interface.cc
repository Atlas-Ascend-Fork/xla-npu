/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
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
