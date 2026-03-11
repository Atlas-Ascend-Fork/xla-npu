/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_INTERFACE_H_
#define XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_INTERFACE_H_

#include "xla/xla_npu/dependency/header/acl_nn/acl_meta.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"

#define DECLARE_ACL_FUNC_IN_NNOPBASE(ret, func_name, args_type, args_name) \
  DECLARE_ACL_FUNC(libnnopbase, ret, func_name, args_type, args_name)

namespace npu {

DECLARE_ACL_FUNC_IN_NNOPBASE(aclTensor*, aclCreateTensor,
                             (const int64_t* viewDims, uint64_t viewDimsNum,
                              aclDataType dataType, const int64_t* stride,
                              int64_t offset, aclFormat format,
                              const int64_t* storageDims,
                              uint64_t storageDimsNum, void* tensorData),
                             (viewDims, viewDimsNum, dataType, stride, offset,
                              format, storageDims, storageDimsNum, tensorData))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclScalar*, aclCreateScalar,
                             (void* value, aclDataType dataType),
                             (value, dataType))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclIntArray*, aclCreateIntArray,
                             (const int64_t* value, uint64_t size),
                             (value, size))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclTensorList*, aclCreateTensorList,
                             (const aclTensor* const* value, uint64_t size),
                             (value, size))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclnnStatus, aclDestroyTensorList,
                             (const aclTensorList* array), (array))

DECLARE_ACL_FUNC_IN_NNOPBASE(void, aclDestroyIntArray,
                             (const aclIntArray* array), (array))

}  // namespace npu

#endif  // XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_INTERFACE_H_
