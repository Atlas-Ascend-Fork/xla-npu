/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_COMMON_NPU_OPERATION_H_
#define XLA_XLA_NPU_COMMON_NPU_OPERATION_H_

#include "absl/status/status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"

namespace xla {

// npu硬件交互接口封装类
class NpuOperation {
   public:
    enum class MemcpyType { H2D, D2H };

    static absl::Status SetDevice(int device_id);
    static absl::Status CreateStream(aclrtStream* stream);
    static absl::Status Malloc(void** new_device_addr, size_t size_bytes);
    static absl::Status Memcpy(void* dst, void* src, size_t size_bytes, MemcpyType memcpyType);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_COMMON_NPU_OPERATION_H_