/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_STREAM_EXECUTOR_NPU_PLATFORM_ID_H_
#define XLA_XLA_NPU_STREAM_EXECUTOR_NPU_PLATFORM_ID_H_

#include "xla/stream_executor/platform.h"

namespace stream_executor {

// Opaque and unique identifier for the npu platform.
// This is needed so that plugins can refer to/identify this platform without
// instantiating a NpuPlatform object.
// This is broken out here to avoid a circular dependency between NpuPlatform
// and NpuStreamExecutor.
extern const Platform::Id kNpuPlatformId;

}  // namespace stream_executor

#endif  // XLA_XLA_NPU_STREAM_EXECUTOR_NPU_PLATFORM_ID_H_
