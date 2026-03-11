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

#ifndef XLA_XLA_NPU_CLIENT_NPU_STREAM_CLIENT_H_
#define XLA_XLA_NPU_CLIENT_NPU_STREAM_CLIENT_H_

#include "absl/status/statusor.h"
#include "xla/xla_npu/pjrt/npu/npu_pjrt_stream_executor_client.h"

namespace xla {

absl::StatusOr<std::unique_ptr<NpuPjrtStreamExecutorClient>>
GetNpuPjrtStreamExecutorClient();

}  // namespace xla

#endif  // XLA_XLA_NPU_CLIENT_NPU_STREAM_CLIENT_H_