/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_NPU_STREAM_EXECUTOR_UTIL_H_
#define XLA_XLA_NPU_SERVICE_NPU_STREAM_EXECUTOR_UTIL_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/layout.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
absl::Mutex& GetNpuMutex(const se::StreamExecutor* stream_exec);
}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_STREAM_EXECUTOR_UTIL_H_
