/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_NPU_LATENCY_HIDING_SCHEDULER_H_
#define XLA_XLA_NPU_SERVICE_NPU_LATENCY_HIDING_SCHEDULER_H_

#include <cstdint>
#include <optional>

#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"

namespace xla {

HloCostAnalysis::ShapeSizeFunction NpuShapeSizeBytesFunction(
    int64_t pointer_size, std::optional<int64_t> memory_space);

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_LATENCY_HIDING_SCHEDULER_H_
