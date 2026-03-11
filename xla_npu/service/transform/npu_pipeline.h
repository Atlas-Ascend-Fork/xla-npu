/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PIPELINE_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PIPELINE_H_

#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {

// Function wrapper around the XLA NPU transform pipeline.
// This is the main entry point for all NPU optimization passes.
// Thread pool may be nullptr.
HloPassPipeline NpuPipeline(const DebugOptions& debug_options,
                            HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
                            tsl::thread::ThreadPool* thread_pool, const se::DeviceDescription& npu_device_info,
                            SymbolicExprContext* symbolic_expr_context);

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PIPELINE_H_
