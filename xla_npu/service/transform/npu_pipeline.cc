/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/xla_npu/service/transform/npu_pipeline.h"

#include <memory>
#include <utility>

#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/xla_npu/service/transform/npu_fusion_pipeline.h"
#include "xla/xla_npu/service/transform/npu_hlo_printer_pass.h"

namespace xla {

HloPassPipeline NpuPipeline(const DebugOptions& debug_options,
                            HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
                            tsl::thread::ThreadPool* thread_pool, const se::DeviceDescription& npu_device_info,
                            SymbolicExprContext* symbolic_expr_context) {
    HloPassFix<HloPassPipeline> pipeline("npu-pipeline");

    HloVerifierOpts opts = HloVerifierOpts().MakeLayoutSensitive().WithInstructionCanChangeLayout(
        LayoutAssignment::InstructionCanChangeLayout);
    pipeline.AddInvariantCheckerDebug<HloVerifier>(std::make_unique<CpuGpuVerifierMetadata>(std::move(opts)),
                                                   "hlo verifier (npu)");

    pipeline.AddPass<NpuHloPrinterPass>("Npu pipeline initial");

    pipeline.AddPass<FlattenCallGraph>();
    pipeline.AddPass<NpuHloPrinterPass>("FlattenCallGraph");

    pipeline.AddPass(std::make_unique<HloPassPipeline>(NpuFusionPipeline(
        debug_options, std::move(shape_size_bytes_function), thread_pool, npu_device_info, symbolic_expr_context)));

    return std::move(pipeline);
}

}  // namespace xla
