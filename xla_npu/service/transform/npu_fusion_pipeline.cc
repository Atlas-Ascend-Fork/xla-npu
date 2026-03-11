/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/service/transform/npu_fusion_pipeline.h"

#include <memory>
#include <utility>

#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

// NPU-specific fusion passes
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/transform/npu_copy_fusion.h"
#include "xla/xla_npu/service/transform/npu_custom_kernel_fusion_rewriter.h"
#include "xla/xla_npu/service/transform/npu_dynamic_slice_fusion_rewriter.h"
#include "xla/xla_npu/service/transform/npu_fusion_block_level_rewriter.h"
#include "xla/xla_npu/service/transform/npu_fusion_dynamic_memcpy_rewriter.h"
#include "xla/xla_npu/service/transform/npu_fusion_dispatch_pipeline.h"
#include "xla/xla_npu/service/transform/npu_fusion_wrapper.h"
#include "xla/xla_npu/service/transform/npu_gemm_fusion.h"
#include "xla/xla_npu/service/transform/npu_gemm_fusion_swap_operands.h"
#include "xla/xla_npu/service/transform/npu_hlo_printer_pass.h"
#include "xla/xla_npu/service/transform/npu_multi_output_fusion.h"
#include "xla/xla_npu/service/transform/npu_nest_gemm_fusion.h"
#include "xla/xla_npu/service/transform/npu_priority_fusion.h"
#include "xla/xla_npu/service/transform/npu_rename_fusions.h"

namespace xla {

HloPassPipeline NpuFusionPipeline(const DebugOptions& debug_options,
                                  HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
                                  tsl::thread::ThreadPool* thread_pool, const se::DeviceDescription& npu_device_info,
                                  SymbolicExprContext* symbolic_expr_context) {
    HloPassFix<HloPassPipeline> fusion("npu-fusion");

    HloVerifierOpts opts = HloVerifierOpts().MakeLayoutSensitive().WithInstructionCanChangeLayout(
        LayoutAssignment::InstructionCanChangeLayout);
    fusion.AddInvariantCheckerDebug<HloVerifier>(std::make_unique<CpuGpuVerifierMetadata>(std::move(opts)),
                                                 "hlo verifier (npufusion)");

    // NPU cost analysis options - use base HloCostAnalysis::Options
    HloCostAnalysis::Options cost_analysis_options{shape_size_bytes_function,
                                                   /*per_second_rates=*/{},
                                                   /*min_latencies_seconds=*/{},
                                                   /*count_multiple_input_accesses=*/true};

    // Print initial HLO
    fusion.AddPass<NpuHloPrinterPass>("Npu fusion initial");

    // NpuGemmFusionSwapOperands - swap GEMM operands for better NPU layout
    // Run BEFORE priority fusion to optimize GEMM operand order
    fusion.AddPass<NpuGemmFusionSwapOperands>();

    // Priority fusion pass for NPU - independent implementation
    fusion.AddPass<NpuPriorityFusion>(thread_pool, npu_device_info, std::move(cost_analysis_options),
                                      symbolic_expr_context);

    // Running CSE affects how many users an op has. This plays a role in what
    // we detect as fusion patterns.
    fusion.AddPass<HloCSE>(
        /*is_layout_sensitive=*/true, /*ignore_control_dependencies=*/false,
        /*should_eliminate_computation=*/&HloComputation::IsFusionComputation);
    fusion.AddPass<HloDCE>();

    // Copy fusion for NPU
    fusion.AddPass<NpuCopyFusion>(npu_device_info);

    // Create MLIR context for passes that need it
    mlir::MLIRContext mlir_context;

    // NpuGemmFusion - create ACLNN GEMM fusions
    // Run after basic fusions to convert suitable GEMM patterns
    fusion.AddPass<NpuGemmFusion>(npu_device_info, &mlir_context);

    fusion.AddPass<HloCSE>(
        /*is_layout_sensitive=*/true, /*ignore_control_dependencies=*/false,
        /*should_eliminate_computation=*/&HloComputation::IsFusionComputation);
    fusion.AddPass<HloDCE>();

    // Dynamic slice fusion rewriter for NPU
    fusion.AddPass<NpuDynamicSliceFusionRewriter>("NPU");

    // Fusion dynamic memcpy rewriter for NPU
    fusion.AddPass<NpuFusionDynamicMemcpyRewriter>();

    // NpuCustomKernelFusionRewriter - match CANN custom kernel patterns
    fusion.AddPass<NpuCustomKernelFusionRewriter>();

    // NpuFusionBlockLevelRewriter - convert loop/input fusions to block-level
    fusion.AddPass<NpuFusionBlockLevelRewriter>(npu_device_info);

    // NpuNestGemmFusion - convert GEMM fusions to nested fusions
    fusion.AddPass<NpuNestGemmFusion>(npu_device_info, symbolic_expr_context);

    // Rename fusions - use NPU implementation for completeness
    fusion.AddPass<NpuRenameFusions>();

    // Final HloDCE pass
    fusion.AddPass<HloDCE>();
    fusion.AddPass<NpuHloPrinterPass>("Final HloDCE");

    return std::move(fusion);
}

}  // namespace xla
