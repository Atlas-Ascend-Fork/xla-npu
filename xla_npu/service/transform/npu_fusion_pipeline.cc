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

#include "xla/xla_npu/service/transform/npu_fusion_pipeline.h"

#include <memory>
#include <utility>

#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
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
#include "xla/xla_npu/service/transform/npu_gelu_fusion_rewriter.h"
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

    // Flatten call graph - must run before buffer assignment
    // This ensures each call site has a unique computation, which is required
    // for correct buffer assignment and dataflow analysis.
    fusion.AddPass<FlattenCallGraph>();

    // GELU fusion rewriter - identify and fuse GELU activation patterns
    // Run BEFORE priority fusion to prevent GELU operations from being split into small fusions
    fusion.AddPass<NpuGeluFusionRewriter>();
    // fusion.AddPass<NpuHloPrinterPass>("NpuGeluFusionRewriter");

    // NpuGemmFusionSwapOperands - swap GEMM operands for better NPU layout
    // Run BEFORE priority fusion to optimize GEMM operand order
    fusion.AddPass<NpuGemmFusionSwapOperands>();
    // fusion.AddPass<NpuHloPrinterPass>("NpuGemmFusionSwapOperands");

    // Priority fusion pass for NPU - independent implementation
    fusion.AddPass<NpuPriorityFusion>(thread_pool, npu_device_info, std::move(cost_analysis_options),
                                      symbolic_expr_context);
    // fusion.AddPass<NpuHloPrinterPass>("NpuPriorityFusion");

    // Running CSE affects how many users an op has. This plays a role in what
    // we detect as fusion patterns.
    fusion.AddPass<HloCSE>(
        /*is_layout_sensitive=*/true, /*ignore_control_dependencies=*/false,
        /*should_eliminate_computation=*/&HloComputation::IsFusionComputation);
    fusion.AddPass<HloDCE>();
    // fusion.AddPass<NpuHloPrinterPass>("CSE + DCE after PriorityFusion");

    // Multi-output fusion for NPU
    fusion.AddPass<NpuMultiOutputFusion>(npu_device_info, shape_size_bytes_function, symbolic_expr_context);
    // fusion.AddPass<NpuHloPrinterPass>("NpuMultiOutputFusion");

    // Copy fusion for NPU
    fusion.AddPass<NpuCopyFusion>(npu_device_info);
    // fusion.AddPass<NpuHloPrinterPass>("NpuCopyFusion");

    // Create MLIR context for passes that need it
    mlir::MLIRContext mlir_context;

    // NpuGemmFusion - create ACLNN GEMM fusions
    // Run after basic fusions to convert suitable GEMM patterns
    fusion.AddPass<NpuGemmFusion>(npu_device_info, &mlir_context);
    // fusion.AddPass<NpuHloPrinterPass>("NpuGemmFusion");

    fusion.AddPass<HloCSE>(
        /*is_layout_sensitive=*/true, /*ignore_control_dependencies=*/false,
        /*should_eliminate_computation=*/&HloComputation::IsFusionComputation);
    fusion.AddPass<HloDCE>();
    // fusion.AddPass<NpuHloPrinterPass>("CSE + DCE after GemmFusion");

    // Dynamic slice fusion rewriter for NPU
    fusion.AddPass<NpuDynamicSliceFusionRewriter>("NPU");
    // fusion.AddPass<NpuHloPrinterPass>("NpuDynamicSliceFusionRewriter");

    // Fusion dynamic memcpy rewriter for NPU
    fusion.AddPass<NpuFusionDynamicMemcpyRewriter>();
    // fusion.AddPass<NpuHloPrinterPass>("NpuFusionDynamicMemcpyRewriter");

    // NpuCustomKernelFusionRewriter - match CANN custom kernel patterns
    fusion.AddPass<NpuCustomKernelFusionRewriter>();
    // fusion.AddPass<NpuHloPrinterPass>("NpuCustomKernelFusionRewriter");

    // NpuFusionBlockLevelRewriter - convert loop/input fusions to block-level
    fusion.AddPass<NpuFusionBlockLevelRewriter>(npu_device_info);
    // fusion.AddPass<NpuHloPrinterPass>("NpuFusionBlockLevelRewriter");

    // NpuNestGemmFusion - convert GEMM fusions to nested fusions
    fusion.AddPass<NpuNestGemmFusion>(npu_device_info, symbolic_expr_context);
    // fusion.AddPass<NpuHloPrinterPass>("NpuNestGemmFusion");

    // DISABLED: Fusion wrapper - causes NPU runtime error 507057 (Synchronize stream failed)
    // TODO: Fix NpuFusionWrapper to generate correct NPU fusion graph
    // fusion.AddPass<NpuFusionWrapper>(npu_device_info);

    // Rename fusions - use NPU implementation for completeness
    fusion.AddPass<NpuRenameFusions>();
    // fusion.AddPass<NpuHloPrinterPass>("NpuRenameFusions");

    // Final HloDCE pass
    fusion.AddPass<HloDCE>();
    fusion.AddPass<NpuHloPrinterPass>("Final HloDCE");

    return std::move(fusion);
}

}  // namespace xla
