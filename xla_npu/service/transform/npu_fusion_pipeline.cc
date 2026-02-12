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
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/gpu/transforms/variadic_op_splitter.h"
#include "xla/service/gpu/transforms/rename_fusions.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/tsl/platform/threadpool.h"

// NPU-specific fusion passes
#include "xla/xla_npu/service/transform/npu_priority_fusion.h"
#include "xla/xla_npu/service/transform/npu_multi_output_fusion.h"
#include "xla/xla_npu/service/transform/npu_copy_fusion.h"
#include "xla/xla_npu/service/transform/npu_dynamic_slice_fusion_rewriter.h"
#include "xla/xla_npu/service/transform/npu_fusion_dynamic_memcpy_rewriter.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

HloPassPipeline NpuFusionPipeline(
    const DebugOptions& debug_options,
    HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
    tsl::thread::ThreadPool* thread_pool,
    const se::DeviceDescription& npu_device_info,
    SymbolicExprContext* symbolic_expr_context) {
  HloPassFix<HloPassPipeline> fusion("npu-fusion");

  // We try to split variadic ops with many parameters into several such ops
  // to avoid exceeding the parameter space. Reuse GPU implementation as it's
  // pure HLO manipulation (not GPU-specific).
  fusion.AddPass<gpu::VariadicOpSplitter>();

  HloVerifierOpts opts =
      HloVerifierOpts().MakeLayoutSensitive().WithInstructionCanChangeLayout(
          LayoutAssignment::InstructionCanChangeLayout);
  fusion.AddInvariantCheckerDebug<HloVerifier>(
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts)),
      "hlo verifier (npufusion)");

  // NPU cost analysis options - use base HloCostAnalysis::Options
  HloCostAnalysis::Options cost_analysis_options{
      shape_size_bytes_function,
      /*per_second_rates=*/{},
      /*min_latencies_seconds=*/{},
      /*count_multiple_input_accesses=*/true};

  // Priority fusion pass for NPU - independent implementation
  fusion.AddPass<NpuPriorityFusion>(thread_pool, npu_device_info,
                                    std::move(cost_analysis_options),
                                    symbolic_expr_context);

  // Running CSE affects how many users an op has. This plays a role in what
  // we detect as fusion patterns.
  fusion.AddPass<HloCSE>(
      /*is_layout_sensitive=*/true, /*ignore_control_dependencies=*/false,
      /*should_eliminate_computation=*/&HloComputation::IsFusionComputation);
  fusion.AddPass<HloDCE>();

  // Multi-output fusion for NPU
  fusion.AddPass<NpuMultiOutputFusion>(npu_device_info, shape_size_bytes_function,
                                       symbolic_expr_context);

  // Copy fusion for NPU
  fusion.AddPass<NpuCopyFusion>(npu_device_info);

  fusion.AddPass<HloCSE>(
      /*is_layout_sensitive=*/true, /*ignore_control_dependencies=*/false,
      /*should_eliminate_computation=*/&HloComputation::IsFusionComputation);
  fusion.AddPass<HloDCE>();

  // Dynamic slice fusion rewriter for NPU
  fusion.AddPass<NpuDynamicSliceFusionRewriter>("NPU");

  // Fusion dynamic memcpy rewriter for NPU
  fusion.AddPass<NpuFusionDynamicMemcpyRewriter>();

  // Rename fusions - reuse GPU implementation as it's pure HLO manipulation
  fusion.AddPass<gpu::RenameFusions>();

  fusion.AddPass<HloDCE>();

  return std::move(fusion);
}

}  // namespace xla
