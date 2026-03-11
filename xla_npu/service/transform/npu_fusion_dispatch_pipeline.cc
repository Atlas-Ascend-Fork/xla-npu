/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/service/transform/npu_fusion_dispatch_pipeline.h"

#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status.h"
#include "xla/xla.pb.h"

// NPU fusion transforms
#include "xla/xla_npu/service/transform/npu_custom_kernel_fusion_rewriter.h"
#include "xla/xla_npu/service/transform/npu_fusion_block_level_rewriter.h"
#include "xla/xla_npu/service/transform/npu_fusion_dynamic_memcpy_rewriter.h"
#include "xla/xla_npu/service/transform/npu_gemm_fusion.h"
#include "xla/xla_npu/service/transform/npu_gemm_fusion_swap_operands.h"
#include "xla/xla_npu/service/transform/npu_nest_gemm_fusion.h"

namespace xla {

NpuFusionDispatchPipeline::NpuFusionDispatchPipeline(
        const se::DeviceDescription& device_description,
        SymbolicExprContext* symbolic_expr_context,
        const DebugOptions& debug_options,
        HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
        tsl::thread::ThreadPool* thread_pool)
    : device_description_(device_description),
      symbolic_expr_context_(symbolic_expr_context),
      debug_options_(debug_options),
      shape_size_bytes_function_(std::move(shape_size_bytes_function)),
      thread_pool_(thread_pool) {}

absl::StatusOr<bool> NpuFusionDispatchPipeline::RunImpl(
        HloModule* module,
        const absl::flat_hash_set<absl::string_view>& execution_threads) {
    bool changed = false;
    bool pass_changed;

    // Phase 1: Preliminary optimizations
    VLOG(2) << "Running NPU Fusion Dispatch Pipeline - Phase 1: Preliminary";

    // Run DCE to clean up
    HloDCE dce_pass;
    TF_ASSIGN_OR_RETURN(pass_changed, dce_pass.Run(module, execution_threads));
    changed |= pass_changed;

    // Run CSE to eliminate common subexpressions
    HloCSE cse_pass(/*is_layout_sensitive=*/true,
                     /*ignore_control_dependencies=*/false,
                     /*should_eliminate_computation=*/nullptr);
    TF_ASSIGN_OR_RETURN(pass_changed, cse_pass.Run(module, execution_threads));
    changed |= pass_changed;

    mlir::MLIRContext mlir_context;

    // Phase 2: GEMM-specific optimizations
    VLOG(2) << "Running NPU Fusion Dispatch Pipeline - Phase 2: GEMM optimizations";

    // Swap GEMM operands for better layout
    NpuGemmFusionSwapOperands swap_operands_pass;
    TF_ASSIGN_OR_RETURN(pass_changed,
                        swap_operands_pass.Run(module, execution_threads));
    changed |= pass_changed;

    // Create GEMM fusions (this creates NPU ACLNN GEMM fusions)
    NpuGemmFusion gemm_fusion_pass(device_description_, &mlir_context);
    TF_ASSIGN_OR_RETURN(pass_changed,
                        gemm_fusion_pass.Run(module, execution_threads));
    changed |= pass_changed;

    // Phase 3: Clean up after GEMM fusion
    VLOG(2) << "Running NPU Fusion Dispatch Pipeline - Phase 3: GEMM cleanup";

    TF_ASSIGN_OR_RETURN(pass_changed, dce_pass.Run(module, execution_threads));
    changed |= pass_changed;

    TF_ASSIGN_OR_RETURN(pass_changed, cse_pass.Run(module, execution_threads));
    changed |= pass_changed;

    // Phase 4: Custom kernel fusion
    VLOG(2) << "Running NPU Fusion Dispatch Pipeline - Phase 4: Custom kernels";

    NpuCustomKernelFusionRewriter custom_kernel_pass;
    TF_ASSIGN_OR_RETURN(pass_changed,
                        custom_kernel_pass.Run(module, execution_threads));
    changed |= pass_changed;

    // Phase 5: Block-level fusion conversion
    VLOG(2) << "Running NPU Fusion Dispatch Pipeline - Phase 5: Block-level fusion";

    NpuFusionBlockLevelRewriter block_level_pass(device_description_);
    TF_ASSIGN_OR_RETURN(pass_changed,
                        block_level_pass.Run(module, execution_threads));
    changed |= pass_changed;

    // Phase 6: Nested GEMM fusion
    VLOG(2) << "Running NPU Fusion Dispatch Pipeline - Phase 6: Nested GEMM";

    NpuNestGemmFusion nest_gemm_pass(device_description_, symbolic_expr_context_);
    TF_ASSIGN_OR_RETURN(pass_changed,
                        nest_gemm_pass.Run(module, execution_threads));
    changed |= pass_changed;

    // Phase 7: Dynamic memcpy rewriter
    VLOG(2) << "Running NPU Fusion Dispatch Pipeline - Phase 7: Dynamic memcpy";

    NpuFusionDynamicMemcpyRewriter dynamic_memcpy_pass;
    TF_ASSIGN_OR_RETURN(pass_changed,
                        dynamic_memcpy_pass.Run(module, execution_threads));
    changed |= pass_changed;

    // Phase 8: Final cleanup
    VLOG(2) << "Running NPU Fusion Dispatch Pipeline - Phase 8: Final cleanup";

    TF_ASSIGN_OR_RETURN(pass_changed, dce_pass.Run(module, execution_threads));
    changed |= pass_changed;

    TF_ASSIGN_OR_RETURN(pass_changed, cse_pass.Run(module, execution_threads));
    changed |= pass_changed;

    VLOG(1) << "NPU Fusion Dispatch Pipeline completed. Changed: " << changed;
    return changed;
}

}  // namespace xla
