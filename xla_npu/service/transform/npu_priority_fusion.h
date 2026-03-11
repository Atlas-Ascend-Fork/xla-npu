/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PRIORITY_FUSION_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PRIORITY_FUSION_H_

#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

// PriorityFusion-like pass for NPU.
class NpuPriorityFusion : public HloModulePass {
   public:
    NpuPriorityFusion(tsl::thread::ThreadPool* thread_pool, const se::DeviceDescription& device,
                      HloCostAnalysis::Options cost_analysis_options, SymbolicExprContext* symbolic_expr_context)
        : thread_pool_(thread_pool),
          device_info_(device),
          cost_analysis_options_(std::move(cost_analysis_options)),
          symbolic_expr_context_(symbolic_expr_context) {}

    absl::string_view name() const override { return "npu-priority-fusion"; }

   protected:
    HloInstruction::FusionKind ChooseKind(const HloInstruction* producer, const HloInstruction* consumer);

    HloInstruction* Fuse(HloInstruction* producer, HloInstruction* consumer, bool use_multi_output_fusion = false);

    absl::StatusOr<bool> RunImpl(HloModule* module,
                                 const absl::flat_hash_set<absl::string_view>& execution_threads) override;

   private:
    // Consumes a unit of compiler fuel and returns true if we should
    // continue with the transformation.
    bool ConsumeFuel(HloInstruction* producer, HloInstruction* consumer);

    // Returns the decision if the constant can be fused into the user.
    bool CanFuseConstant(const HloInstruction* constant, const HloInstruction* user);

    tsl::thread::ThreadPool* thread_pool_;
    const se::DeviceDescription& device_info_;

    // Cost model options that defines priorities in the queue.
    // Using the base HloCostAnalysis::Options (not GPU-specific).
    HloCostAnalysis::Options cost_analysis_options_;

    SymbolicExprContext* symbolic_expr_context_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PRIORITY_FUSION_H_
