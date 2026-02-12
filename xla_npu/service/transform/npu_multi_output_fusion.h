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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_MULTI_OUTPUT_FUSION_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_MULTI_OUTPUT_FUSION_H_

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_dfs_reachability.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

// Multi-output fusion pass for NPU, completely independent of GPU implementation.
// Uses base HloCostAnalysis (not GPU-specific).
class NpuMultiOutputFusion : public HloModulePass {
 public:
  NpuMultiOutputFusion(const se::DeviceDescription& device_description,
                       HloCostAnalysis::ShapeSizeFunction shape_size_function,
                       SymbolicExprContext* symbolic_expr_context)
      : device_description_(device_description),
        shape_size_function_(shape_size_function),
        symbolic_expr_context_(symbolic_expr_context) {}

  absl::string_view name() const override { return "npu-multi-output-fusion"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::DeviceDescription& device_description_;
  HloCostAnalysis::ShapeSizeFunction shape_size_function_;
  SymbolicExprContext* symbolic_expr_context_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_MULTI_OUTPUT_FUSION_H_
