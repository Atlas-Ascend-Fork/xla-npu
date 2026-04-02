/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_NEST_GEMM_FUSION_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_NEST_GEMM_FUSION_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"

namespace xla {

// 重写 NPU GEMM 融合为嵌套块级融合
// 该 PASS 将 kCustom 融合且 fusion_backend_config.kind 为 "__npu_aclnn_gemm"
// 的 GEMM 融合重写为嵌套融合，使用 "__npu_nested_gemm" kind。
// 嵌套融合将 GEMM 操作分解为多个块级计算，每个块级计算有自己的配置。
// 这种模式可以更好地利用 NPU 的块级内存布局和并行计算能力。
// 转换示例: Before: fusion with kind="__npu_aclnn_gemm"
//  After: fusion with kind="__npu_nested_gemm"
//         - lhs nested fusion (for left operand processing)
//         - rhs nested fusion (for right operand processing)
class NpuNestGemmFusion : public HloModulePass {
 public:
    explicit NpuNestGemmFusion(const se::DeviceDescription& device_description,
                               SymbolicExprContext* symbolic_expr_context)
        : device_description_(device_description),
          symbolic_expr_context_(symbolic_expr_context) {}

    ~NpuNestGemmFusion() override = default;

    absl::string_view name() const override { return "npu-nest-gemm-fusion"; }

 protected:
    absl::StatusOr<bool> RunImpl(
            HloModule* module,
            const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
    // Check if fusion is a NPU GEMM fusion that can be nested
    bool IsNpuGemmFusion(const HloInstruction* fusion);

    // Run on a single computation
    absl::StatusOr<bool> RunOnComputation(HloComputation* computation);

    // Transform a GEMM fusion into nested fusion
    absl::StatusOr<bool> TransformGemmFusionToNested(HloFusionInstruction* fusion);

    // Find the dot instruction in a fusion
    absl::StatusOr<HloInstruction*> FindDotInFusion(HloComputation* fusion);

    // Create a nested fusion for the given operand fusion
    absl::StatusOr<HloInstruction*> CreateNestedFusionForOperand(
            HloInstruction* operand,
            HloInstruction* consumer,
            absl::string_view kind,
            int64_t tile_size);

    // Update the nested fusion config
    absl::Status UpdateNestedFusionConfig(HloFusionInstruction* nested_fusion,
                                          int64_t tile_size,
                                          int64_t num_blocks);

    const se::DeviceDescription device_description_;
    SymbolicExprContext* symbolic_expr_context_;
};

// NPU fusion kind constants
namespace {
constexpr char kNpuAclnnGemmFusionKind[] = "__npu_aclnn_gemm";
constexpr char kNpuNestedGemmFusionKind[] = "__npu_nested_gemm";
constexpr char kNpuCustomFusionKind[] = "__npu_custom";
constexpr char kNpuBlockingFusionKind[] = "__npu_blocking";
}  // namespace

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_NEST_GEMM_FUSION_H_
