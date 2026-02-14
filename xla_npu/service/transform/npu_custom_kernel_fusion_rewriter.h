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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_CUSTOM_KERNEL_FUSION_REWRITER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_CUSTOM_KERNEL_FUSION_REWRITER_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"

namespace xla {

// Custom kernel pattern for NPU fusion
struct NpuCustomKernelPattern {
    // Pattern name (e.g., "acl_gemm_fusion", "acl_softmax")
    std::string name;
    // Target fusion kind
    std::string fusion_kind;
    // Match function - returns true if pattern matches
    bool (*match)(const std::vector<const HloInstruction*>&);
    // Rewrite function - creates the fusion
    absl::StatusOr<HloInstruction*> (*rewrite)(
            const std::vector<const HloInstruction*>&, HloModule*);
};

// 模式匹配 HLO 指令到 CANN 自定义内核融合
//
// 该 PASS 检测支持 CANN API 自定义调用的模式，并将它们转换为 custom fusion。
// 支持的模式包括:
// - acl_gemm_fusion: GEMM with fusion patterns
// - acl_softmax: Softmax operation
// - acl_layer_norm: Layer normalization
// - acl_matmul: MatMul operation
//
// 转换示例:
// Before: softmax(logits) -> output
//  After: fusion(logits) with backend_config.kind="__npu_custom" and custom_call_target="acl_softmax"
class NpuCustomKernelFusionRewriter : public HloModulePass {
 public:
    NpuCustomKernelFusionRewriter() = default;
    ~NpuCustomKernelFusionRewriter() override = default;

    absl::string_view name() const override {
        return "npu-custom-kernel-fusion-rewriter";
    }

 protected:
    absl::StatusOr<bool> RunImpl(
            HloModule* module,
            const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
    // Register custom kernel patterns
    void RegisterPatterns();

    // Check if instruction matches a custom kernel pattern
    absl::StatusOr<NpuCustomKernelPattern*> FindMatchingPattern(
            const HloInstruction* instr);

    // Check if a sequence of instructions matches a softwax pattern
    static bool MatchSoftmaxPattern(
            const std::vector<const HloInstruction*>& instrs);

    // Check if a sequence of instructions matches a layer norm pattern
    static bool MatchLayerNormPattern(
            const std::vector<const HloInstruction*>& instrs);

    // Rewrite softmax pattern to custom fusion
    static absl::StatusOr<HloInstruction*> RewriteSoftmax(
            const std::vector<const HloInstruction*>& instrs, HloModule* module);

    // Rewrite layer norm pattern to custom fusion
    static absl::StatusOr<HloInstruction*> RewriteLayerNorm(
            const std::vector<const HloInstruction*>& instrs, HloModule* module);

    // Run on a single computation
    absl::StatusOr<bool> RunOnComputation(HloComputation* computation);

    std::vector<NpuCustomKernelPattern> patterns_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_CUSTOM_KERNEL_FUSION_REWRITER_H_
