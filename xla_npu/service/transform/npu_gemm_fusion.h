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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_GEMM_FUSION_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_GEMM_FUSION_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"

namespace xla {

// NPU fusion kind constants
constexpr char kNpuAclnnGemmFusionKind[] = "__npu_aclnn_gemm";
constexpr char kNpuNestedGemmFusionKind[] = "__npu_nested_gemm";
constexpr char kNpuCustomFusionKind[] = "__npu_custom";
constexpr char kNpuBlockFusionKind[] = "__npu_block";

// Supported activation functions for ACLNN GEMM fusion
enum class ActivationMode {
    kNone = 0,
    kRelu = 1,
    kGelu = 2,
    kSigmoid = 3,
    kTanh = 4,
};

// NPU GEMM fusion configuration
struct NpuGemmFusionConfig {
    // Whether to fuse bias addition
    bool has_bias = false;
    // Activation mode after GEMM
    ActivationMode activation = ActivationMode::kNone;
    // Whether to use NPU native layout (FRACTAL_NZ)
    bool use_nz_layout = true;
    // Block size for NPU computation
    int block_m = 128;
    int block_n = 128;
    int block_k = 32;
};

// 将 GEMM 操作重写为 NPU 优化的融合计算
// 该 PASS 检测适合 NPU ACLNN 的 GEMM 模式，并将它们转换为 custom fusion。
// 支持的模式包括:
// - 基础 GEMM (MatMul)
// - GEMM + Bias
// - GEMM + Bias + Activation (ReLU, GELU 等)
// 转换示例: Before: dot(lhs, rhs) -> add(bias) -> gelu
//  After: fusion(lhs, rhs, bias) with backend_config.kind="__npu_aclnn_gemm"
class NpuGemmFusion : public HloModulePass {
 public:
    explicit NpuGemmFusion(const se::DeviceDescription& device_description,
                           mlir::MLIRContext* mlir_context);
    ~NpuGemmFusion() override = default;

    absl::string_view name() const override { return "npu-gemm-fusion"; }

    // 检查 GEMM 是否应该由 NPU 处理
    bool ShouldNpuHandleGEMM(const HloDotInstruction& dot);

    // 分析 GEMM 融合模式并生成配置
    absl::StatusOr<NpuGemmFusionConfig> AnalyzeGemmFusionPattern(
            const HloDotInstruction& dot);

    // 创建 Dot 融合
    absl::StatusOr<HloInstruction*> CreateDotFusion(
            const HloDotInstruction& dot, HloModule* module,
            const NpuGemmFusionConfig& config);

 protected:
    absl::StatusOr<bool> RunImpl(
            HloModule* module,
            const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:

    // 检查操作是否可以融合到 GEMM 中
    bool IsGemmSupportedElementwise(const HloInstruction* instr);

    // 向操作数方向融合
    absl::StatusOr<bool> FuseDotOperand(HloInstruction* dot,
                                        HloInstruction* operand,
                                        HloComputation::Builder& builder,
                                        std::vector<HloInstruction*>& fusion_params);

    // 向输出方向融合
    absl::StatusOr<bool> FuseDotOutput(HloInstruction* dot, HloComputation::Builder& builder,
                                      std::vector<HloInstruction*>& fusion_params,
                                      NpuGemmFusionConfig& config);

    // 检查是否为支持的激活函数
    ActivationMode GetActivationMode(const HloInstruction* instr);

    const se::DeviceDescription& device_description_;
    mlir::MLIRContext* mlir_context_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_GEMM_FUSION_H_
