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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_GEMM_FUSION_SWAP_OPERANDS_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_GEMM_FUSION_SWAP_OPERANDS_H_

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"

namespace xla {

// NPU指定的内存布局格式
enum class NpuLayoutFormat {
    kDefault,      // 默认布局 (Row-major)
    kNC1HWC0,      // NPU专用卷积布局
    kFRACTAL_NZ,   // NPU矩阵计算的高效布局
    kNDHWC,        // 3D卷积布局
    kNCHW,         // 通道优先布局
};

// 交换GEMM操作数以获得更好的NPU布局/分块性能
//
// 该PASS分析GEMM操作并可能交换操作数顺序以优化NPU内存布局。
// NPU对FRACTAL_NZ布局有特定的访问模式，交换操作数可以：
// 1. 更好地利用NPU的分块计算能力
// 2. 减少内存访问延迟
// 3. 提高缓存利用率
//
// 转换示例:
// Before: dot(A, B) where A.shape = [M, K], B.shape = [K, N]
// After:  dot(transpose(B), transpose(A)) with adjusted dimension mapping
//
// 注意: 交换操作数需要在dot_dimension_numbers中相应地调整维度映射。
class NpuGemmFusionSwapOperands : public HloModulePass {
 public:
    NpuGemmFusionSwapOperands() = default;
    ~NpuGemmFusionSwapOperands() override = default;

    absl::string_view name() const override {
        return "npu-gemm-fusion-swap-operands";
    }

 protected:
    absl::StatusOr<bool> RunImpl(
            HloModule* module,
            const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
    // Run on a single computation
    absl::StatusOr<bool> RunOnComputation(HloComputation* computation);

    // Check if GEMM operands should be swapped
    bool ShouldSwapOperands(const HloDotInstruction* dot);

    // Get the preferred layout format for a given tensor
    NpuLayoutFormat GetPreferredLayoutFormat(const Shape& shape,
                                             bool is_lhs);

    // Check if the shape is compatible with a given layout format
    bool IsLayoutCompatible(const Shape& shape, NpuLayoutFormat format);

    // Swap the operands of a GEMM instruction
    absl::StatusOr<bool> SwapGemmOperands(HloDotInstruction* dot);

    // Create a transpose instruction for layout conversion
    absl::StatusOr<HloInstruction*> CreateLayoutTranspose(
            HloInstruction* operand,
            const std::vector<int64_t>& permutation);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_GEMM_FUSION_SWAP_OPERANDS_H_
