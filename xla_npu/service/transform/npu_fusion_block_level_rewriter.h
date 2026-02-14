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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_BLOCK_LEVEL_REWRITER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_BLOCK_LEVEL_REWRITER_H_

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

// 将现有的 loop/input 融合转换为 NPU 块级融合
//
// 该 PASS 将 NPU 后端的 loop/input 融合转换为块级融合，使用 "__npu_block" kind。
// 块级融合可以更好地利用 NPU 的统一缓冲区 (UB) 内存层级和块级并行计算。
//
// 转换示例:
// Before: fusion with kind="kLoop" that operates elementwise
//  After: fusion with kind="__npu_block" with block-level tiling config
//
// 块级配置包括:
// - 输出平铺大小 (output_tile_sizes)
// - 块数量 (num_blocks)
// - 阶段数量 (num_stages) - 用于流水线优化
class NpuFusionBlockLevelRewriter : public HloModulePass {
 public:
    explicit NpuFusionBlockLevelRewriter(
            const se::DeviceDescription& device_description)
        : device_description_(device_description) {}

    ~NpuFusionBlockLevelRewriter() override = default;

    absl::string_view name() const override {
        return "npu-fusion-block-level-rewriter";
    }

 protected:
    absl::StatusOr<bool> RunImpl(
            HloModule* module,
            const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
    // Check if a fusion can be converted to block-level
    bool CanConvertToBlockLevel(const HloInstruction* fusion);

    // Run on a single computation
    absl::StatusOr<bool> RunOnComputation(HloComputation* computation);

    // Convert a loop/input fusion to block-level fusion
    absl::StatusOr<bool> ConvertFusionToBlockLevel(HloFusionInstruction* fusion);

    // Compute optimal tile sizes for the fusion
    absl::StatusOr<std::vector<int64_t>> ComputeTileSizes(
            const HloInstruction* fusion);

    // Check if the fusion contains only elementwise operations
    bool ContainsOnlyElementwiseOps(HloComputation* fusion);

    // Set up block-level configuration for the fusion
    absl::Status SetBlockLevelConfig(HloFusionInstruction* fusion,
                                      const std::vector<int64_t>& tile_sizes,
                                      int64_t num_blocks);

    const se::DeviceDescription device_description_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_BLOCK_LEVEL_REWRITER_H_
