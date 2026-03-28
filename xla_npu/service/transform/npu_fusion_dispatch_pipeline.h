/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DISPATCH_PIPELINE_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DISPATCH_PIPELINE_H_

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {

// NPU融合分发管道
// 该PASS负责协调NPU融合优化的执行顺序。它执行以下操作：
// 1. 先运行GEMM特定的融合优化
// 2. 然后运行自定义kernel融合重写
// 3. 接着将适合的融合转换为块级融合
// 4. 最后执行嵌套GEMM融合转换
// 与GPU的FusionDispatchPipeline不同，NPU版本：
// - 不包含Triton特定的优化
// - 专注于CANN/ACLNN后端的融合模式
// - 包含NPU特定的内存布局优化
class NpuFusionDispatchPipeline : public HloModulePass {
 public:
    NpuFusionDispatchPipeline(
            const se::DeviceDescription& device_description,
            SymbolicExprContext* symbolic_expr_context,
            const DebugOptions& debug_options,
            HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
            tsl::thread::ThreadPool* thread_pool);

    ~NpuFusionDispatchPipeline() override = default;

    absl::string_view name() const override {
        return "npu-fusion-dispatch-pipeline";
    }

 protected:
    absl::StatusOr<bool> RunImpl(
            HloModule* module,
            const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
    const se::DeviceDescription device_description_;
    SymbolicExprContext* symbolic_expr_context_;
    const DebugOptions debug_options_;
    HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function_;
    tsl::thread::ThreadPool* thread_pool_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DISPATCH_PIPELINE_H_
