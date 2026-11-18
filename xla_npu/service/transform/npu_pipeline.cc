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

#include "xla/xla_npu/service/transform/npu_pipeline.h"

#include <memory>
#include <utility>

#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/literal_util.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/xla_npu/service/transform/npu_fusion_pipeline.h"
#include "xla/xla_npu/service/transform/npu_hlo_printer_pass.h"

namespace xla {

namespace {

// XLA-NPU's PJRT path does not propagate in-place DynamicUpdateSlice aliases
// back to the caller, so the result buffer still holds the old data. Force a
// materialized root by adding zero to every DUS result; the DUS still updates
// its operand in place, but all consumers read the fresh buffer.
class MaterializeDynamicUpdateSlice : public HloModulePass {
 public:
  absl::string_view name() const override { return "materialize-dus"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override {
    bool changed = false;
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      if (computation->IsFusionComputation()) continue;
      std::vector<HloInstruction*> dus_list;
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
          dus_list.push_back(instruction);
        }
      }
      for (HloInstruction* dus : dus_list) {
        // XLA-NPU cannot expose an in-place DUS result to later consumers or
        // to PJRT. Materialize every DUS through an opaque copy.
        HloInstruction* materialized = computation->AddInstruction(
            HloInstruction::CreateCustomCall(
                dus->shape(), absl::MakeSpan(&dus, 1), "npu$copy"));
        if (computation->root_instruction() == dus) {
          computation->set_root_instruction(materialized);
        } else {
          TF_RETURN_IF_ERROR(dus->ReplaceAllUsesWith(materialized));
        }
        changed = true;
      }
    }
    return changed;
  }
};

}  // namespace

HloPassPipeline NpuPipeline(
    const DebugOptions& debug_options,
    HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
    tsl::thread::ThreadPool* thread_pool,
    const se::DeviceDescription& npu_device_info,
    SymbolicExprContext* symbolic_expr_context) {
  HloPassFix<HloPassPipeline> pipeline("npu-pipeline");

  HloVerifierOpts opts =
      HloVerifierOpts().MakeLayoutSensitive().WithInstructionCanChangeLayout(
          LayoutAssignment::InstructionCanChangeLayout);
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts)),
      "hlo verifier (npu)");

  pipeline.AddPass<NpuHloPrinterPass>("Npu pipeline initial");

  pipeline.AddPass<FlattenCallGraph>();
  pipeline.AddPass<NpuHloPrinterPass>("FlattenCallGraph");

  // ZML and other PJRT frontends can produce a trivial tuple/get-tuple-element
  // root; the AFIR converter handles arrays only.
  pipeline.AddPass<TupleSimplifier>();
  pipeline.AddPass<NpuHloPrinterPass>("TupleSimplifier");

#if NPU_AUTO_FUSE_BACKEND != 0
  pipeline.AddPass(std::make_unique<HloPassPipeline>(
      NpuFusionPipeline(debug_options, std::move(shape_size_bytes_function),
                        thread_pool, npu_device_info, symbolic_expr_context)));
#endif

  pipeline.AddPass<MaterializeDynamicUpdateSlice>();
  pipeline.AddPass<NpuHloPrinterPass>("MaterializeDynamicUpdateSlice");

  return std::move(pipeline);
}

}  // namespace xla
