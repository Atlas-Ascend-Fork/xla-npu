#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_SUPER_FUSION_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_SUPER_FUSION_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// TODO remove
// 融合所有hlo instruction,仅用作调试
// A pass that fuses all HLO instructions (except parameters) into a single
// large fusion instruction for the purpose of unified execution.
// This is primarily for debugging and testing purposes.
class NpuSuperFusion : public HloModulePass {
   public:
    NpuSuperFusion() = default;

    absl::string_view name() const override { return "npu-super-fusion"; }

    using HloPassInterface::Run;  // Use base class Run method

   protected:
    absl::StatusOr<bool> RunImpl(HloModule* module,
                                 const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_SUPER_FUSION_H_
