#ifndef SERVICE_NPU_ALIAS_INFO_H_
#define SERVICE_NPU_ALIAS_INFO_H_

#include <optional>

#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
class NpuAliasInfo : public AliasInfo {
   public:
    explicit NpuAliasInfo(const se::DeviceDescription& device_description) : device_description_(device_description) {}

    // Backend-specific may-alias hint. If an empty optional is returned, the
    // default rules in HloDataflowAnalysis are used. `operand` should be an
    // operand of `user`. `operand_index` should be the output index of `operand`,
    // `user_index` should be the output index of `user`.
    std::optional<bool> MayAlias(const HloInstruction* operand, const ShapeIndex& operand_index,
                                 const HloInstruction* user, const ShapeIndex& user_index) const override;

   protected:
    se::DeviceDescription device_description_;
};
}  // namespace xla::gpu

#endif  // SERVICE_NPU_ALIAS_INFO_H_
