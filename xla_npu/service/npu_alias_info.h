/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_NPU_ALIAS_INFO_H_
#define XLA_XLA_NPU_SERVICE_NPU_ALIAS_INFO_H_

#include <optional>

#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

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

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_ALIAS_INFO_H_
