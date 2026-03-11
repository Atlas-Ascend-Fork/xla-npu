/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_WRAPPER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_WRAPPER_H_

#include "absl/strings/string_view.h"
#include "xla/codegen/emitters/fusion_wrapper_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

// Wraps leftover unfused instructions in the entry computation that have no NPU-specific
// equivalent by containing just that instruction in a fusion.
class NpuFusionWrapper : public emitters::FusionWrapperBase {
public:
    explicit NpuFusionWrapper(const se::DeviceDescription& device_description)
        : device_description_(device_description) {}

    absl::string_view name() const override { return "npu-fusion-wrapper"; }

    bool MustWrapInstruction(HloOpcode opcode) override;
    HloInstruction::FusionKind ChooseFusionKind(
        const HloInstruction& producer, const HloInstruction& consumer) override;

private:
    const se::DeviceDescription& device_description_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_WRAPPER_H_
