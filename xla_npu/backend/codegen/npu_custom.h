/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_BACKEND_CODEGEN_NPU_CUSTOM_H_
#define XLA_XLA_NPU_BACKEND_CODEGEN_NPU_CUSTOM_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/xla_npu/backend/codegen/npu_fusion_emitter.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

namespace xla {

// A wrapper for fusions implemented using the mechanism in
// xla/service/gpu/kernels. See custom_kernel_fusion.h in that folder for
// details.
class NpuCustomFusion : public NpuFusionInterface {
 public:
  absl::StatusOr<NpuFusionEmissionResult> Emit(
      NpuIrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_CODEGEN_NPU_CUSTOM_H_
