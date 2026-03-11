/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_BACKEND_CODEGEN_NPU_FUSION_EMITTER_H_
#define XLA_XLA_NPU_BACKEND_CODEGEN_NPU_FUSION_EMITTER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

namespace xla {

struct NpuFusionEmissionResult {
  std::vector<std::unique_ptr<NpuThunk>> thunks;
};

class NpuFusionInterface {
 public:
  virtual ~NpuFusionInterface() = default;

  virtual absl::StatusOr<NpuFusionEmissionResult> Emit(
      NpuIrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const = 0;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_CODEGEN_NPU_FUSION_EMITTER_H_
