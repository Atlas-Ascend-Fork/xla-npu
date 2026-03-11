/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DYNAMIC_MEMCPY_REWRITER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DYNAMIC_MEMCPY_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// FusionDynamicMemcpyRewriter-like pass for NPU.
// Detects fusions that can be emitted using DynamicMemcpyFusion and annotates them accordingly.
class NpuFusionDynamicMemcpyRewriter : public HloModulePass {
   public:
    absl::string_view name() const override { return "npu-fusion-dynamic-memcpy-rewriter"; }

   protected:
    absl::StatusOr<bool> RunImpl(HloModule* module,
                                 const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DYNAMIC_MEMCPY_REWRITER_H_
