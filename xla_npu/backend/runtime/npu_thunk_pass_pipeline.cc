/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/backend/runtime/npu_thunk_pass_pipeline.h"

#include <memory>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
// #include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"

namespace xla {

absl::StatusOr<bool> NpuThunkPassPipeline::Run(NpuSequentialThunk* root_thunk, const DebugOptions& debug_options,
                                               const HloModule* absl_nullable hlo_module,
                                               const se::DeviceDescription& device_info,
                                               NpuThunkPassBufferAllocator& allocator) {
    bool changed = false;
    for (const auto& pass : passes_) {
        VLOG(1) << "Running ThunkPass: " << pass->name();
        TF_ASSIGN_OR_RETURN(bool pass_changed,
                            pass->Run(root_thunk, debug_options, hlo_module, device_info, allocator));
        changed |= pass_changed;
    }
    return changed;
}

}  // namespace xla
