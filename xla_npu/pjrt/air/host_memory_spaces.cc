/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tsl/platform/fingerprint.h"

#include "host_memory_spaces.h"

namespace xla {

NpuDeviceMemorySpace::NpuDeviceMemorySpace(int id, PjRtDevice* device) : id_(id), device_(device)
{
    DCHECK(device_ != nullptr && device_->client() != nullptr);
    auto* client = device_->client();
    debug_string_ = absl::StrFormat("NpuDeviceMemory(id=%i, process_index=%i, client=%s)", id_, client->process_index(),
                                    client->platform_name());
    to_string_ = absl::StrFormat("NPU_DEVICE_%i", id_);
}

const int NpuDeviceMemorySpace::kKindId = []() {
    uint32_t kind_id = tsl::Fingerprint32(NpuDeviceMemorySpace::kKind);
    return static_cast<int>(kind_id);
}();

}  // namespace xla