/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/pjrt/plugin/npu_device_description.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/xla_npu/common/npu_constant.h"

namespace xla {
NpuDeviceDescription::NpuDeviceDescription(int process_id, int local_device_id)
    : id_(PjRtGlobalDeviceId(process_id * 16 + local_device_id)),
      process_index_(process_id),
      local_hardware_id_(local_device_id) {
    debug_string_ = absl::StrCat("XLART_NPU_", id_.value());
    to_string_ = absl::StrCat("NpuDevice(id=", id_.value(), ")");
}

absl::string_view NpuDeviceDescription::device_kind() const { return NpuConstant::platformName; }

absl::string_view NpuDeviceDescription::DebugString() const { return debug_string_; }

absl::string_view NpuDeviceDescription::ToString() const { return to_string_; }
}  // namespace xla