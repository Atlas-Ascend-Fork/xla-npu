/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_PJRT_PLUGIN_NPU_DEVICE_DESCRIPTION_H_
#define XLA_XLA_NPU_PJRT_PLUGIN_NPU_DEVICE_DESCRIPTION_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_device_description.h"

namespace xla {
class NpuDeviceDescription final : public PjRtDeviceDescription {
   public:
    explicit NpuDeviceDescription(int process_id, int local_device_id);

    ~NpuDeviceDescription() override = default;

    int id() const override { return id_.value(); }

    int process_index() const override { return process_index_; }

    int local_hardware_id() const { return local_hardware_id_; }

    absl::string_view device_kind() const override;

    absl::string_view DebugString() const override;

    absl::string_view ToString() const override;

    const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes() const override { return attributes_; }

   private:
    PjRtGlobalDeviceId id_;
    int process_index_;
    int local_hardware_id_;
    std::string debug_string_;
    std::string to_string_;
    absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_ = {};
};
}  // namespace xla
#endif  // XLA_XLA_NPU_PJRT_PLUGIN_NPU_DEVICE_DESCRIPTION_H_