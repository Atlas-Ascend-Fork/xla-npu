/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/pjrt/plugin/npu_topology_description.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/xla_npu/pjrt/plugin/npu_device_description.h"
#include "xla/xla_npu/pjrt/plugin/npu_topology.h"

namespace xla {

absl::StatusOr<Layout> NpuTopologyDescription::GetDefaultLayout(PrimitiveType element_type,
                                                                absl::Span<const int64_t> dims) const {
    Shape shape = ShapeUtil::MakeShape(element_type, dims);
    return LayoutUtil::GetWithDefaultLayout(shape).layout();
}

absl::StatusOr<std::string> NpuTopologyDescription::Serialize() const {
    std::string result;
    return result;
}

absl::StatusOr<std::pair<PjRtDeviceDimensions, int32_t>> NpuTopologyDescription::LogicalDeviceOfDefaultTypeForId(
    xla::PjRtGlobalDeviceId device_id) const {
    return std::make_pair(PjRtDeviceDimensions{0, 0, device_id.value()}, 0);
}

std::vector<std::unique_ptr<const PjRtDeviceDescription>> NpuTopologyDescription::DeviceDescriptions() const {
    std::vector<std::unique_ptr<const PjRtDeviceDescription>> devices;
    devices.reserve(npu_topology_.number_of_devices());
    for (const NpuTopology::NpuDevice& device : npu_topology_.devices()) {
        devices.push_back(std::make_unique<NpuDeviceDescription>(device.process_id, device.local_device_id));
    }
    return devices;
}

absl::StatusOr<xla::PjRtTopologyDescriptionProto> NpuTopologyDescription::ToProto() const {
    PjRtTopologyDescriptionProto proto;
    proto.set_platform_id(platform_id());
    proto.set_platform_name(platform_name());
    proto.set_platform_version(platform_version());
    proto.set_is_subslice_topology(is_subslice_topology());

    return proto;
}

}  // namespace xla