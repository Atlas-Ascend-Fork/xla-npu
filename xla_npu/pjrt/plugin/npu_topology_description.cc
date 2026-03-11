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
// #include "xla/pjrt/plugin/xla_cpu/cpu_device_description.h"
// #include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
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
    // if (!tsl::SerializeToStringDeterministic(npu_topology_.ToProto(), &result)) {
    //     return absl::InternalError("Failed to serialize npu_topology");
    // }
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

    // CpuTopologyProto cpu_topology_proto = npu_topology_.ToProto();
    // proto.mutable_platform_specific_topology()->PackFrom(cpu_topology_proto);
    return proto;
}

// absl::StatusOr<std::unique_ptr<NpuTopologyDescription>>
// NpuTopologyDescription::FromProto(
//     const xla::PjRtTopologyDescriptionProto& proto) {
//     if (proto.platform_id() != xla::CpuId()) {
//         return absl::InvalidArgumentError(
//             absl::StrCat("The platform_id is not a CPU platform. platform_id: ",
//                         proto.platform_id()));
//     }

//     if (!proto.platform_specific_topology().Is<CpuTopologyProto>()) {
//         return absl::InvalidArgumentError(
//             "The platform_specific_topology is not a CpuTopologyProto.");
//     }
//     CpuTopologyProto cpu_topology_proto;
//     proto.platform_specific_topology().UnpackTo(&cpu_topology_proto);
//     auto cpu_topology = std::shared_ptr<const NpuTopology>(NpuTopology::FromProto(cpu_topology_proto));
//     std::vector<xla::NpuTopology::NpuDevice> npu_devices;
//     npu_devices.assign(cpu_topology->devices().begin(), cpu_topology->devices().end());
//     return std::make_unique<NpuTopologyDescription>(
//         proto.platform_id(), proto.platform_name(), proto.platform_version(),
//         npu_devices, cpu_topology->machine_attributes());
// }

}  // namespace xla