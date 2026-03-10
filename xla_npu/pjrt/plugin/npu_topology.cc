/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/pjrt/plugin/npu_topology.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// #include "xla/pjrt/plugin/xla_cpu/cpu_topology.pb.h"

namespace xla {

// std::unique_ptr<const NpuTopology> NpuTopology::FromProto(
//     const CpuTopologyProto& cpu_topology_proto) {
//   std::vector<NpuTopology::CpuDevice> devices;
//   devices.reserve(cpu_topology_proto.cpu_devices_size());

//   for (size_t i = 0; i < cpu_topology_proto.cpu_devices_size(); ++i) {
//     auto& cpu_device_proto = cpu_topology_proto.cpu_devices(i);
//     devices.push_back(CpuDevice{cpu_device_proto.process_index(),
//                                 cpu_device_proto.local_hardware_id()});
//   }

//   std::vector<std::string> machine_attributes;
//   machine_attributes.reserve(cpu_topology_proto.machine_attributes_size());
//   for (size_t i = 0; i < cpu_topology_proto.machine_attributes_size(); ++i) {
//     machine_attributes.push_back(cpu_topology_proto.machine_attributes(i));
//   }

//   return std::make_unique<NpuTopology>(std::move(devices),
//                                        std::move(machine_attributes));
// }

// CpuTopologyProto NpuTopology::ToProto() const {
//   CpuTopologyProto proto;
//   for (auto& cpu_device : cpu_devices_) {
//     auto* cpu_device_proto = proto.add_cpu_devices();
//     cpu_device_proto->set_process_index(cpu_device.process_id);
//     cpu_device_proto->set_local_hardware_id(cpu_device.local_device_id);
//   }
//   for (const std::string& machine_attribute : machine_attributes_) {
//     proto.add_machine_attributes(machine_attribute);
//   }
//   return proto;
// }

}  // namespace xla
