/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/pjrt/npu/npu_pjrt_stream_executor_client.h"

#include "tsl/platform/fingerprint.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/xla_npu/pjrt/plugin/npu_topology_description.h"

namespace xla {

// Returns the NPU devices from the given PjRtNpuDevices.
// Precondition: `devices` doesn't contain nullptr.
static std::vector<NpuTopology::NpuDevice> GetNpuDevices(
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices) {
    std::vector<NpuTopology::NpuDevice> npu_devices;
    npu_devices.reserve(devices.size());
    for (const auto& device : devices) {
        npu_devices.push_back(NpuTopology::NpuDevice{device->process_index(), device->local_hardware_id().value()});
    }
    return npu_devices;
}

NpuPjrtStreamExecutorClient::NpuPjrtStreamExecutorClient(
    std::string platform_name, LocalClient* client, std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
    int process_index, std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces,
    std::unique_ptr<se::DeviceMemoryAllocator> allocator, std::unique_ptr<tsl::Allocator> host_memory_allocator,
    bool should_stage_host_to_device_transfers, std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options)
    : PjRtStreamExecutorClient(platform_name, client, std::move(devices), process_index, std::move(memory_spaces),
                               std::move(allocator), std::move(host_memory_allocator),
                               should_stage_host_to_device_transfers, std::move(gpu_run_options)),
      topology_description_(tsl::Fingerprint64(platform_name), platform_name, "Default pjrt version",
                            GetNpuDevices(std::move(devices)), std::vector<std::string>{}) {}

}  // namespace xla
