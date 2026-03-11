/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_PJRT_NPU_NPU_PJRT_STREAM_EXECUTOR_CLIENT_H_
#define XLA_XLA_NPU_PJRT_NPU_NPU_PJRT_STREAM_EXECUTOR_CLIENT_H_

#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/xla_npu/pjrt/plugin/npu_topology.h"
#include "xla/xla_npu/pjrt/plugin/npu_topology_description.h"

namespace xla {

class NpuPjrtStreamExecutorClient : public PjRtStreamExecutorClient {
   public:
    NpuPjrtStreamExecutorClient(std::string platform_name, LocalClient* client,
                                std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices, int process_index,
                                std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces,
                                std::unique_ptr<se::DeviceMemoryAllocator> allocator,
                                std::unique_ptr<tsl::Allocator> host_memory_allocator,
                                bool should_stage_host_to_device_transfers,
                                std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options);

    ~NpuPjrtStreamExecutorClient() override = default;

    // Gets the pointer to the topology description held by the client.
    absl::StatusOr<const PjRtTopologyDescription*> GetTopologyDescription() const override {
        return &topology_description_;
    }

   private:
    // 只是用于为dump提供上下文信息,在PjRtStreamExecutorClient::Compile开头使用,没有其它用处
    NpuTopologyDescription topology_description_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_PJRT_NPU_NPU_PJRT_STREAM_EXECUTOR_CLIENT_H_
