/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_STREAM_EXECUTOR_NPU_PLATFORM_H_
#define XLA_XLA_NPU_STREAM_EXECUTOR_NPU_PLATFORM_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

// Npu platform plugin, registered as a singleton value via module initializer.
class NpuPlatform : public Platform {
   public:
    NpuPlatform();
    ~NpuPlatform() override;

    Platform::Id id() const override;

    // Returns true if the platform has been initialized.
    bool Initialized() const override;

    // Initializes the platform. The platform must be initialized before obtaining
    // StreamExecutor objects.
    absl::Status Initialize() override;

    // Device count is less clear-cut for NPUs than accelerators. This call
    // currently returns the number of thread units in the host.
    int VisibleDeviceCount() const override;

    const std::string& Name() const override;

    absl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(int ordinal) const override;

    absl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;

   private:
    // Returns a device constructed with ordinal without
    // looking in or storing to the Platform's executor cache.
    // Ownership IS transferred to the caller.
    absl::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(int ordinal);

    // This platform's name.
    std::string name_;

    // Cache of created StreamExecutors.
    ExecutorCache executor_cache_;

    NpuPlatform(const NpuPlatform&) = delete;
    void operator=(const NpuPlatform&) = delete;

    bool initialized_ = false;
};

}  // namespace stream_executor

#endif  // XLA_XLA_NPU_STREAM_EXECUTOR_NPU_PLATFORM_H_
