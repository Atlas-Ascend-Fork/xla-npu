/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "xla/xla_npu/stream_executor/npu_platform.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "tsl/platform/status.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/stream_executor/npu_executor.h"
#include "xla/xla_npu/stream_executor/npu_platform_id.h"

namespace stream_executor {

NpuPlatform::NpuPlatform() : name_(xla::NpuConstant::platformName) {}

NpuPlatform::~NpuPlatform() {
  NPU_LOG_FORMAT_PRINT(INFO, "Finalize platform for '%s'.", name_);
  if (initialized_) {
    auto ret = npu::aclFinalize();
    if (ret != ACL_SUCCESS) {
      NPU_LOG_FORMAT_PRINT(ERROR,
                           "Finalize platform for '%s' failed, error code %d",
                           name_, ret);
    } else {
      NPU_LOG_FORMAT_PRINT(INFO, "Finalize platform for '%s' success.", name_);
    }

    initialized_ = false;
  }
}

Platform::Id NpuPlatform::id() const { return kNpuPlatformId; }

bool NpuPlatform::Initialized() const { return initialized_; }

absl::Status NpuPlatform::Initialize() {
  const char* acl_config_json_file_path = nullptr;
  aclError acl_ret = npu::aclInit(acl_config_json_file_path);
  if (acl_ret != ACL_SUCCESS) {
    NPU_LOG_FORMAT_PRINT(ERROR, "Init platform for '%s' failed.", name_);
    return absl::UnavailableError("Acl init failed.");
  }

  initialized_ = true;

  NPU_LOG_FORMAT_PRINT(INFO, "Init platform for '%s' success.", name_);
  return absl::OkStatus();
}

int NpuPlatform::VisibleDeviceCount() const {
  uint32_t device_count = 0;
  aclError acl_ret = npu::aclrtGetDeviceCount(&device_count);
  if (acl_ret != ACL_SUCCESS) {
    NPU_LOG_FORMAT_PRINT(ERROR, "Get device count for '%s' failed.", name_);
    return 0;
  }

  NPU_LOG_FORMAT_PRINT(
      INFO, "Get device count for '%s' success, total device num is %d.", name_,
      device_count);
  return device_count;
}

const std::string& NpuPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
NpuPlatform::DescriptionForDevice(int ordinal) const {
  return NpuExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> NpuPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
NpuPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<NpuExecutor>(this, ordinal);
  auto init_status = executor->Init();
  if (!init_status.ok()) {
    return absl::InternalError(absl::StrFormat(
        "failed initializing StreamExecutor for device ordinal %d: %s", ordinal,
        init_status.ToString().c_str()));
  }

  return std::move(executor);
}

}  // namespace stream_executor
