#include "xla/xla_npu/stream_executor/npu_platform.h"

#include <memory>
#include <string>
// #include <thread>  // NOLINT
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "tsl/platform/status.h"
#include "xla/stream_executor/device_description.h"
// #include "xla/stream_executor/host/host_executor.h"
// #include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/stream_executor/npu_executor.h"
#include "xla/xla_npu/stream_executor/npu_platform_id.h"

namespace stream_executor {

NpuPlatform::NpuPlatform() : name_(xla::NpuConstant::platformName) {}

NpuPlatform::~NpuPlatform() {}

Platform::Id NpuPlatform::id() const { return kNpuPlatformId; }

int NpuPlatform::VisibleDeviceCount() const { return 1; }

const std::string& NpuPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>> NpuPlatform::DescriptionForDevice(int ordinal) const {
    return NpuExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> NpuPlatform::ExecutorForDevice(int ordinal) {
    return executor_cache_.GetOrCreate(ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<std::unique_ptr<StreamExecutor>> NpuPlatform::GetUncachedExecutor(int ordinal) {
    auto executor = std::make_unique<NpuExecutor>(this, ordinal);
    auto init_status = executor->Init();
    if (!init_status.ok()) {
        return absl::InternalError(absl::StrFormat("failed initializing StreamExecutor for device ordinal %d: %s",
                                                   ordinal, init_status.ToString().c_str()));
    }

    return std::move(executor);
}

static void InitializeNpuPlatform() {
    std::unique_ptr<Platform> platform(new NpuPlatform);
    TF_CHECK_OK(PlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(npu_platform, stream_executor::InitializeNpuPlatform());
