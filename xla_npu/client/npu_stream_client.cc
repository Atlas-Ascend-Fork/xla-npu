#include "xla/xla_npu/client/npu_stream_client.h"

#include <set>

#include "absl/status/statusor.h"
#include "xla/client/client_library.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/client/npu_module.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/service/npu_compiler.h"
#include "xla/xla_npu/stream_executor/npu_platform.h"
#include "xla/xla_npu/stream_executor/npu_platform_id.h"

namespace xla {

absl::StatusOr<std::unique_ptr<PjRtStreamExecutorClient>> CreateNpuPjrtStreamExecutorClient() {
    constexpr int used_device_id = 0;

    InitNpuSubModule();

    // 获取platform时默认会触发调用platform的Initialize
    TF_ASSIGN_OR_RETURN(stream_executor::Platform* const platform,
                        PlatformUtil::GetPlatform(NpuConstant::platformName));

    LocalClientOptions local_client_options;
    local_client_options.set_platform(platform);
    local_client_options.set_number_of_replicas(1);
    local_client_options.set_allowed_devices(std::set<int>{used_device_id});
    TF_ASSIGN_OR_RETURN(LocalClient* const local_client, ClientLibrary::GetOrCreateLocalClient(local_client_options));

    TF_ASSIGN_OR_RETURN(stream_executor::StreamExecutor * executor, platform->ExecutorForDevice(used_device_id));
    auto device_state = std::make_unique<LocalDeviceState>(executor, local_client, LocalDeviceState::kSynchronous,
                                                           /*max_inflight_computations=*/32,
                                                           /*allow_event_reuse=*/false, /*use_callback_stream=*/false);
    int local_device_id = device_state->local_device_id().value();
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
    devices.emplace_back(std::make_unique<PjRtStreamExecutorDevice>(
        0, std::move(device_state), local_device_id, /*process_index=*/0,
        /*process_index_in_partition=*/0, /*partition_index=*/0, NpuConstant::platformName));
    std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces;
    memory_spaces.emplace_back(
        std::make_unique<PjRtStreamExecutorMemorySpace>(0, devices.back().get(), NpuConstant::platformName, 0));
    devices.back()->AttachMemorySpace(memory_spaces.back().get(),
                                      /*is_default=*/true);
    return std::make_unique<PjRtStreamExecutorClient>(
        NpuConstant::platformName, local_client, std::move(devices),
        /*process_index=*/0, std::move(memory_spaces),
        /*allocator=*/nullptr,              // 不传递则会使用local_client的backend().memory_allocator(),
                                            // 其由backend构建时自动创建(类型:StreamExecutorMemoryAllocator)
                                            // StreamExecutorMemoryAllocator中的内存分配和释放都是调用的executor的接口
        /*host_memory_allocator=*/nullptr,  // 不传递则会使用默认的CpuAllocator
        /*should_stage_host_to_device_transfers=*/false,
        /*gpu_run_options=*/nullptr);
}

}  // namespace xla