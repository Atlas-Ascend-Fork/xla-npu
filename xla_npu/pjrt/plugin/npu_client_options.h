// Not used
/*
#ifndef PJRT_PLUGIN_NPU_CLIENT_OPTIONS_H_
#define PJRT_PLUGIN_NPU_CLIENT_OPTIONS_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
// #include "xla/backends/cpu/collectives/cpu_collectives.h"
// #include "xla/pjrt/plugin/xla_cpu/cpu_memory.h"
#include "xla/service/hlo_module_config.h"

namespace xla {

// Options for creating an PjRtNpuClient.
struct NpuClientOptions {
    // Used to control whether asynchronous computation dispatch is available for
    // this client. Only applies to non-parallel computations, because collectives
    // may exist when there are multiple npu devices and we need to do async
    // dispatch in that case. If it is set to be `false`, we will always run
    // computations inline.
    bool asynchronous = false;

    // Number of NPU devices.
    int npu_device_count = 1;

    // int max_inflight_computations_per_device = 32;

    // My process ID.
    int process_id = 0;

    // // Distributed collectives implementation. Optional. If not provided, an
    // // in-process collectives implementation will be used.
    // std::shared_ptr<cpu::CpuCollectives> collectives;

    // If defined this function will be called on the HloModuleConfig before
    // compilation, and allows users to set custom flags.
    std::function<void(HloModuleConfig&)> customize_hlo_module_config;

    // // If defined this function will be called by PjRtClient to allocate memory
    // // for constructed PjRtBuffers and a temporary allocation passed to XLA:CPU
    // // executable.
    // std::function<absl::StatusOr<std::unique_ptr<CpuMemory>>(size_t size_bytes, size_t alignment)> allocator;
};

}  // namespace xla

#endif  // PJRT_PLUGIN_NPU_CLIENT_OPTIONS_H_
*/
