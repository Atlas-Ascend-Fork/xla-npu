#ifndef NPU_CLIENT_H
#define NPU_CLIENT_H
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
// #include "xla/pjrt/cpu/cpu_async_execution_tracker.h"
// #include "xla/pjrt/cpu/execution_stream_event_map.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/semaphore.h"
#include "tsl/platform/fingerprint.h"

#include "xla/xla_npu/pjrt_npu/pjrt_common.h"
#include "xla/xla_npu/pjrt_npu/npu/npu_device.h"
#include "xla/xla_npu/pjrt_npu/plugin/npu_topology_description.h"
#include "xla/xla_npu/pjrt_npu/plugin/npu_device_description.h"

namespace xla {

class PjRtNpuClient final : public PjRtClient {
public:
    PjRtNpuClient(int process_index, std::vector<std::unique_ptr<PjRtNpuDevice>> devices);

    ~PjRtNpuClient() override = default;

    int process_index() const override
    {
        return process_index_;
    }

    int device_count() const override
    {
        return devices_.size();
    }

    int addressable_device_count() const override
    {
        return addressable_devices_.size();
    }
    absl::Span<PjRtDevice* const> devices() const override
    {
        return devices_;
    }

    absl::Span<PjRtDevice* const> addressable_devices() const override
    {
        return addressable_devices_;
    }
    absl::StatusOr<PjRtDevice*> LookupDevice(PjRtGlobalDeviceId global_device_id) const override;

    absl::StatusOr<PjRtDevice*> LookupAddressableDevice(PjRtLocalDeviceId local_device_id) const override;

    absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

    PjRtPlatformId platform_id() const override
    {
        return tsl::Fingerprint64(NpuName());
    }

    absl::string_view platform_name() const override
    {
        return NpuName();
    }

    absl::string_view platform_version() const override
    {
        return NpuName();
    }

    // Gets the pointer to the topology description held by the client.
    absl::StatusOr<const PjRtTopologyDescription*> GetTopologyDescription() const override
    {
        return &topology_;
    }

private:
    friend absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtNpuClient(int process_index, std::vector<std::unique_ptr<PjRtNpuDevice>> devices);
    int process_index_;
    // Includes all devices, including non-addressable devices.
    std::vector<std::unique_ptr<PjRtNpuDevice>> owned_devices_;
    // Pointers to `owned_devices_`.
    std::vector<PjRtDevice*> devices_;
    // Maps Device::id() to the corresponding Device. Includes all devices.
    absl::flat_hash_map<PjRtGlobalDeviceId, PjRtNpuDevice*> id_to_device_;
    // Addressable devices indexed by core_id.
    std::vector<PjRtDevice*> addressable_devices_;
    std::unique_ptr<ComputationPlacer> computation_placer_;

    // Addressable memory spaces.
    std::vector<std::unique_ptr<PjRtMemorySpace>> owned_memory_spaces_;
    // Pointers to `owned_memory_spaces_`.
    std::vector<PjRtMemorySpace*> memory_spaces_;
    NpuTopologyDescription topology_;
};

inline absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtNpuClient(int process_index, std::vector<std::unique_ptr<PjRtNpuDevice>> devices)
{
  return std::unique_ptr<PjRtClient>(new PjRtNpuClient(process_index, std::move(devices)));
}
}  // namespace xla
#endif  //NPU_CLIENT_H
