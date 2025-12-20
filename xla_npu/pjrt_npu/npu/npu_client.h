#ifndef NPU_CLIENT_H_
#define NPU_CLIENT_H_

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
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "tsl/platform/fingerprint.h"

#include "xla/xla_npu/pjrt_npu/pjrt_common.h"
#include "xla/xla_npu/pjrt_npu/npu/npu_device.h"
#include "xla/xla_npu/pjrt_npu/plugin/npu_topology_description.h"
#include "xla/xla_npu/pjrt_npu/plugin/npu_device_description.h"
#include "xla/xla_npu/third_party/register/aclInterface.h"

namespace xla {

inline const char* DataLayout() {
    static constexpr char kDataLayout[] =
        "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64";
    return kDataLayout;
}

class PjRtNpuClient final : public CommonPjRtClient {
public:
    PjRtNpuClient(int process_index, std::vector<std::unique_ptr<PjRtNpuDevice>> devices);

    ~PjRtNpuClient() override = default;

    int process_index() const override
    {
        return process_index_;
    }

    int device_count() const override
    {
        npu::AclGetErrMsg(); // only for debug
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
    
    absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
        const XlaComputation& computation, CompileOptions options) override;

    absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
        mlir::ModuleOp module, CompileOptions options) override;


    // Computes the memory requirements for storing shape on memory_space.
    absl::StatusOr<int64_t> GetOnDeviceBytesCount(
      PjRtMemorySpace* memory_space, const xla::Shape& shape) const override;

    // Allocates a raw buffer of a particular size after an optional
    // allocate_after. Backends may support retrying allocation on oom which
    // can be controlled via retry_on_oom.
    absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
      AllocateRawBuffer(PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
                  bool retry_on_oom,
                  tsl::AsyncValueRef<bool> allocate_after) override;

    absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> LinearizeHostBufferInto(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      const xla::Shape& device_shape,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) override;

    // Defines a pjrt buffer from a shape, raw_buffer and definition events.
    absl::StatusOr<std::unique_ptr<PjRtBuffer>> DefineBuffer(
      const Shape& on_device_shape,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
      absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>
          definition_device_events,
      bool raw_buffer_is_mutable) override;

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
    // A cache for transpose plans. We use transposes to convert
    // (possibly strided) buffers provided to BufferFromHostBuffer into dense
    // major-to-minor layout.
    absl::Mutex transpose_mu_;
    TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);
};

inline absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtNpuClient(int process_index, std::vector<std::unique_ptr<PjRtNpuDevice>> devices)
{
  return std::unique_ptr<PjRtClient>(new PjRtNpuClient(process_index, std::move(devices)));
}

}  // namespace xla
#endif  //NPU_CLIENT_H_
