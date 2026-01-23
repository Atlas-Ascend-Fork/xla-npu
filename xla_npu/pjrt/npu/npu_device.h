// Not used
/*
#ifndef PJRT_NPU_NPU_DEVICE_H_
#define PJRT_NPU_NPU_DEVICE_H_

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
#include "xla/xla_npu/pjrt/plugin/npu_device_description.h"

namespace xla {
class PjRtNpuDevice final : public PjRtDevice {
   public:
    explicit PjRtNpuDevice(int process_id, int local_device_id);

    ~PjRtNpuDevice() override = default;

    const NpuDeviceDescription& description() const override { return description_; }

    void SetClient(PjRtClient* client) {
        CHECK(client_ == nullptr);
        client_ = client;
    }

    PjRtClient* client() const override { return client_; }

    bool IsAddressable() const override { return process_index() == client()->process_index(); }

    PjRtLocalDeviceId local_device_id() const override { return PjRtLocalDeviceId(local_hardware_id().value()); }

    PjRtLocalHardwareId local_hardware_id() const override {
        return PjRtLocalHardwareId(description_.local_hardware_id());
    }

    std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(absl::string_view description) const override {
        return nullptr;
    }

    absl::Status TransferToInfeed(const LiteralSlice& literal) override {
        return absl::UnimplementedError("TransferToInfeed is unimplemented");
    }

    absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
        return absl::UnimplementedError("TransferFromOutfeed is unimplemented");
    }

    void AttachMemorySpace(PjRtMemorySpace* memory_space);

    absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

    absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

    absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(absl::string_view memory_space_kind) const override;

    absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind_id(int id) const;

    absl::StatusOr<bool> PoisonExecution(int32_t launch_id, absl::Status error) override;

   private:
    PjRtClient* client_ = nullptr;
    NpuDeviceDescription description_;
    absl::InlinedVector<PjRtMemorySpace*, 1> memory_spaces_;
    absl::flat_hash_map<int, PjRtMemorySpace*> memory_spaces_by_id_;
};

}  // namespace xla
#endif  // PJRT_NPU_NPU_DEVICE_H_
*/