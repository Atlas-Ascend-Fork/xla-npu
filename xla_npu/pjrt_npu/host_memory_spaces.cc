#include "tsl/platform/fingerprint.h"

#include "host_memory_spaces.h"

namespace xla {

NpuDeviceMemorySpace::NpuDeviceMemorySpace(int id, PjRtDevice* device) : id_(id), device_(device)
{
    DCHECK(device_ != nullptr && device_->client() != nullptr);
    auto* client = device_->client();
    debug_string_ = absl::StrFormat("NpuDeviceMemory(id=%i, process_index=%i, client=%s)", id_, client->process_index(),
                                    client->platform_name());
    to_string_ = absl::StrFormat("NPU_DEVICE_%i", id_);
}

const int NpuDeviceMemorySpace::kKindId = []() {
    uint32_t kind_id = tsl::Fingerprint32(NpuDeviceMemorySpace::kKind);
    return static_cast<int>(kind_id);
}();

}  // namespace xla