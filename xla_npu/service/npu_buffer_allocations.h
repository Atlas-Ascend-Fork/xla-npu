#ifndef XLA_XLA_NPU_SERVICE_NPU_BUFFER_ALLOCATIONS_H_
#define XLA_XLA_NPU_SERVICE_NPU_BUFFER_ALLOCATIONS_H_

#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"

namespace xla {

// A thread-compatible class that encapsulates the base addresses of the
// allocated device buffers.
class NpuBufferAllocations {
   public:
    NpuBufferAllocations(absl::Span<se::DeviceMemoryBase const> buffers, int device_ordinal,
                         se::DeviceMemoryAllocator* memory_allocator)
        : buffers_(buffers.begin(), buffers.end()),
          device_ordinal_(device_ordinal),
          memory_allocator_(memory_allocator) {}

    NpuBufferAllocations(NpuBufferAllocations&& other) = default;
    NpuBufferAllocations& operator=(NpuBufferAllocations&& other) = default;
    NpuBufferAllocations(const NpuBufferAllocations&) = delete;
    NpuBufferAllocations& operator=(const NpuBufferAllocations&) = delete;

    se::DeviceMemoryAllocator* memory_allocator() const { return memory_allocator_; }
    int device_ordinal() const { return device_ordinal_; }

    // Returns the device address of buffer `buffer_index`. `buffer_index` must be
    // a valid index, i.e., in [0, buffer_count). This function returns null if
    // `buffer_index` is not assigned to a buffer address.
    se::DeviceMemoryBase GetDeviceAddress(BufferAllocation::Index buffer_index) const;

    // Returns a mutable value for the allocation at a given `buffer_index`.
    se::DeviceMemoryBase& GetMutableDeviceAddress(BufferAllocation::Index buffer_index);

    // Same as above, but also adjusts the returned address for the offset and
    // size contained in the given slice.
    se::DeviceMemoryBase GetDeviceAddress(const BufferAllocation::Slice& buffer_slice) const;

    // Tears down all buffers allocated by this object that are not in
    // `live_addresses`.
    absl::Status TearDown(const std::set<se::DeviceMemoryBase>& live_addresses,
                          absl::Span<const BufferAllocation* const> allocations);

    std::string ToString() const {
        std::string out;
        for (BufferAllocation::Index i = 0; i < buffers_.size(); ++i) {
            const auto& buf = buffers_[i];
            absl::StrAppendFormat(&out, "Buffer %d -> %p (%d B)", i, buf.opaque(), buf.size());
        }
        return out;
    }

    size_t size() const { return buffers_.size(); }

   private:
    // An array of device pointers that stores the address of each buffer
    // indexed by Index. Each element can point to a temporary buffer, an
    // input buffer, or nullptr if no buffer is needed for that Index.
    std::vector<se::DeviceMemoryBase> buffers_;
    int device_ordinal_;
    se::DeviceMemoryAllocator* memory_allocator_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_BUFFER_ALLOCATIONS_H_
