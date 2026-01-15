#ifndef BACKEND_NPU_BUFFER_ALLOCATION_H_
#define BACKEND_NPU_BUFFER_ALLOCATION_H_

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/util.h"

namespace xla {

// Buffer allocation is a container for device buffers allocated for a
// particular XLA execution. Buffers are indexed by the buffer allocation index.
class NpuBufferAllocations {
 public:
  using Buffers = absl::InlinedVector<se::DeviceMemoryBase, 8>;

  explicit NpuBufferAllocations(Buffers buffers);
  explicit NpuBufferAllocations(absl::Span<const se::DeviceMemoryBase> buffers);
  explicit NpuBufferAllocations(absl::Span<const MaybeOwningDeviceMemory> buffers);

  // Returns the device address of buffer at the given index. Returns an error
  // if the index is out of range.
  absl::StatusOr<se::DeviceMemoryBase> GetDeviceAddress(
      BufferAllocation::Index index) const;

  // Same as above, but also adjusts the returned address for the offset and
  // size contained in the given slice.
  absl::StatusOr<se::DeviceMemoryBase> GetDeviceAddress(
      BufferAllocation::Slice slice) const;

  // Unchecked version of `GetDeviceAddress` that does not check the buffer
  // index and assumes it is valid.
  se::DeviceMemoryBase GetDeviceAddressUnchecked(
      BufferAllocation::Index buffer_index) const;

  // Unchecked version of `GetDeviceAddress` that does not check the slice
  // buffer index, offset and size and assumes they all are valid.
  se::DeviceMemoryBase GetDeviceAddressUnchecked(
      BufferAllocation::Slice slice) const;

 private:
  absl::InlinedVector<se::DeviceMemoryBase, 8> buffers_;
  se::DeviceMemoryBase* buffers_data_;  // buffers_.data()
  size_t num_buffers_;
};

inline NpuBufferAllocations::NpuBufferAllocations(Buffers buffers)
    : buffers_(std::move(buffers)),
      buffers_data_(buffers_.data()),
      num_buffers_(buffers_.size()) {}

inline NpuBufferAllocations::NpuBufferAllocations(
    absl::Span<const se::DeviceMemoryBase> buffers)
    : buffers_(buffers.begin(), buffers.end()),
      buffers_data_(buffers_.data()),
      num_buffers_(buffers_.size()) {}

inline NpuBufferAllocations::NpuBufferAllocations(
    absl::Span<const MaybeOwningDeviceMemory> buffers)
    : buffers_(buffers.size()),
      buffers_data_(buffers_.data()),
      num_buffers_(buffers_.size()) {
  for (size_t i = 0; i < buffers.size(); ++i) {
    buffers_[i] = buffers[i].AsDeviceMemoryBase();
  }
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE absl::StatusOr<se::DeviceMemoryBase>
NpuBufferAllocations::GetDeviceAddress(BufferAllocation::Index index) const {
  if (ABSL_PREDICT_FALSE(index < 0 || index >= num_buffers_)) {
    return InvalidArgument(
        "Invalid buffer index %d. It must be in the range [0, %d)", index,
        num_buffers_);
  }

  return buffers_[index];
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE absl::StatusOr<se::DeviceMemoryBase>
NpuBufferAllocations::GetDeviceAddress(BufferAllocation::Slice slice) const {
  // Handle empty slices explicitly and return a null pointer device memory to
  // guarantee that we do not accidentally write through the empty slice which
  // would hide a real bug in the code.
  if (ABSL_PREDICT_FALSE(slice.size() == 0)) {
    return se::DeviceMemoryBase(nullptr, 0);
  }

  int64_t index = slice.index();
  if (ABSL_PREDICT_FALSE(index < 0 || index >= num_buffers_)) {
    return InvalidArgument(
        "Invalid buffer index %d. It must be in the range [0, %d)", index,
        num_buffers_);
  }
  const se::DeviceMemoryBase& base = buffers_data_[index];

  int64_t offset = slice.offset();
  int64_t extent = offset + slice.size();

  if (ABSL_PREDICT_FALSE(offset < 0)) {
    return InvalidArgument("Buffer slice offset %d must be non-negative",
                           offset);
  }

  if (ABSL_PREDICT_FALSE(offset >= base.size())) {
    return InvalidArgument(
        "Buffer slice offset %d is out of range for buffer #%d of size %d",
        offset, index, base.size());
  }

  if (ABSL_PREDICT_FALSE(extent > base.size())) {
    return InvalidArgument(
        "Buffer slice extent %d is out of range for buffer #%d of size %d",
        extent, index, base.size());
  }

  return base.GetByteSlice(offset, slice.size());
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE se::DeviceMemoryBase
NpuBufferAllocations::GetDeviceAddressUnchecked(
    BufferAllocation::Index buffer_index) const {
  return buffers_data_[buffer_index];
}

// Unchecked version of `GetDeviceAddress` that does not check the slice
// buffer index, offset and size and assumes they are valid.
inline ABSL_ATTRIBUTE_ALWAYS_INLINE se::DeviceMemoryBase
NpuBufferAllocations::GetDeviceAddressUnchecked(
    BufferAllocation::Slice slice) const {
  return buffers_data_[slice.index()].GetByteSlice(slice.offset(),
                                                   slice.size());
}

}  // namespace xla

#endif  // BACKEND_NPU_BUFFER_ALLOCATION_H_
