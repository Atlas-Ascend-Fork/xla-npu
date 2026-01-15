#ifndef BACKEND_NPU_CONSTANT_ALLOCATION_H_
#define BACKEND_NPU_CONSTANT_ALLOCATION_H_

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"

namespace xla {

// A storage (or an alias) for constant allocations data.
struct NpuConstantAllocation {
  se::DeviceMemoryBase AsDeviceMemoryBase() const;

  BufferAllocation::Index index = -1;
  std::variant<std::monostate, std::unique_ptr<Literal>, absl::Span<const uint8_t>> data;
};

// Creates a vector of constant allocations from the given buffer assignment.
absl::StatusOr<std::vector<NpuConstantAllocation>> CreateNpuConstantAllocations(const BufferAssignment& assignment);

}  // namespace xla

#endif  // BACKEND_NPU_CONSTANT_ALLOCATION_H_
