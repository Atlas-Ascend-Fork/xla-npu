#include "xla/xla_npu/pjrt_npu/service/npu_buffer_assignment.h"

#include "xla/service/buffer_assignment.h"

namespace xla {

const BufferAllocation& NpuBufferAssignment::GetAllocation(BufferAllocation::Index index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, allocations_.size());
    return allocations_[index];
}

BufferAllocation* NpuBufferAssignment::NewEmptyAllocation(
    int64_t size, LogicalBuffer::Color color) {
    BufferAllocation::Index index = allocations_.size();
    allocations_.emplace_back(index, size, color);
    BufferAllocation* allocation = &allocations_.back();
    return allocation;
}

void NpuBufferAssignment::Finalize() {
    // TODO
}

}  // namespace xla