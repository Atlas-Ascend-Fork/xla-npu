#ifndef SERVICE_NPU_BUFFER_ASSIGNMENT_H_
#define SERVICE_NPU_BUFFER_ASSIGNMENT_H_

#include <vector>
#include "xla/service/buffer_assignment.h"

namespace xla {

class NpuBufferAssignment {
public:
    // Returns the vector containing all buffer allocations in this assignment.
    const std::vector<BufferAllocation>& Allocations() const {
        return allocations_;
    }

    const BufferAllocation& GetAllocation(BufferAllocation::Index index) const;

    BufferAllocation* NewEmptyAllocation(int64_t size, LogicalBuffer::Color color);

    void Finalize();

private:
    // The vector of buffer allocations. Indexed by BufferAllocation::Index.
    std::vector<BufferAllocation> allocations_;
};

}  // namespace xla

#endif  // SERVICE_NPU_BUFFER_ASSIGNMENT_H_