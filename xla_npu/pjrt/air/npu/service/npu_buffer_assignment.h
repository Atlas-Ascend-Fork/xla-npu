/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_SERVICE_NPU_BUFFER_ASSIGNMENT_H_
#define NPU_SERVICE_NPU_BUFFER_ASSIGNMENT_H_

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

    BufferAllocation* NewEmptyAllocation(
        int64_t size, LogicalBuffer::Color color);

    void Finalize();

private:
    // The vector of buffer allocations. Indexed by BufferAllocation::Index.
    std::vector<BufferAllocation> allocations_;
};

}  // namespace xla

#endif  // NPU_SERVICE_NPU_BUFFER_ASSIGNMENT_H_