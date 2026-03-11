/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/service/buffer_assignment.h"

#include "xla/xla_npu/pjrt/air/npu/service/npu_buffer_assignment.h"

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