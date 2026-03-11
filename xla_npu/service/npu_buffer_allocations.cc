/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/service/npu_buffer_allocations.h"

#include <cstdint>
#include <set>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"

namespace xla {

absl::Status NpuBufferAllocations::TearDown(const std::set<se::DeviceMemoryBase>& live_addresses,
                                            absl::Span<const BufferAllocation* const> allocations) {
    // Deallocate temporary buffers, taking care to try to deallocate all of them
    // even if one of the deallocations fails.
    absl::Status status;
    const int64_t num_buffers = allocations.size();
    for (BufferAllocation::Index i = 0; i < num_buffers; ++i) {
        const BufferAllocation& allocation = *allocations[i];
        se::DeviceMemoryBase buffer_address = GetDeviceAddress(allocation.index());
        // Deallocate buffers marked "maybe_live_out" but aren't actually live out,
        // and temp buffers.
        if ((allocation.maybe_live_out() && !live_addresses.count(buffer_address)) ||
            allocation.IsPreallocatedTempBuffer()) {
            auto dealloc_result = memory_allocator_->Deallocate(device_ordinal_, buffer_address);
            if (!dealloc_result.ok() && status.ok()) {
                status = dealloc_result;
            }
        }
    }
    return status;
}

se::DeviceMemoryBase NpuBufferAllocations::GetDeviceAddress(BufferAllocation::Index buffer_index) const {
    CHECK_GE(buffer_index, 0);
    CHECK_LT(buffer_index, buffers_.size());
    return buffers_[buffer_index];
}

se::DeviceMemoryBase& NpuBufferAllocations::GetMutableDeviceAddress(BufferAllocation::Index buffer_index) {
    CHECK_GE(buffer_index, 0);
    CHECK_LT(buffer_index, buffers_.size());
    return buffers_[buffer_index];
}

se::DeviceMemoryBase NpuBufferAllocations::GetDeviceAddress(const BufferAllocation::Slice& buffer_slice) const {
    int64_t index = buffer_slice.index();
    se::DeviceMemoryBase base = GetDeviceAddress(index);

    int64_t offset = buffer_slice.offset();
    CHECK_LE(buffer_slice.offset(), base.size())
        << "slice offset " << offset << " must be smaller than buffer #" << index << " size " << base.size();

    int64_t extent = offset + buffer_slice.size();
    CHECK_LE(extent, base.size()) << "slice extent " << extent << " must be smaller than buffer #" << index << " size "
                                  << base.size();

    return base.GetByteSlice(buffer_slice.offset(), buffer_slice.size());
}

}  // namespace xla
