/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_NPU_TRANSFER_MANAGER_H_
#define XLA_XLA_NPU_SERVICE_NPU_TRANSFER_MANAGER_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/service/generic_transfer_manager.h"
#include "xla/service/transfer_manager.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {

// An implementation of the XLA GenericTransferManager that
// handles NPU-specific infeed.
class NpuTransferManager : public GenericTransferManager {
   public:
    NpuTransferManager();
    ~NpuTransferManager() override {}

    absl::Status TransferLiteralToInfeed(stream_executor::StreamExecutor* executor,
                                         const LiteralSlice& literal) override;
    absl::Status TransferLiteralFromOutfeed(stream_executor::StreamExecutor* executor,
                                            MutableBorrowingLiteral literal) override;

    bool CanShapedBufferBeAccessedNow(stream_executor::StreamExecutor* executor,
                                      const ShapedBuffer& device_buffer) const override {
        return true;
    }

    bool CanBufferBeAccessedNow(stream_executor::StreamExecutor* executor,
                                const stream_executor::DeviceMemoryBase& device_buffer) const override {
        return true;
    }

    absl::Status ReadDynamicShapes(stream_executor::Stream* stream, const ShapedBuffer* device_buffer,
                                   Shape* device_shape) override;

   private:
    bool PackSubbyteTypes() const override { return true; }

    NpuTransferManager(const NpuTransferManager&) = delete;
    NpuTransferManager& operator=(const NpuTransferManager&) = delete;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_TRANSFER_MANAGER_H_
