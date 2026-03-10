/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/service/npu_transfer_manager.h"

#include <memory>

#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiler.h"
// #include "xla/service/cpu/cpu_runtime.h"
// #include "xla/service/cpu/cpu_xfeed.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
// #include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_npu/stream_executor/npu_platform_id.h"

namespace xla {

NpuTransferManager::NpuTransferManager()
    : GenericTransferManager(stream_executor::kNpuPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

absl::Status NpuTransferManager::TransferLiteralToInfeed(stream_executor::StreamExecutor* executor,
                                                         const LiteralSlice& literal) {
    // return TransferLiteralToInfeedOnCpu(executor->device_ordinal(), literal);
    return absl::UnimplementedError("NpuTransferManager::TransferLiteralToInfeed is not supported.");
}

absl::Status NpuTransferManager::TransferLiteralFromOutfeed(stream_executor::StreamExecutor* executor,
                                                            MutableBorrowingLiteral literal) {
    // return TransferLiteralFromOutfeedOnCpu(executor->device_ordinal(), literal);
    return absl::UnimplementedError("NpuTransferManager::TransferLiteralFromOutfeed is not supported.");
}

absl::Status NpuTransferManager::ReadDynamicShapes(stream_executor::Stream* stream, const ShapedBuffer* device_buffer,
                                                   Shape* device_shape) {
    // if (stream != nullptr) {
    //     // When a stream is presented, respect the stream dependency.
    //     return TransferManager::ReadDynamicShapes(stream, device_buffer, device_shape);
    // }
    // TF_ASSIGN_OR_RETURN(auto platform, stream_executor::PlatformManager::PlatformWithId(PlatformId()));
    // TF_ASSIGN_OR_RETURN(auto compiler, Compiler::GetForPlatform(platform));
    // return ReadDynamicShapesOnCpu(device_buffer, device_shape, compiler->ShapeSizeBytesFunction());
    return absl::UnimplementedError("NpuTransferManager::ReadDynamicShapes is not supported.");
}

}  // namespace xla
