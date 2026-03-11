/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/backend/runtime/npu_custom_kernel_thunk.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
// #include "xla/backends/gpu/runtime/print_buffer_contents.h"
// #include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
// #include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/kernel/npu_custom_kernel.h"

namespace xla {

NpuCustomKernelThunk::NpuCustomKernelThunk(NpuThunk::ThunkInfo thunk_info, NpuCustomKernel custom_kernel,
                                           const emitters::KernelArguments& kernel_arguments)
    : NpuThunk(Kind::kCustomKernel, std::move(thunk_info)),
      args_(kernel_arguments.GetArgumentBufferSlices()),
      written_(kernel_arguments.GetArgumentOutputFlags()),
      custom_kernel_(std::move(custom_kernel)) {}

std::string NpuCustomKernelThunk::ToString(int indent) const { return custom_kernel_.ToString(); }

absl::Status NpuCustomKernelThunk::Initialize(const InitializeParams& params) {
    absl::MutexLock lock(mutex_);

    if (!kernel_cache_.contains(params.executor)) {
        TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                            params.executor->LoadKernel(custom_kernel_.kernel_spec()));
        kernel_cache_.emplace(params.executor, std::move(kernel));
    }

    return absl::OkStatus();
}

absl::Status NpuCustomKernelThunk::ExecuteOnStream(const ExecuteParams& params) {
    if (params.stream == nullptr) {
        const std::string& errMsg = absl::StrFormat("Xla stream need to be created or init before execute npu thunk.");
        NPU_LOG_FORMAT_PRINT(ERROR, errMsg);
        return absl::InternalError(errMsg);
    }

    se::StreamExecutor* executor = params.stream->parent();

    se::Kernel* kernel = [&] {
        absl::MutexLock lock(mutex_);
        return kernel_cache_[executor].get();
    }();

    int device_ordinal = executor->device_ordinal();

    NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
    NPU_LOG_FORMAT_PRINT(DEBUG, "Device id: %d, launching %s as npu fusion kernel %s", device_ordinal,
                         custom_kernel_.ToString(), kernel->name());

    int id = 0;
    absl::InlinedVector<se::DeviceMemoryBase, 4> buffer_args;
    for (const BufferAllocation::Slice& arg : args_) {
        se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
        buffer_args.push_back(buf);
        NPU_LOG_FORMAT_PRINT(DEBUG, "---- Param %d info before execute: ----", id);
        NPU_LOG_FORMAT_PRINT(DEBUG, "Device id: %d, Arg: alloc #%d, offset: %d: %p (%dB)", device_ordinal, arg.index(),
                             arg.offset(), buf.opaque(), buf.size());
        NpuTool::PrintNpuDeviceMemoryValueByShape("  Value on device:", thunk_info().param_shapes[id], buf.opaque());

        id++;
    }

    // if (VLOG_IS_ON(100)) {
    //     absl::InlinedVector<se::KernelArgument, 4> kernel_args;
    //     for (const se::DeviceMemoryBase& arg : buffer_args) {
    //         kernel_args.push_back(arg);
    //     }
    //     PrintBufferContents(params.stream, kernel_args);
    // }

    se::KernelArgsDeviceMemoryArray args(buffer_args, custom_kernel_.shared_memory_bytes());

    auto ret = kernel->Launch(custom_kernel_.thread_dims(), custom_kernel_.block_dims(), custom_kernel_.cluster_dims(),
                              params.stream, args);

    id = 0;
    for (const BufferAllocation::Slice& arg : args_) {
        se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
        NPU_LOG_FORMAT_PRINT(DEBUG, "---- Param %d info after execute: ----", id);
        NPU_LOG_FORMAT_PRINT(DEBUG, "Device id: %d, Arg: alloc #%d, offset: %d: %p (%dB)", device_ordinal, arg.index(),
                             arg.offset(), buf.opaque(), buf.size());
        NpuTool::PrintNpuDeviceMemoryValueByShape("  Value on device:", thunk_info().param_shapes[id], buf.opaque());

        id++;
    }

    return ret;
}

NpuThunk::BufferUses NpuCustomKernelThunk::buffer_uses() const {
    NpuThunk::BufferUses buffers;
    buffers.reserve(args_.size());
    for (int i = 0; i < args_.size(); ++i) {
        // We assume that any buffer is either an input or an output of the
        // kernel, and inout buffers are represented as 2 separate arguments.
        if (written_[i]) {
            buffers.push_back(BufferUse::Write(args_[i]));
        } else {
            buffers.push_back(BufferUse::Read(args_[i]));
        }
    }
    return buffers;
}

}  // namespace xla
