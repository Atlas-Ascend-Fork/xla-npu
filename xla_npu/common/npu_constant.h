/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_COMMON_NPU_CONSTANT_H_
#define XLA_XLA_NPU_COMMON_NPU_CONSTANT_H_

#include <string>

#include "xla/codegen/emitters/kernel_arguments.h"

namespace xla {

// npu常量配置类
class NpuConstant {
   public:
    static const std::string platformName;

    // Minimum alignment for buffers passed as incoming arguments by TensorFlow.
    static constexpr int64_t kEntryParameterAlignBytes = 16;

    // Minimum alignment for buffers allocated by XLA: the temp buffers and the live
    // out (result) buffers.
    static constexpr int64_t kXlaAllocatedBufferAlignBytes = 256;

    // Minimum alignment for constant buffers.
    static constexpr int64_t kConstantBufferAlignBytes = kXlaAllocatedBufferAlignBytes;

    // Device memory alignment in bytes for aclrtMalloc allocations.
    static constexpr int64_t kDeviceMemoryAlignmentBytes = 512;

    // Extra bytes to allocate beyond the requested size for device memory.
    static constexpr int64_t kDeviceMemoryExtraBytes = 32;

    static emitters::KernelArguments::BufferAlignment GetDefaultBufferAlignment() {
        emitters::KernelArguments::BufferAlignment buffer_alignment;
        buffer_alignment.entry_parameter_align_bytes = kEntryParameterAlignBytes;
        buffer_alignment.xla_allocated_buffer_align_bytes = kXlaAllocatedBufferAlignBytes;
        buffer_alignment.constant_buffer_align_bytes = kConstantBufferAlignBytes;

        return buffer_alignment;
    }

    // 用于标记AFIR入口的Computation的key name, 在转换reduce运算时使用
    static const std::string entryComputationKeyName;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_COMMON_NPU_CONSTANT_H_