#ifndef COMMON_NPU_CONSTANT_H_
#define COMMON_NPU_CONSTANT_H_

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

    static emitters::KernelArguments::BufferAlignment GetDefaultBufferAlignment() {
        emitters::KernelArguments::BufferAlignment buffer_alignment;
        buffer_alignment.entry_parameter_align_bytes = kEntryParameterAlignBytes;
        buffer_alignment.xla_allocated_buffer_align_bytes = kXlaAllocatedBufferAlignBytes;
        buffer_alignment.constant_buffer_align_bytes = kConstantBufferAlignBytes;

        return buffer_alignment;
    }
};

}  // namespace xla

#endif  // COMMON_NPU_CONSTANT_H_