#ifndef KERNEL_NPU_KERNEL_CONFIG_H_
#define KERNEL_NPU_KERNEL_CONFIG_H_

#include <string>
#include "xla/codegen/emitters/kernel_arguments.h"

namespace xla {

class NpuKernelConfig {
   public:
    enum class NpuKernelType {
        FUSION,
    };

    NpuKernelType npu_kernel_type;
};

}  // namespace xla

#endif  // KERNEL_NPU_KERNEL_CONFIG_H_