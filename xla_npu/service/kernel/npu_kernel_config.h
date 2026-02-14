#ifndef XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_
#define XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_

#include <string>

#include "xla/xla_npu/service/kernel/npu_compile_result.h"

namespace xla {

class NpuKernelConfig {
   public:
    enum class NpuKernelType {
        FUSION = 0,
    };

    NpuKernelType npu_kernel_type;
    NpuArtifactInfo npu_artifact_info;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_