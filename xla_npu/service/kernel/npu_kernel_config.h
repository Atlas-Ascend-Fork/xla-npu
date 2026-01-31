#ifndef XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_
#define XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_

namespace xla {

class NpuKernelConfig {
   public:
    enum class NpuKernelType {
        FUSION,
    };

    NpuKernelType npu_kernel_type;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_