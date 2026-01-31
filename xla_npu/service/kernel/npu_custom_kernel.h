#ifndef SERVICE_KERNEL_NPU_CUSTOM_KERNEL_H_
#define SERVICE_KERNEL_NPU_CUSTOM_KERNEL_H_

#include <cstddef>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/xla_npu/service/kernel/npu_kernel_config.h"

// WARNING: This header (and a build target) should have minimal dependencies as
// it's included into all device kernel implementations, and we want to minimize
// the number of (very expensive!) recompilations.

namespace xla::gpu {
namespace se = ::stream_executor;  // NOLINT

// 此类用于生成执行NPU算子的相关参数
// Custom kernel is a mechanism for plugging pre-compiled device kernels into
// XLA GPU runtime. Custom kernel defines how to load the kernel on an executor
// and what are the grid size requirements for running it.
//
// We use this API to hide kernel implementation details from XLA (e.g. we can
// export CUTLASS gemm kernels as custom kernels to XLA), so that XLA can only
// implement generic interfaces, e.g. command buffer implementation for CUDA
// can automatically add a kernel node to a graph for arbitrary custom kernel.
//
// TODO(ezhulenev): Add custom kernel signature to track number and types of
// buffer arguments, and a way to mark one of an arguments as a workspace and
// define if it has to be zeroed first.
class NpuCustomKernel {
   public:
    NpuCustomKernel(NpuKernelConfig npu_kernel_config, std::string name, se::BlockDim block_dims,
                    size_t shared_memory_bytes);

    // NpuCustomKernel(std::string name, se::KernelLoaderSpec kernel_spec, se::BlockDim block_dims,
    //                 se::ThreadDim thread_dims, size_t shared_memory_bytes);

    // NpuCustomKernel(std::string name, se::KernelLoaderSpec kernel_spec, se::BlockDim block_dims,
    //                 se::ThreadDim thread_dims, se::ClusterDim cluster_dims, size_t shared_memory_bytes);

    absl::string_view name() const { return name_; }

    const se::KernelLoaderSpec& kernel_spec() { return kernel_spec_; }

    se::BlockDim block_dims() const { return block_dims_; }  // npu仅使用第一维,即x轴

    se::ThreadDim thread_dims() const { return thread_dims_; }  // npu不使用,仅作适配xla框架的占位符

    std::optional<se::ClusterDim> cluster_dims() const { return cluster_dims_; }  // npu不使用,仅作适配xla框架的占位符

    size_t shared_memory_bytes() const { return shared_memory_bytes_; }

    std::string ToString() const;

    NpuKernelConfig npu_kernel_config() const { return npu_kernel_config_; }

   private:
    std::string name_;
    se::KernelLoaderSpec kernel_spec_;
    se::BlockDim block_dims_;
    se::ThreadDim thread_dims_;
    std::optional<se::ClusterDim> cluster_dims_ = std::nullopt;
    size_t shared_memory_bytes_;
    // kernel_spec中的InProcessSymbol类型payload指向此数据结构,
    // 需要保证使用这个payload时,这个NpuCustomKernel变量生命周期还存在,因此需要保证所有使用NpuCustomKernel的地方都是右值传递
    // 此设计为了兼容xla框架通过kernel_spec_传递参数创建具体执行的kernel
    NpuKernelConfig npu_kernel_config_;
};

}  // namespace xla::gpu

#endif  // SERVICE_KERNEL_NPU_CUSTOM_KERNEL_H_
