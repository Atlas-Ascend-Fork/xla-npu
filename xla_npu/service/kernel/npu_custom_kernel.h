/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_KERNEL_NPU_CUSTOM_KERNEL_H_
#define XLA_XLA_NPU_SERVICE_KERNEL_NPU_CUSTOM_KERNEL_H_

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

namespace xla {
namespace se = ::stream_executor;  // NOLINT

// 此类用于生成执行NPU算子的相关参数
// Custom kernel is a mechanism for plugging pre-compiled device kernels into
// XLA GPU runtime. Custom kernel defines how to load the kernel on an executor
// and what are the grid size requirements for running it.
// We use this API to hide kernel implementation details from XLA (e.g. we can
// export CUTLASS gemm kernels as custom kernels to XLA), so that XLA can only
// implement generic interfaces, e.g. command buffer implementation for CUDA
// can automatically add a kernel node to a graph for arbitrary custom kernel.
class NpuCustomKernel {
   public:
    NpuCustomKernel(std::unique_ptr<NpuKernelConfig> npu_kernel_config_ptr, std::string name, se::BlockDim block_dims,
                    size_t shared_memory_bytes);

    // 显式禁用复制操作，因为kernel_spec_中的指针指向npu_kernel_config_
    NpuCustomKernel(const NpuCustomKernel&) = delete;
    NpuCustomKernel& operator=(const NpuCustomKernel&) = delete;

    // 自定义移动操作，确保移动后kernel_spec_中的指向npu_kernel_config_的指针保持正确
    NpuCustomKernel(NpuCustomKernel&&) noexcept;
    NpuCustomKernel& operator=(NpuCustomKernel&&) noexcept;

    absl::string_view name() const { return name_; }

    const se::KernelLoaderSpec& kernel_spec() const { return kernel_spec_; }

    se::BlockDim block_dims() const { return block_dims_; }  // npu仅使用第一维,即x轴

    se::ThreadDim thread_dims() const { return thread_dims_; }  // npu不使用,仅作适配xla框架的占位符

    std::optional<se::ClusterDim> cluster_dims() const { return cluster_dims_; }  // npu不使用,仅作适配xla框架的占位符

    size_t shared_memory_bytes() const { return shared_memory_bytes_; }

    std::string ToString() const;

    NpuKernelConfig* npu_kernel_config_ptr() const { return npu_kernel_config_ptr_.get(); }

   private:
    // 注意：npu_kernel_config_ptr_ 必须在 kernel_spec_ 之前声明，
    // 因为 kernel_spec_ 的初始化依赖 npu_kernel_config_ptr_
    std::unique_ptr<NpuKernelConfig> npu_kernel_config_ptr_;
    std::string name_;
    se::KernelLoaderSpec kernel_spec_;
    se::BlockDim block_dims_;
    se::ThreadDim thread_dims_;
    std::optional<se::ClusterDim> cluster_dims_ = std::nullopt;
    size_t shared_memory_bytes_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_KERNEL_NPU_CUSTOM_KERNEL_H_
