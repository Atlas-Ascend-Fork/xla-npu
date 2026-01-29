#include "xla/xla_npu/service/kernel/npu_custom_kernel.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {

NpuCustomKernel::NpuCustomKernel(std::string name, se::KernelLoaderSpec kernel_spec, se::BlockDim block_dims,
                                 size_t shared_memory_bytes)
    : name_(std::move(name)),
      kernel_spec_(std::move(kernel_spec)),
      block_dims_(block_dims),
      shared_memory_bytes_(shared_memory_bytes) {}

// NpuCustomKernel::NpuCustomKernel(std::string name, se::KernelLoaderSpec kernel_spec, se::BlockDim block_dims,
//                                  se::ThreadDim thread_dims, size_t shared_memory_bytes)
//     : name_(std::move(name)),
//       kernel_spec_(std::move(kernel_spec)),
//       block_dims_(block_dims),
//       thread_dims_(thread_dims),
//       cluster_dims_(std::nullopt),
//       shared_memory_bytes_(shared_memory_bytes) {}

// NpuCustomKernel::NpuCustomKernel(std::string name, se::KernelLoaderSpec kernel_spec, se::BlockDim block_dims,
//                                  se::ThreadDim thread_dims, se::ClusterDim cluster_dims, size_t shared_memory_bytes)
//     : name_(std::move(name)),
//       kernel_spec_(std::move(kernel_spec)),
//       block_dims_(block_dims),
//       thread_dims_(thread_dims),
//       cluster_dims_(cluster_dims),
//       shared_memory_bytes_(shared_memory_bytes) {}

std::string NpuCustomKernel::ToString() const {
    return "";

    // std::string cluster_dims_str =
    //     cluster_dims_.has_value()
    //         ? absl::StrFormat("cluster: [%d, %d, %d]", cluster_dims_->x, cluster_dims_->y, cluster_dims_->z)
    //         : "";
    // return absl::StrFormat(
    //     "%s grid: [%d, %d, %d] threads: [%d, %d, %d] %s "
    //     "shared_memory: %d bytes",
    //     name_, block_dims_.x, block_dims_.y, block_dims_.z, thread_dims_.x, thread_dims_.y, thread_dims_.z,
    //     cluster_dims_str, shared_memory_bytes_);
}

}  // namespace xla::gpu
