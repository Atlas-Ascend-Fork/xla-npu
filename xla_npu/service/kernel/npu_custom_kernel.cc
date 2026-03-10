/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/service/kernel/npu_custom_kernel.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/xla_npu/service/kernel/npu_kernel_config.h"

namespace xla {

NpuCustomKernel::NpuCustomKernel(std::unique_ptr<NpuKernelConfig> npu_kernel_config_ptr, std::string name,
                                 se::BlockDim block_dims, size_t shared_memory_bytes)
    : npu_kernel_config_ptr_(std::move(npu_kernel_config_ptr)),
      name_(std::move(name)),
      // 第一个指针参数通过unique_ptr指针保证生命周期有效性(与NpuCustomKernel生命周期同步)
      // auto fusion算子执行暂不依赖这里设置的arity(CreateInProcessSymbolSpec接口倒数第二个参数)表示的算子参数个数
      kernel_spec_(std::move(stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          static_cast<void*>(this->npu_kernel_config_ptr()), name, 0, nullptr))),
      block_dims_(block_dims),
      shared_memory_bytes_(shared_memory_bytes) {}

// 移动构造函数
NpuCustomKernel::NpuCustomKernel(NpuCustomKernel&& other) noexcept
    : npu_kernel_config_ptr_(std::move(other.npu_kernel_config_ptr_)),
      name_(std::move(other.name_)),
      kernel_spec_(std::move(stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          static_cast<void*>(this->npu_kernel_config_ptr()), name_, 0, nullptr))),
      block_dims_(std::move(other.block_dims_)),
      thread_dims_(std::move(other.thread_dims_)),
      cluster_dims_(std::move(other.cluster_dims_)),
      shared_memory_bytes_(other.shared_memory_bytes_) {
    // 重置源对象的状态
    other.block_dims_ = se::BlockDim(0);
    other.thread_dims_ = se::ThreadDim(0, 0, 0);
    other.cluster_dims_ = std::nullopt;
    other.shared_memory_bytes_ = 0;
}

// 移动赋值运算符
NpuCustomKernel& NpuCustomKernel::operator=(NpuCustomKernel&& other) noexcept {
    if (this != &other) {
        npu_kernel_config_ptr_ = std::move(other.npu_kernel_config_ptr_);
        name_ = std::move(other.name_);
        // 重新创建 kernel_spec_，确保其 symbol 指针指向正确的 npu_kernel_config_ptr_
        kernel_spec_ = std::move(stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
            static_cast<void*>(npu_kernel_config_ptr_.get()), name_, 0, nullptr));
        block_dims_ = std::move(other.block_dims_);
        thread_dims_ = std::move(other.thread_dims_);
        cluster_dims_ = std::move(other.cluster_dims_);
        shared_memory_bytes_ = other.shared_memory_bytes_;
        // 重置源对象的状态
        other.block_dims_ = se::BlockDim(0);
        other.thread_dims_ = se::ThreadDim(0, 0, 0);
        other.cluster_dims_ = std::nullopt;
        other.shared_memory_bytes_ = 0;
    }
    return *this;
}

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
    return absl::StrFormat("%s block_dim: %d shared_memory: %d bytes", name_, block_dims_.x, shared_memory_bytes_);
}

}  // namespace xla
