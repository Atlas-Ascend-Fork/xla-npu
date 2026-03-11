/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef XLA_XLA_NPU_STREAM_EXECUTOR_NPU_AUTOFUSE_KERNEL_H_
#define XLA_XLA_NPU_STREAM_EXECUTOR_NPU_AUTOFUSE_KERNEL_H_

#include "absl/status/status.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"

namespace stream_executor {

class NpuAutoFuseKernel : public Kernel {
 public:
  explicit NpuAutoFuseKernel(const std::string& load_file_base_dir,
                             const std::string& kernel_name,
                             StreamExecutor* executor)
      : load_file_base_dir_(load_file_base_dir),
        kernel_name_(kernel_name),
        executor_(executor) {}

  void set_arity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const override { return arity_; }

  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const override {
    return 0;
  }

  absl::Status Load();

  // 当前tiling函数文件和kernel文件链接在一起,
  // 原本TilingFile为分开的afir编译目录下的"libhost_tiling.so"
  std::string GetTilingFileName() const { return GetKernelFileName(); }
  std::string GetKernelFileName() const { return kernel_name_; }

 private:
  absl::Status Launch(const ThreadDim& thread_dims, const BlockDim& block_dims,
                      const std::optional<ClusterDim>& cluster_dims,
                      Stream* stream, const KernelArgs& args) override;

  StreamExecutor* executor_ = nullptr;
  unsigned arity_ = 0;  // number of formal parameters the kernel takes

  using graphStatus = uint32_t;
  using GetTilingSizeFunc = size_t (*)();
  using TilingFunc = graphStatus (*)(void*, uint32_t*, uint32_t*, void*);
  using LaunchFunc = uint32_t (*)(uint32_t, void*, void**, int32_t, void**,
                                  int32_t, void*, void*);

  GetTilingSizeFunc GetTilingSizeFunc_;
  TilingFunc TilingFunc_;
  LaunchFunc LaunchFunc_;

  std::string load_file_base_dir_;  // 编译生成的.o, .bin, .so后缀文件的根路径
  std::string kernel_name_;
  // 由于这里的库生命周期仅伴随单个融合算子,因此不使用functionLoader中的FunctionRegister类单例
  std::unordered_map<std::string, std::unique_ptr<npu::FunctionLoader>>
      registry;
};

}  // namespace stream_executor

#endif  // XLA_XLA_NPU_STREAM_EXECUTOR_NPU_AUTOFUSE_KERNEL_H_
