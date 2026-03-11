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

#ifndef XLA_XLA_NPU_STREAM_EXECUTOR_NPU_EXECUTOR_H_
#define XLA_XLA_NPU_STREAM_EXECUTOR_NPU_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_common.h"
#include "xla/tsl/platform/threadpool.h"

namespace stream_executor {

// Declares the NpuExecutor class, which is a NPU-only implementation of
// the StreamExecutor interface. For now, this is used for testing and to
// examine the performance of host-based StreamExecutor code.
//
// This is useful for evaluating the performance of host-based or fallback
// routines executed under the context of a GPU executor.
class NpuExecutor : public StreamExecutorCommon {
 public:
  NpuExecutor(Platform* platform, int device_ordinal)
      : StreamExecutorCommon(platform), device_ordinal_(device_ordinal) {}

  absl::Status Init() override;

  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const KernelLoaderSpec& spec) override;

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;
  void Deallocate(DeviceMemoryBase* mem) override;

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override {
    void* ptr = new char[size];
    return std::make_unique<GenericMemoryAllocation>(
        ptr, size,
        [](void* ptr, uint64_t size) { delete[] static_cast<char*>(ptr); });
  }

  bool SynchronizeAllActivity() override { return true; }
  absl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64_t size) override;

  absl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64_t size) override;
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64_t size) override;

  void DeallocateStream(Stream* stream) override;

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return CreateDeviceDescription(0);
  }

  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);
  int device_ordinal() const override { return device_ordinal_; }

  absl::Status EnablePeerAccessTo(StreamExecutor* other) override {
    return absl::OkStatus();
  }

  bool CanEnablePeerAccessTo(StreamExecutor* other) override { return true; }

  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;

  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override;
  absl::StatusOr<std::unique_ptr<MemoryAllocator>> CreateMemoryAllocator(
      MemoryType type) override;

  absl::StatusOr<std::shared_ptr<DeviceMemoryBase>> CreateOrShareConstant(
      Stream* stream, absl::Span<const uint8_t> content) override;

 private:
  int device_ordinal_;
  std::shared_ptr<tsl::thread::ThreadPool> thread_pool_;
  // Track freed memory addresses to prevent double-free
  std::set<void*> freed_addresses_;
  std::mutex freed_addresses_mutex_;
};

}  // namespace stream_executor

#endif  // XLA_XLA_NPU_STREAM_EXECUTOR_NPU_EXECUTOR_H_
