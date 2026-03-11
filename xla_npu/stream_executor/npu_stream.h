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

#ifndef XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_H_
#define XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_H_

#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_npu/common/npu_operation.h"

namespace stream_executor {

// NpuStream for launching work on the host NPU. In contrast to the device
// streams, NpuStream is fully synchronous and launches all operations in the
// caller thread.
class NpuStream : public StreamCommon {
 public:
  explicit NpuStream(StreamExecutor* executor);
  ~NpuStream() override;

  absl::Status Init();

  PlatformSpecificHandle platform_specific_handle() const override;

  absl::Status BlockHostUntilDone() override;

  absl::Status WaitFor(Stream* other) override;
  absl::Status WaitFor(Event* event) override;
  absl::Status RecordEvent(Event* event) override;
  absl::Status MemZero(DeviceMemoryBase* location, uint64_t size) override;
  absl::Status Memset32(DeviceMemoryBase* location, uint32_t pattern,
                        uint64_t size) override;
  absl::Status Memcpy(DeviceMemoryBase* npu_dst,
                      const DeviceMemoryBase& npu_src, uint64_t size) override;
  absl::Status Memcpy(DeviceMemoryBase* npu_dst, const void* host_src,
                      uint64_t size) override;
  absl::Status Memcpy(void* host_dst, const DeviceMemoryBase& npu_src,
                      uint64_t size) override;
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;

 private:
  int device_ordinal_;
  aclrtStream stream_;  // aclrtStream = void*
  bool stream_init_ = false;
};

}  // namespace stream_executor

#endif  // XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_H_
