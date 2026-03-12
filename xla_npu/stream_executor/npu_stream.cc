/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/stream_executor/npu_stream.h"

#include <string.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/xla_npu/common/npu_operation.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/stream_executor/npu_event.h"

namespace stream_executor {

NpuStream::NpuStream(StreamExecutor* executor) : StreamCommon(executor) {}

NpuStream::~NpuStream() { parent()->DeallocateStream(this); }

absl::Status NpuStream::Init() {
  // 保证当前线程执行过set device
  xla::NpuOperation::SetDevice(parent()->device_ordinal());

  TF_RETURN_IF_ERROR(xla::NpuOperation::CreateStream(&stream_));
  stream_init_ = true;

  return absl::OkStatus();
}

NpuStream::PlatformSpecificHandle NpuStream::platform_specific_handle() const {
  if (stream_init_) {
    return {stream_};
  } else {
    NPU_LOG_FORMAT_PRINT(WARN,
                         "The stream '%s' has not been initialized, failing to "
                         "obtain the stream handle.",
                         GetName());
    return {};
  }
}

absl::Status NpuStream::BlockHostUntilDone() {
  NPU_LOG_FORMAT_PRINT(DEBUG, "Block host until stream '%s' done", GetName());
  TF_RETURN_IF_ERROR(xla::NpuOperation::SetDevice(parent()->device_ordinal()));

  auto acl_ret = npu::aclrtSynchronizeStream(stream_);
  CHECK_COND_AND_PRINT_NPU_LOG(acl_ret != ACL_SUCCESS,
                               "Synchronize stream failed. Error code: %d",
                               acl_ret);

  return absl::OkStatus();
}

absl::Status NpuStream::Memcpy(DeviceMemoryBase* npu_dst,
                               const DeviceMemoryBase& npu_src, uint64_t size) {
  TF_RETURN_IF_ERROR(xla::NpuOperation::SetDevice(parent()->device_ordinal()));

  void* dst_mem = npu_dst->opaque();
  void* src_mem = const_cast<void*>(npu_src.opaque());
  auto acl_ret = npu::aclrtMemcpy(dst_mem, size, src_mem, size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (acl_ret != ACL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("Execute aclrtMemcpy d2d on stream '", GetName(),
                     "' failed, error code: ", acl_ret));
  }
  NPU_LOG_FORMAT_PRINT(DEBUG,
                       "Npu memory copy d2d on stream '%s' success, dst npu "
                       "addr %p, src npu addr %p, size %d",
                       GetName(), dst_mem, src_mem, size);
  return absl::OkStatus();
}

absl::Status NpuStream::Memcpy(void* host_dst, const DeviceMemoryBase& npu_src,
                               uint64_t size) {
  TF_RETURN_IF_ERROR(xla::NpuOperation::SetDevice(parent()->device_ordinal()));

  void* src_mem = const_cast<void*>(npu_src.opaque());
  auto acl_ret = npu::aclrtMemcpy(host_dst, size, src_mem, size,
                                  ACL_MEMCPY_DEVICE_TO_HOST);
  if (acl_ret != ACL_SUCCESS) {
    return absl::InternalError(absl::StrCat(
        "Execute aclrtMemcpy d2h on stream '%s' failed, error code: %d",
        GetName(), acl_ret));
  }
  NPU_LOG_FORMAT_PRINT(DEBUG,
                       "Npu memory copy d2h on stream '%s' success, dst cpu "
                       "addr %p, src npu addr %p, size %d",
                       GetName(), host_dst, src_mem, size);
  return absl::OkStatus();
}

absl::Status NpuStream::Memcpy(DeviceMemoryBase* npu_dst, const void* host_src,
                               uint64_t size) {
  TF_RETURN_IF_ERROR(xla::NpuOperation::SetDevice(parent()->device_ordinal()));

  void* dst_mem = npu_dst->opaque();
  auto acl_ret = npu::aclrtMemcpy(dst_mem, size, host_src, size,
                                  ACL_MEMCPY_HOST_TO_DEVICE);
  if (acl_ret != ACL_SUCCESS) {
    return absl::InternalError(absl::StrCat(
        "Execute aclrtMemcpy h2d on stream '%s' failed, error code: %d",
        GetName(), acl_ret));
  }
  NPU_LOG_FORMAT_PRINT(DEBUG,
                       "Npu memory copy h2d on stream '%s' success, dst npu "
                       "addr %p, src cpu addr %p, size %d",
                       GetName(), dst_mem, host_src, size);
  return absl::OkStatus();
}

absl::Status NpuStream::Memset32(DeviceMemoryBase* location, uint32_t pattern,
                                 uint64_t size) {
  return absl::UnimplementedError("NpuStream::Memset32 is not supported.");
}

absl::Status NpuStream::MemZero(DeviceMemoryBase* location, uint64_t size) {
  void* npu_mem = location->opaque();
  if (npu_mem == nullptr) {
    return absl::InvalidArgumentError("Cannot memset nullptr");
  }
  // 使用aclrtMemset将内存清零
  aclError ret = npu::aclrtMemset(npu_mem, size, 0, size);
  if (ret != ACL_SUCCESS) {
    return absl::InternalError(absl::StrCat("Execute aclrtMemset on stream '",
                                            GetName(),
                                            "' failed, error code: ", ret));
  }
  NPU_LOG_FORMAT_PRINT(
      DEBUG, "Execute aclrtMemset on stream '%s' success, addr %p, size %d",
      GetName(), npu_mem, size);
  return absl::OkStatus();
}

absl::Status NpuStream::WaitFor(Stream* other) {
  return absl::UnimplementedError("NpuStream::WaitFor other is not supported.");
}

absl::Status NpuStream::WaitFor(Event* event) {
  return absl::UnimplementedError("NpuStream::WaitFor event is not supported.");
}

absl::Status NpuStream::RecordEvent(Event* event) {
  // notification->Notify();
  return absl::UnimplementedError("NpuStream::RecordEvent is not supported.");
}

absl::Status NpuStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "Execute host callback on stream '%s'",
                       GetName());

  return std::move(callback)();
}

}  // namespace stream_executor
