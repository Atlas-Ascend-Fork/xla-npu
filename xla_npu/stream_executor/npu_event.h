/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_STREAM_EXECUTOR_NPU_EVENT_H_
#define XLA_XLA_NPU_STREAM_EXECUTOR_NPU_EVENT_H_

#include <memory>

#include "absl/synchronization/notification.h"
#include "xla/stream_executor/event.h"

namespace stream_executor {

// This class is a npu-side implementation of the Event interface. It is
// intended to be used with the NpuStream implementation.
class NpuEvent : public Event {
 public:
  NpuEvent() : notification_(std::make_shared<absl::Notification>()) {}

  std::shared_ptr<absl::Notification>& notification() { return notification_; }

  Status PollForStatus() override {
    return notification_->HasBeenNotified() ? Event::Status::kComplete
                                            : Event::Status::kPending;
  }

 private:
  // We use a std::shared_ptr here because the client may delete the NpuEvent
  // object while there are still RecordEvent and WaitForEvent callbacks pending
  // on a stream.
  std::shared_ptr<absl::Notification> notification_;
};

}  // namespace stream_executor

#endif  // XLA_XLA_NPU_STREAM_EXECUTOR_NPU_EVENT_H_
