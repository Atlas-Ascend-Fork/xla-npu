/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "./status.h"

#include <stdio.h>

#include <deque>
#include <map>

namespace mlir::air {

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

const std::string& Status::empty_string() {
  static std::string* empty = new std::string;
  return *empty;
}

void Status::SlowCopyFrom(const State* src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    std::string result = "status error";
    return result;
  }
}

void Status::ReplaceAllPayloads(
    const std::unordered_map<std::string, std::string>& payloads) {
  if (ok() || payloads.empty()) return;
  if (state_ == nullptr) state_ = std::make_unique<State>();
  state_->payloads = payloads;
}

void Status::IgnoreError() const {
  // no-op
}

const std::unordered_map<std::string, std::string> Status::GetAllPayloads()
    const {
  if (ok()) return {};
  return state_->payloads;
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

// kDerivedMarker is appended to the Status message string to indicate whether a
// Status object is the root cause of an error or if it has been triggered by
// cancelling/aborting a step.
static const char* kDerivedMarker = "[_Derived_]";

void StatusGroup::Update(const Status& s) {
  if (s.ok()) {
    ++num_ok_;
  } else {
    ok_ = false;
    children_.push_back(s);
  }
}

bool StatusGroup::IsDerived(const Status& s) {
  return s.error_message().find(kDerivedMarker) != std::string::npos;
}
}  // namespace mlir::air
