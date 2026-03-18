/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TRANSLATIONS_STATUS_H_
#define AIR_TRANSLATIONS_STATUS_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>

namespace mlir::air {

/// @ingroup core
/// Denotes success or failure of a call.
class Status {
 public:
  /// Create a success status.
  Status() {}

  /// Copy the specified status.
  Status(const Status& s);
  Status& operator=(const Status& s);

  static Status OK() { return Status(); }

  /// Returns true iff the status indicates success.
  bool ok() const { return (state_ == nullptr); }

  const std::string& error_message() const {
    return ok() ? empty_string() : state_->msg;
  }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
  /// preserves the current status, but may augment with additional
  /// information about `new_status`.
  ///
  /// Convenient way of keeping track of the first error encountered.
  /// Instead of:
  ///   `if (overall_status.ok()) overall_status = new_status`
  /// Use:
  ///   `overall_status.Update(new_status);`
  void Update(const Status& new_status);

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  std::string ToString() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

  // Returns all the payload information.
  // Returns an empty result if status is ok.
  const std::unordered_map<std::string, std::string> GetAllPayloads() const;

  // Copies all the payloads using the input and discards existing payloads.
  // Does nothing if status is ok or 'payloads' is empty.
  void ReplaceAllPayloads(
      const std::unordered_map<std::string, std::string>& payloads);

 private:
  static const std::string& empty_string();

  struct State {
    std::string msg;
    std::unordered_map<std::string, std::string> payloads;
  };

  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  std::unique_ptr<State> state_;

  void SlowCopyFrom(const State* src);
};

// Helper class to manage multiple child status values.
class StatusGroup {
 public:
  // Utility function to mark a Status as derived. By marking derived status,
  // Derived status messages are ignored when reporting errors to end users.
  static bool IsDerived(const Status& s);

  // Enable warning and error log collection for appending to the aggregated
  // status. This function may be called more than once.

  // Return a merged status with combined child status messages with a summary.
  Status as_summary_status() const;
  // Return a merged status with combined child status messages with
  // concatenation.
  Status as_concatenated_status() const;

  bool ok() const { return ok_; }

  // Augment this group with the child status `status`.
  void Update(const Status& status);

  // Attach recent warning and error log messages
  bool HasLogMessages() const { return !recent_logs_.empty(); }

 private:
  bool ok_ = true;
  size_t num_ok_ = 0;
  std::vector<Status> children_;
  std::vector<std::string> recent_logs_;  // recent warning and error logs
};

inline Status::Status(const Status& s)
    : state_((s.state_ == nullptr) ? nullptr : new State(*s.state_)) {}

inline Status& Status::operator=(const Status& s) {
  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    SlowCopyFrom(s.state_.get());
  }
  return *this;
}

inline bool Status::operator==(const Status& x) const {
  return (this->state_ == x.state_) || (ToString() == x.ToString());
}

inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

/// @ingroup core
std::ostream& operator<<(std::ostream& os, const Status& x);

typedef std::function<void(const Status&)> StatusCallback;

}  // namespace mlir::air

#endif  // AIR_TRANSLATIONS_STATUS_H_
