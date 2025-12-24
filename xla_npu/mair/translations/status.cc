#include "./status.h"

#include <stdio.h>

#include <deque>
#include <map>


namespace mlir::air {

void Status::Update(const Status &new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State *src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

const std::string &Status::empty_string() {
  static std::string *empty = new std::string;
  return *empty;
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    std::string result = "status error";
    // string result(error_name(code()));
    // result += ": ";
    // result += state_->msg;
    return result;
  }
}

void Status::IgnoreError() const {
  // no-op
}

const std::unordered_map<std::string, std::string> Status::GetAllPayloads() const {
  if (ok()) return {};
  return state_->payloads;
}

void Status::ReplaceAllPayloads(const std::unordered_map<std::string, std::string> &payloads) {
  if (ok() || payloads.empty()) return;
  if (state_ == nullptr) state_ = std::make_unique<State>();
  state_->payloads = payloads;
}

std::ostream &operator<<(std::ostream &os, const Status &x) {
  os << x.ToString();
  return os;
}

// kDerivedMarker is appended to the Status message string to indicate whether a
// Status object is the root cause of an error or if it has been triggered by
// cancelling/aborting a step.
static const char *kDerivedMarker = "[_Derived_]";

// Status StatusGroup::MakeDerived(const Status& s) {
//   if (IsDerived(s)) {
//     return s;
//   } else {
//     return Status(s.code(), strings::StrCat(kDerivedMarker, s.error_message()));
//   }
// }

bool StatusGroup::IsDerived(const Status &s) {
  return s.error_message().find(kDerivedMarker) != std::string::npos;
}

// void StatusGroup::ConfigureLogHistory() {
//   StatusLogSink::GetInstance()->enable();
// }

void StatusGroup::Update(const Status &s) {
  if (s.ok()) {
    ++num_ok_;
  } else {
    ok_ = false;
    children_.push_back(s);
  }
}

// void StatusGroup::AttachLogMessages() {
//   recent_logs_.clear();
//   StatusLogSink::GetInstance()->GetMessages(&recent_logs_);
// }

}  // namespace mlir::air
