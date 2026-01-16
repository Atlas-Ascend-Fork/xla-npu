#include "xla/xla_npu/pjrt/plugin/npu_device_description.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/xla_npu/pjrt/npu_pjrt_common.h"

namespace xla {
NpuDeviceDescription::NpuDeviceDescription(int process_id, int local_device_id)
    : id_(PjRtGlobalDeviceId(process_id * 16 + local_device_id)),
      process_index_(process_id),
      local_hardware_id_(local_device_id) {
    debug_string_ = absl::StrCat("XLART_NPU_", id_.value());
    to_string_ = absl::StrCat("NpuDevice(id=", id_.value(), ")");
}

absl::string_view NpuDeviceDescription::device_kind() const { return NpuName(); }

absl::string_view NpuDeviceDescription::DebugString() const { return debug_string_; }

absl::string_view NpuDeviceDescription::ToString() const { return to_string_; }
}  // namespace xla