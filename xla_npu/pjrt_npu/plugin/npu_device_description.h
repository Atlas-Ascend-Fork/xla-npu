#ifndef NPU_DEVICE_DESCRIPTION_H
#define NPU_DEVICE_DESCRIPTION_H

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_device_description.h"

namespace xla {
class NpuDeviceDescription final : public PjRtDeviceDescription {
public:
    explicit NpuDeviceDescription(int process_id, int local_device_id);

    ~NpuDeviceDescription() override = default;

    int id() const override
    {
        return id_.value();
    }

    int process_index() const override
    {
        return process_index_;
    }

    int local_hardware_id() const
    {
        return local_hardware_id_;
    }

    absl::string_view device_kind() const override;

    absl::string_view DebugString() const override;

    absl::string_view ToString() const override;

    const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes() const override
    {
        return attributes_;
    }

private:
    PjRtGlobalDeviceId id_;
    int process_index_;
    int local_hardware_id_;
    std::string debug_string_;
    std::string to_string_;
    absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_ = {};
};
}  // namespace xla
#endif  //NPU_DEVICE_DESCRIPTION_H