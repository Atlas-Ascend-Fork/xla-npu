#ifndef NPU_TOPOLOGY_DESCRIPTION_H_
#define NPU_TOPOLOGY_DESCRIPTION_H_
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
// #include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
// #include "xla/pjrt/pjrt_npu/plugin/npu_device_description.h"

namespace xla {
class NpuTopologyDescription : public PjRtTopologyDescription {
public:
    struct Device {
        int process_id;
        int local_device_id;

        bool operator==(const Device& other) const
        {
            return process_id == other.process_id && local_device_id == other.local_device_id;
        }
    };
    NpuTopologyDescription(const PjRtPlatformId platform_id, const absl::string_view platform_name,
                           const absl::string_view platform_version, std::vector<Device> devices)
        : platform_id_(platform_id),
          platform_name_(platform_name),
          platform_version_(platform_version),
          devices_(std::move(devices))
    {
    }

    ~NpuTopologyDescription() override = default;

    bool operator==(const NpuTopologyDescription& other) const
    {
        return this->platform_id() == other.platform_id() && this->platform_name() == other.platform_name() &&
               this->platform_version() == other.platform_version();
    }

    PjRtPlatformId platform_id() const override
    {
        return platform_id_;
    }

    absl::string_view platform_name() const override
    {
        return platform_name_;
    }

    absl::string_view platform_version() const override
    {
        return platform_version_;
    }

    std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions() const override;

    absl::StatusOr<std::string> Serialize() const override
    {
        return absl::UnimplementedError("Serialize is unsupported.");
    }

    // Returns vendor specific attributes about the topology.
    const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes() const override
    {
        return attributes_;
    }

    absl::StatusOr<Layout> GetDefaultLayout(PrimitiveType element_type,
                                                 absl::Span<const int64_t> dims) const override {
      return absl::StatusOr<Layout>{};
    }

private:
    const PjRtPlatformId platform_id_;
    const std::string platform_name_;
    const std::string platform_version_;
    std::vector<Device> devices_;
    absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_ = {};
};
}  // namespace xla
#endif  //NPU_TOPOLOGY_DESCRIPTION_H_