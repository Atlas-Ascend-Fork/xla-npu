#include "xla/xla_npu/pjrt/plugin/npu_topology_description.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
// #include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/strings/proto_serialization.h"

#include "xla/xla_npu/pjrt/plugin/npu_device_description.h"

namespace xla {
std::vector<std::unique_ptr<const PjRtDeviceDescription>> NpuTopologyDescription::DeviceDescriptions() const
{
    std::vector<std::unique_ptr<const PjRtDeviceDescription>> devices;
    devices.reserve(devices_.size());
    for (const Device& device : devices_) {
        devices.push_back(std::make_unique<NpuDeviceDescription>(device.process_id, device.local_device_id));
    }
    return devices;
}
}  // namespace xla