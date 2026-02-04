#ifndef XLA_XLA_NPU_PJRT_PLUGIN_NPU_TOPOLOGY_H_
#define XLA_XLA_NPU_PJRT_PLUGIN_NPU_TOPOLOGY_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/pjrt/pjrt_common.h"
// #include "xla/pjrt/plugin/xla_cpu/cpu_topology.pb.h"

namespace xla {
class NpuTopology {
   public:
    struct NpuDevice {
        int process_id;
        int local_device_id;

        bool operator==(const NpuDevice& other) const {
            return process_id == other.process_id && local_device_id == other.local_device_id;
        }
    };

    explicit NpuTopology(std::vector<NpuDevice> npu_devices, std::vector<std::string> machine_attributes)
        : npu_devices_(std::move(npu_devices)), machine_attributes_(std::move(machine_attributes)) {}

    int number_of_devices() const { return npu_devices_.size(); }
    absl::Span<const NpuDevice> devices() const { return npu_devices_; }
    absl::Span<const std::string> machine_attributes() const { return machine_attributes_; }

    // static std::unique_ptr<const NpuTopology> FromProto(const CpuTopologyProto& proto);
    // CpuTopologyProto ToProto() const;

   private:
    const std::vector<NpuDevice> npu_devices_;
    const std::vector<std::string> machine_attributes_;
};

static const int kMaxNpuDevicesPerProcess = 1 << 11;

inline PjRtGlobalDeviceId PackCpuDeviceId(int process_index, int device_id) {
    return PjRtGlobalDeviceId(kMaxNpuDevicesPerProcess * process_index + device_id);
}

inline int UnpackCpuProcessIndex(PjRtGlobalDeviceId global_device_id) {
    return global_device_id.value() / kMaxNpuDevicesPerProcess;
}

inline int UnpackCpuLocalDeviceId(PjRtGlobalDeviceId global_device_id) {
    return global_device_id.value() % kMaxNpuDevicesPerProcess;
}

}  // namespace xla

#endif  // XLA_XLA_NPU_PJRT_PLUGIN_NPU_TOPOLOGY_H_
