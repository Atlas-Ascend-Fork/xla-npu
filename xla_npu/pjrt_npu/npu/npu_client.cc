#include "xla/xla_npu/pjrt_npu/npu/npu_client.h"
#include "xla/xla_npu/pjrt_npu/npu/npu_device.h"
#include "xla/xla_npu/pjrt_npu/host_memory_spaces.h"

namespace xla {
// Returns the CPU devices from the given PjRtCpuDevices.
// Precondition: `devices` doesn't contain nullptr.
static std::vector<NpuTopologyDescription::Device> GetDevices(absl::Span<const std::unique_ptr<PjRtNpuDevice>> devices)
{
    std::vector<NpuTopologyDescription::Device> npu_devices;
    npu_devices.reserve(devices.size());
    for (const auto& device : devices) {
        npu_devices.push_back(
            NpuTopologyDescription::Device{device->process_index(), device->local_hardware_id().value()});
    }
    return npu_devices;
}

PjRtNpuClient::PjRtNpuClient(int process_index, std::vector<std::unique_ptr<PjRtNpuDevice>> devices)
    : process_index_(process_index),
      owned_devices_(std::move(devices)),
      topology_(platform_id(), platform_name(), platform_version(), GetDevices(devices))
{
    for (const std::unique_ptr<PjRtNpuDevice>& device : owned_devices_) {
        devices_.push_back(device.get());
        CHECK(id_to_device_.insert({device->global_device_id(), device.get()}).second)
            << "Duplicate device id: " << device->global_device_id();

        device->SetClient(this);
        //TODO addressable_devices_与devices_的区别是什么
        if (device->IsAddressable()) {
            int idx = device->local_hardware_id().value();
            if (idx >= addressable_devices_.size()) {
                addressable_devices_.resize(idx + 1);
            }
            CHECK(addressable_devices_[idx] == nullptr) << idx;
            addressable_devices_[idx] = device.get();
        }
    }

    for (int idx = 0; idx < addressable_devices_.size(); ++idx) {
        auto* const device = addressable_devices_[idx];
        CHECK(device != nullptr) << idx;
        auto* npu_device = tsl::down_cast<PjRtNpuDevice*>(device);

        // Use the device id to construct a globally unique memory space id.
        const int id = device->id();

        // The first attached memory space is returned as the default by
        // PjRtCpuDevice, so attach the device memory space first.
        auto npu_device_memory_space = std::make_unique<NpuDeviceMemorySpace>(id * 3, device);
        npu_device->AttachMemorySpace(npu_device_memory_space.get());
        memory_spaces_.push_back(npu_device_memory_space.get());
        owned_memory_spaces_.push_back(std::move(npu_device_memory_space));
        /****************************** TODO 内存管理
                                auto pinned_memory_space =
                                    std::make_unique<PinnedHostMemorySpace>(id * 3 + 1, device);
                                cpu_device->AttachMemorySpace(pinned_memory_space.get());
                                memory_spaces_.push_back(pinned_memory_space.get());
                                owned_memory_spaces_.push_back(std::move(pinned_memory_space));

                                auto unpinned_memory_space =
                                    std::make_unique<UnpinnedHostMemorySpace>(id * 3 + 2, device);
                                cpu_device->AttachMemorySpace(unpinned_memory_space.get());
                                memory_spaces_.push_back(unpinned_memory_space.get());
                                owned_memory_spaces_.push_back(std::move(unpinned_memory_space));
                            }*/
    }
    // VLOG(1) << "PjRtCpuClient created.";
}

absl::StatusOr<PjRtDevice*> PjRtNpuClient::LookupDevice(PjRtGlobalDeviceId global_device_id) const
{
    auto it = id_to_device_.find(global_device_id);
    if (it != id_to_device_.end()) {
        return it->second;
    }
    return InvalidArgument("No matching device found for device_id %d", global_device_id.value());
}

absl::StatusOr<PjRtDevice*> PjRtNpuClient::LookupAddressableDevice(PjRtLocalDeviceId local_device_id) const
{
    for (auto* device : addressable_devices_) {
        if (local_device_id == device->local_device_id()) {
            return device;
        }
    }
    return InvalidArgument("No matching device found for local_device_id %d", local_device_id.value());
}

absl::Span<PjRtMemorySpace* const> PjRtNpuClient::memory_spaces() const
{
    return memory_spaces_;
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtCpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options) {
    std::string result;
    llvm:raw_string_ostream os(result);
    module.print(os);
    std::cout << os.str << std::endl;
    return absl::UnimplementedError("Compile with MLIR Module is not supported.");
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtCpuClient::CompileAndLoad(const XlaComputation& computation, CompileOptions options) {
    return absl::UnimplementedError("CompileAndLoad with XlaComputation is not supported.");
}
}  // namespace xla
