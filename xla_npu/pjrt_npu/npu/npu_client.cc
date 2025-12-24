#include <map>
#include <memory>
#include <llvm/IR/DataLayout.h>
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/pjrt/thread_pool_async_work_runner.h"

#include "xla/xla_npu/pjrt_npu/npu/npu_client.h"
#include "xla/xla_npu/pjrt_npu/npu/npu_device.h"
#include "xla/xla_npu/pjrt_npu/npu/raw_buffer.h"
#include "xla/xla_npu/pjrt_npu/npu/ge/graph_data.h"
#include "xla/xla_npu/pjrt_npu/npu/service/npu_executable.h"
#include "xla/xla_npu/pjrt_npu/npu/abstract_npu_buffer.h"
#include "xla/xla_npu/pjrt_npu/host_memory_spaces.h"
#include "xla/xla_npu/third_party/metadef/external/graph/graph.h"
#include "xla/xla_npu/third_party/metadef/external/graph/ascend_string.h"
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

static tsl::ThreadOptions GetThreadOptions() {
  tsl::ThreadOptions thread_options;
  // On Mac OS the default stack size is 512KiB, which is too small for some
  // BLAS and LAPACK functions (https://github.com/google/jax/issues/20428).
  // On Linux we also observed that 2MB wasn't enough to run some OpenBLAS
  // functions.
  thread_options.stack_size = 8 * 1024 * 1024;
  return thread_options;
}

PjRtNpuClient::PjRtNpuClient(int process_index, std::vector<std::unique_ptr<PjRtNpuDevice>> devices)
    : process_index_(process_index),
      owned_devices_(std::move(devices)),
      transpose_cache_(1024),
      topology_(platform_id(), platform_name(), platform_version(), GetDevices(devices)),
      pjrt_client_thread_pool_(new tsl::thread::ThreadPool(
        tsl::Env::Default(), GetThreadOptions(), "XLAPjRtNpuClient", 1)), //TODO 线程池大小
      async_work_runner_(MakeThreadPoolAsyncWorkRunner(pjrt_client_thread_pool_.get()))
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

absl::StatusOr<int64_t> PjRtNpuClient::GetOnDeviceBytesCount(
    PjRtMemorySpace* memory_space, const Shape& shape) const
{
    auto pointer_size = llvm::DataLayout(DataLayout())
            .getPointerSize(0 /* default address space */);
    if (shape.IsTuple() || shape.is_static()) {
        return ShapeUtil::ByteSizeOf(shape, pointer_size);
    }
    int64_t metadata_size = sizeof(int32_t) * shape.dimensions().size();
    return ShapeUtil::ByteSizeOf(shape, pointer_size) + metadata_size;
}

absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>> PjRtNpuClient::AllocateRawBuffer(
    PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
    bool retry_on_oom,
    tsl::AsyncValueRef<bool> allocate_after)
{
    return xla::NpuRawBuffer::Allocate(memory_space, on_device_bytes_count);
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> PjRtNpuClient::LinearizeHostBufferInto(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    const xla::Shape& device_shape,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer)
{
    return tsl::down_cast<NpuRawBuffer*>(raw_buffer.get())
        ->CopyFromHostBuffer(
            data, type, dims, byte_strides, host_buffer_semantics,
            std::move(on_done_with_host_buffer), device_shape,
            async_work_runner(), &transpose_mu_, &transpose_cache_);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> PjRtNpuClient::DefineBuffer(
    const Shape& on_device_shape,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
    absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>
        definition_device_events,
    bool raw_buffer_is_mutable)
{
    absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4> definition_events;
    for (auto& ev : definition_device_events) {
        std::cout << "PjRtNpuClient::DefineBuffer event: " << ev.get() << std::endl;
        definition_events.push_back(
            tsl::down_cast<NpuTrackedDeviceEvent*>(ev.get())->event());
    }
    return std::unique_ptr<PjRtBuffer>(std::make_unique<CommonPjRtBufferImpl>(
        on_device_shape,
        std::make_unique<TrackedNpuDeviceBuffer>(
            /*owns_buffers=*/raw_buffer_is_mutable,
            tsl::down_cast<NpuRawBuffer*>(raw_buffer.get())->buffer(),
            ShapeUtil::ByteSizeOf(on_device_shape), std::move(definition_events)),
        raw_buffer->memory_space()));
}

absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>, tsl::RCReference<PjRtDeviceEvent>>>
PjRtNpuClient::CreateLinkedEventPromise(PjRtMemorySpace* memory_space,
    absl::string_view debug_info) {
    auto definition_event_promise = tsl::MakeIndirectAsyncValue();
    auto definition_event = tsl::MakeRef<NpuTrackedDeviceEvent>(
        tsl::AsyncValueRef<NpuEvent>(definition_event_promise));
    return std::make_pair(tsl::MakeRef<NpuTrackedDeviceEventPromise>(
                          std::move(definition_event_promise)),
                          std::move(definition_event));
}


absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtNpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options) { 
    std::string result;
    llvm::raw_string_ostream os(result);
    module.print(os);
    std::cout << os.str() << std::endl;

    auto graph_data = std::make_unique<npu::GraphData>();
    graph_data->graph = std::make_shared<ge::Graph>("XLAGraph");
    graph_data->id = 0;

    std::map<ge::AscendString, ge::AscendString> ge_options = {
        {"ge.exec.deviceId", "0"}
    };
    if (!ge_initialized) {    
        ge_initialized = true;    
        if (ge::GEInitialize(ge_options) != ge::SUCCESS) {
            return absl::InternalError("Failed to initialize GE.");
        } else {
            (void)ge_options.emplace(ge::AscendString("ge.session_device_id"), ge::AscendString("0"));
            global_ge_session = std::make_unique<ge::Session>(ge_options);
            if (global_ge_session == nullptr) {
                return absl::InternalError("Failed to create GE session.");
            }
        }
    }

    if (global_ge_session->AddGraph(graph_data->id, *graph_data->graph, ge_options) != ge::SUCCESS) {
        return absl::InternalError("Failed to AddGraph.");
    }
    if (global_ge_session->CompileGraph(graph_data->id) != ge::SUCCESS) {
        return absl::InternalError("Failed to CompileGraph.");
    }
    std::shared_ptr<ge::CompiledGraphSummary> summary = global_ge_session->GetCompiledGraphSummary(graph_data->id);
    if (summary ==  nullptr) {
        return absl::InternalError("Failed to GetCompiledGraphSummary.");
    }


    TF_ASSIGN_OR_RETURN(
      auto npu_executable,
      npu::NpuExecutable::Create(nullptr, nullptr, nullptr, nullptr, std::move(graph_data)));

    int num_replicas;
    int num_partitions;
    std::shared_ptr<DeviceAssignment> device_assignment;
    absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices;
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids;
    std::vector<PjRtDevice*> addressable_devices;
    auto input_options = options;

    auto executable = std::make_unique<PjRtNpuExecutable>(
      num_replicas, num_partitions, std::move(device_assignment),
      options.parameter_is_tupled_arguments, std::move(input_options),
      std::move(npu_executable), std::move(result_buffer_indices),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this, nullptr);

    return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtNpuClient::CompileAndLoad(const XlaComputation& computation, CompileOptions options) {
    return absl::UnimplementedError("CompileAndLoad with XlaComputation is not supported.");
}

PjRtNpuExecutable::PjRtNpuExecutable(
    int num_replicas, int num_partitions,
    std::shared_ptr<DeviceAssignment> device_assignment,
    bool parameter_is_tupled_arguments, CompileOptions compile_options,
    std::unique_ptr<Executable> npu_executable,
    absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, PjRtNpuClient* client,
    std::unique_ptr<HloModule> unoptimized_hlo_module)
    : client_(client),
      num_replicas_(num_replicas),
      num_partitions_(num_partitions),
      device_assignment_(std::move(device_assignment)),
      parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      compile_options_(std::move(compile_options)),
      npu_executable_(std::move(npu_executable)),
      result_buffer_indices_(std::move(result_buffer_indices)),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)),
      unoptimized_hlo_module_(std::move(unoptimized_hlo_module)) {
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
PjRtNpuExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<Future<>>>& returned_futures) const {
    return absl::UnimplementedError("PjRtNpuExecutable::Execute is not supported.");
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtNpuExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
    return absl::UnimplementedError("PjRtNpuExecutable::ExecutePortable is not supported.");
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtNpuExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
    return absl::UnimplementedError("PjRtNpuExecutable::ExecutePortable is not supported."); 
}

void PjRtNpuExecutable::Delete() {}

bool PjRtNpuExecutable::IsDeleted() const { return false; }

absl::StatusOr<std::string> PjRtNpuExecutable::SerializeExecutable() const {
    return absl::UnimplementedError("PjRtNpuExecutable::SerializeExecutable is not supported."); 
}

absl::StatusOr<CompiledMemoryStats> PjRtNpuExecutable::GetCompiledMemoryStats()
    const {
    return absl::UnimplementedError("PjRtNpuExecutable::GetCompiledMemoryStats is not supported."); 
}

}  // namespace xla