#include <map>
#include <memory>
#include <numeric>
#include <functional>
#include <llvm/IR/DataLayout.h>
#include "tsl/profiler/lib/traceme.h"
#include "xla/pjrt/utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/pjrt/thread_pool_async_work_runner.h"

#include "xla/xla_npu/pjrt_npu/npu/npu_client.h"

#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/hlo_module_util.h"
#include "xla/xla_npu/pjrt_npu/npu/npu_device.h"
#include "xla/xla_npu/pjrt_npu/npu/raw_buffer.h"
#include "xla/xla_npu/pjrt_npu/npu/abstract_npu_buffer.h"
#include "xla/xla_npu/pjrt_npu/npu/ge/graph_data.h"
#include "xla/xla_npu/pjrt_npu/npu/service/npu_executable.h"
#include "xla/xla_npu/pjrt_npu/host_memory_spaces.h"
#include "xla/xla_npu/third_party/metadef/external/graph/graph.h"
#include "xla/xla_npu/third_party/metadef/external/graph/tensor.h"
#include "xla/xla_npu/third_party/metadef/external/graph/ascend_string.h"
#include "xla/xla_npu/third_party/metadef/external/graph/operator_factory.h"
#include "xla/xla_npu/third_party/register/aclInterface.h"
#include "xla/xla_npu/mair/test/test_mair_gen.h"
#include "xla/xla_npu/mair/stablehlo2mair_converter.h"


namespace xla {

struct BufferInfo {
    tsl::AsyncValueRef<NpuDeviceMemory> buffer;
    bool owns_buffer;
    size_t buffer_size;
};

struct BufferAlloc {
    // All data members should have the same size.
    absl::InlinedVector<tsl::AsyncValueRef<NpuDeviceMemory>, 4> buffers;
    absl::InlinedVector<size_t, 4> allocation_sizes;

    void Allocate(const NpuDeviceMemory::Allocator& allocator) {
        for (int i = 0; i < buffers.size(); ++i) {
        auto status = NpuDeviceMemory::AllocateInto(
            allocation_sizes[i], buffers[i].AsPtr(), allocator);
        if (!status.ok()) {
            buffers[i].SetError(status);
            return;
        }
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(buffers[i]->untyped_data(),
                                            allocation_sizes[i]);
        }
    }
};

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

// The following few helpers are adapted from XLA:NPU to create a buffer table
// and assemble the buffer pointers in order to call into NpuExecutable.
static absl::StatusOr<BufferInfo> MemoryForAllocation(
    const BufferAllocation& allocation,
    // absl::Span<const cpu::ConstantAllocation> constants,
    absl::Span<std::pair<bool, TrackedNpuDeviceBuffer*> const> arguments,
    BufferAlloc& buffer_alloc)
    // BufferAllocAndCopy& buffer_alloc_and_copy,
    // const tsl::AsyncValueRef<NpuDeviceMemory>& tuple_index_table)
{
    BufferInfo buffer_info;
    if (allocation.is_entry_computation_parameter()) {
        bool can_donate = false;
        TrackedNpuDeviceBuffer* arg = nullptr;
        size_t buffer_size;
        tsl::AsyncValuePtr<NpuDeviceMemory> out;
    //     if (tuple_index_table) {
    //     if (allocation.param_shape_index().empty()) {
    //         out = tuple_index_table.AsPtr();
    //         buffer_size = arguments.size() * sizeof(void*);
    //     } else if (allocation.param_shape_index().size() == 1) {
    //         std::tie(can_donate, arg) =
    //             arguments[allocation.param_shape_index()[0]];
    //         out = arg->buffer().AsPtr();
    //         buffer_size = arg->BufferSize();
    //     } else {
    //         return absl::InvalidArgumentError(absl::StrCat(
    //             "Nested tuples are not supported for argument: ",
    //             allocation.parameter_number(),
    //             " at shape index:", allocation.param_shape_index().ToString()));
    //     }
    //     } else if (!allocation.param_shape_index().empty()) {
    //     return absl::InvalidArgumentError(absl::StrCat(
    //         "Nested tuples are not supported for argument: ",
    //         allocation.parameter_number(),
    //         " at shape index:", allocation.param_shape_index().ToString()));
    //     } else {
        std::tie(can_donate, arg) = arguments[allocation.parameter_number()];
        out = arg->buffer().AsPtr();
        buffer_size = arg->BufferSize();
            // }
        CHECK_EQ(allocation.size(), buffer_size)
            << "Size mismatch on param " << allocation.parameter_number()
            << " at shape index " << allocation.param_shape_index().ToString();

        // // If we don't own the buffer, we can't overwrite it or donate it. For
        // // example we might be pointing to a buffer owned by the client whose
        // // lifetime will not extend past the lifetime of the donated input buffer.
        // if ((!can_donate || (arg && !arg->owns_buffers())) &&
        //     !allocation.is_readonly()) {
        // auto copy = CpuDeviceMemory::CreateDelayedMemory();

        // buffer_alloc_and_copy.src_buffers.push_back(out.CopyRef());
        // buffer_alloc_and_copy.dst_buffers.push_back(copy);
        // buffer_alloc_and_copy.allocation_sizes.push_back(allocation.size());

        // buffer_info.buffer = std::move(copy);
        // buffer_info.owns_buffer = true;
        // buffer_info.buffer_size = allocation.size();
        // return buffer_info;
        // }

        buffer_info.buffer = out.CopyRef();
        buffer_info.owns_buffer = true;
        buffer_info.buffer_size = buffer_size;
        return buffer_info;
    }
    // } else if (allocation.is_constant() && allocation.index() < constants.size()) {
    //     se::DeviceAddressBase constant = constants[allocation.index()].AsDeviceMemoryBase();
    //     buffer_info.buffer = CpuDeviceMemory::CreateConstantMemory(constant.opaque(), constant.size());
    //     buffer_info.owns_buffer = false;
    //     buffer_info.buffer_size = constant.size();
    //     return buffer_info;

    // } else if (allocation.is_constant() || allocation.is_thread_local()) {
    //     buffer_info.buffer = CpuDeviceMemory::CreateConstantMemory(nullptr, 0);
    //     buffer_info.owns_buffer = true;
    //     buffer_info.buffer_size = 0;
    //     return buffer_info;
    // }

    // Output and temporary buffer.
    auto out = NpuDeviceMemory::CreateDelayedMemory();

    buffer_alloc.buffers.push_back(out);
    buffer_alloc.allocation_sizes.push_back(allocation.size());

    buffer_info.buffer = std::move(out);
    buffer_info.owns_buffer = true;
    buffer_info.buffer_size = allocation.size();
    return buffer_info;
}

static absl::StatusOr<std::vector<BufferInfo>> CreateBufferTable(
    const NpuBufferAssignment& assignment,
    // absl::Span<const cpu::ConstantAllocation> constants,
    absl::Span<std::pair<bool, TrackedNpuDeviceBuffer*> const> arguments,
    BufferAlloc& buffer_alloc)
    // BufferAllocAndCopy& buffer_alloc_and_copy,
    // const tsl::AsyncValueRef<NpuDeviceMemory>& tuple_index_table)
{
    std::vector<BufferInfo> buffer_table(assignment.Allocations().size());
    for (BufferAllocation::Index i = 0; i < buffer_table.size(); ++i) {
        const BufferAllocation& allocation = assignment.GetAllocation(i);
        TF_ASSIGN_OR_RETURN(buffer_table[i], MemoryForAllocation(allocation, arguments, buffer_alloc));
            // MemoryForAllocation(allocation, constants, arguments, buffer_alloc,
            //     buffer_alloc_and_copy, tuple_index_table));
    }
    return std::move(buffer_table);
}

static absl::InlinedVector<BufferInfo, 4> CreateResultBufferInfo(
    absl::Span<const BufferAllocation::Index> buffer_indices,
    absl::Span<const BufferInfo> buffer_table)
{
    absl::InlinedVector<BufferInfo, 4> output_buffer_info;
    output_buffer_info.reserve(buffer_indices.size());
    for (int i = 0; i < buffer_indices.size(); ++i) {
        output_buffer_info.push_back(buffer_table[buffer_indices[i]]);
    }
    return output_buffer_info;
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

static void  TestGraph(ge::GraphPtr &graph) {
    graph = std::make_shared<ge::Graph>("XLAGraph");

    // ge::TensorDesc desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_FLOAT);
    // auto data0 = ge::OperatorFactory::CreateOperator("Data_0", "Data");
    // data0.SetAttr("index", 0);
    // data0.UpdateInputDesc("x", desc);
    // data0.UpdateOutputDesc("y", desc);

    // auto add = ge::OperatorFactory::CreateOperator("add", "Add");
    // add.SetInput("x1", data0);
    // add.SetInput("x2", data0);

    // std::vector<ge::Operator> inputs{data0};
    // std::vector<ge::Operator> outputs{add};

    // graph->SetInputs(inputs).SetOutputs(outputs);
    auto module = mlir::air::GraphGenTester::BuildMAIRGeluModul();
    mlir::air::GraphGenTester::Convert(module, graph);
}

static std::string DebugString(const ge::Shape &shape) {
    std::stringstream ss;
    auto dims = shape.GetDims();
    if (dims.empty()) {
        return "empty";
    }
    ss << "[";
    size_t index = 0U;
    for (; index < (dims.size() - 1U); index++) {
    ss << dims[index] << ", ";
    }
    ss << dims[index] << "]";
    return ss.str(); 
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtNpuClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_layout_pointers,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options,
    const GraphData& graph_data,
    const AotCompilationOptions* absl_nullable aot_options) {

  auto input_options = options;

  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());

  if (layout_canonicalization_callback) {
    options.executable_build_options.set_layout_canonicalization_callback(
        layout_canonicalization_callback);
  }

    // int num_replicas;
    // int num_partitions;
    // std::shared_ptr<DeviceAssignment> device_assignment;
    // TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
    //     options.compile_portable_executable, &options.executable_build_options,
    //     [this](int num_replicas, int num_partitions) {
    //         return this->GetDefaultDeviceAssignment(num_replicas, num_partitions);
    //     },
    //     &num_replicas, &num_partitions, &device_assignment));

    ExecutableBuildOptions& build_options = options.executable_build_options;

    TF_ASSIGN_OR_RETURN(ProgramShape program_shape, computation.GetProgramShape());

    //   std::unique_ptr<Executable> cpu_executable;
    ExecutionOptions execution_options =
        CreateExecutionOptions(build_options, &program_shape);

    // Unoptimized HloModuleConfig.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModuleConfig> hlo_module_config,
        CreateModuleConfig(program_shape, argument_layout_pointers,
                            &execution_options, execution_options.num_replicas(),
                            1,
                            aot_options));

    // Unoptimized HloModule.
    const xla::HloModuleProto& hlo_module_proto = computation.proto();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> hlo_module,
        xla::HloModule::CreateFromProto(hlo_module_proto, *hlo_module_config));

  // if (aot_options) {
  //   TF_ASSIGN_OR_RETURN(cpu_executable,
  //                       CompileAheadOfTime(std::move(hlo_module), build_options,
  //                                          execution_options, *aot_options));
  // } else {
  //   xla::Compiler::CompileOptions compile_options{
  //       build_options.device_allocator(), build_options.compile_thread_pool(),
  //       build_options.layout_canonicalization_callback()};
  //   if (!compile_options.thread_pool) {
  //     compile_options.thread_pool = pjrt_client_thread_pool();
  //   }
  //   TF_ASSIGN_OR_RETURN(cpu_executable,
  //                       JitCompile(std::move(hlo_module), build_options,
  //                                  execution_options, compile_options));
  // }
  //
  // auto cpu_executable_ptr =
  //     tsl::down_cast<cpu::CpuExecutable*>(cpu_executable.get());
  //
  // // `result_buffer_indices` has the buffer allocation indices that make up the
  // // output buffer (could be tuple).
  // TF_ASSIGN_OR_RETURN(
  //     auto result_buffer_indices,
  //     FindResultBufferAllocationIndex(cpu_executable_ptr->buffer_assignment(),
  //                                     cpu_executable->module()));
  //

    // Create device assignment(Do not delete!!! step into PjRtNpuExecutable::Execute requires it)
    int num_replicas = 1;
    int num_partitions = 1;
    TF_ASSIGN_OR_RETURN(auto default_device_assignment, GetDefaultDeviceAssignment(num_replicas, num_partitions));
    auto device_assignment = std::make_shared<DeviceAssignment>(default_device_assignment);
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds> addressable_device_logical_ids;
    std::vector<PjRtDevice*> addressable_devices;
    if (device_assignment != nullptr) {
        addressable_device_logical_ids.reserve(num_replicas * num_partitions);
        addressable_devices.reserve(num_replicas * num_partitions);
        for (int replica = 0; replica < num_replicas; ++replica) {
            for (int partition = 0; partition < num_partitions; ++partition) {
                PjRtGlobalDeviceId device_id((*device_assignment)(replica, partition));
                // if (UnpackCpuProcessIndex(device_id) != process_index()) {
                //   VLOG(3) << "Non-local device: " << device_id;
                //   continue;
                // }
                TF_ASSIGN_OR_RETURN(PjRtDevice * device, LookupDevice(device_id));
                PjRtLoadedExecutable::LogicalDeviceIds logica_device_ids;
                logica_device_ids.replica = replica;
                logica_device_ids.partition = partition;
                addressable_device_logical_ids.push_back(std::move(logica_device_ids));
                addressable_devices.push_back(device);
            }
        }
        if (!addressable_devices.empty() && build_options.device_ordinal() < 0) {
            build_options.set_device_ordinal(addressable_devices.front()->local_hardware_id().value());
        }
    }
    // create buffer assignment(Do not delete!!! alloc output buffer depends on it)
    auto buffer_assignment = std::make_unique<NpuBufferAssignment>();
    auto output_shape = graph_data.outputs_shape.back();
    auto output_type = graph_data.output_dtypes.back();
    int64_t size = output_shape.GetShapeSize() * GetSizeByDataType(output_type);
    auto allocation = buffer_assignment->NewEmptyAllocation(size, BufferValue::Color(0));
    allocation->set_entry_computation_parameter(0, ShapeIndex{0}, false);

    TF_ASSIGN_OR_RETURN(
            auto npu_executable,
            NpuExecutable::Create(std::move(buffer_assignment), std::move(hlo_module), nullptr, nullptr, graph_data));

    absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices;
    result_buffer_indices.push_back(0);
    auto executable = std::make_unique<PjRtNpuExecutable>(
            num_replicas, num_partitions, device_assignment,
            options.parameter_is_tupled_arguments, std::move(input_options),
            std::move(npu_executable), std::move(result_buffer_indices),
            std::move(addressable_device_logical_ids), std::move(addressable_devices),
            this, nullptr);

    return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<DeviceAssignment> PjRtNpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
    if (num_partitions * num_replicas <= addressable_devices().size()) {
        xla::DeviceAssignment assignment(num_replicas, num_partitions);
        for (int i = 0; i < num_replicas; ++i) {
            for (int j = 0; j < num_partitions; ++j) {
                assignment(i, j) = addressable_devices().at(i * num_partitions + j)->id();
            }
        }
        return assignment;
    }
    std::cout << "Exception PjRtNpuClient::GetDefaultDeviceAssignment num_replicas: " << num_replicas
        << ", num_partitions: " << num_partitions << std::endl;
    return computation_placer_->AssignDevices(num_replicas, num_partitions);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtNpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options) { 
    tsl::profiler::TraceMe traceme("PjRtNpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options)");
    std::string result;
    llvm::raw_string_ostream os(result);
    module.print(os);
    std::cout << "stablehlo module: " << os.str() << std::endl;

    mlir::ModuleOp mair_module = module.clone();
    xla::mair::Stablehlo2MairConverter(mair_module);

    std::string mair_result;
    llvm::raw_string_ostream mair_os(mair_result);
    mair_module.print(mair_os);
    std::cout << "mair module: " << mair_os.str() << std::endl;

    auto graph_data = GraphData();
    graph_data.graph = std::make_shared<ge::Graph>("XLAGraph");
    mlir::air::GraphGenTester::Convert(mair_module, graph_data.graph);
    // TestGraph(graph_data.graph);
    graph_data.id = 0;
    graph_data.inputs_shape.push_back(ge::Shape({3}));

    std::map<ge::AscendString, ge::AscendString> ge_options = {
        {"ge.exec.deviceId", "0"},
        {"ge.exec.enableDump", "1"},
        {"ge.exec.dumpPath", "/tmp/data_dump"},
        {"ge.exec.dumpMode", "all"}
    };

    if (!ge_initialized_) {    
        ge_initialized_ = true;    
        if (ge::GEInitialize(ge_options) != ge::SUCCESS) {
            return absl::InternalError("Failed to initialize GE.");
        } else {
            (void)ge_options.emplace(ge::AscendString("ge.session_device_id"), ge::AscendString("0"));
            global_ge_session_ = std::make_unique<ge::Session>(ge_options);
            if (global_ge_session_ == nullptr) {
                return absl::InternalError("Failed to create GE session.");
            }
        }
    }

    if (global_ge_session_->AddGraph(graph_data.id, *graph_data.graph, ge_options) != ge::SUCCESS) {
        return absl::InternalError("Failed to AddGraph.");
    }
    // if (global_ge_session_->CompileGraph(graph_data.id) != ge::SUCCESS) {
    //     return absl::InternalError("Failed to CompileGraph.");
    // }
    // std::shared_ptr<ge::CompiledGraphSummary> summary = global_ge_session_->GetCompiledGraphSummary(graph_data.id);
    // if (summary ==  nullptr) {
    //     return absl::InternalError("Failed to GetCompiledGraphSummary.");
    // }
    // summary->GetOutputShapes(graph_data.outputs_shape);
    // summary->GetOutputDtypes(graph_data.output_dtypes);
    graph_data.outputs_shape.push_back(ge::Shape({3}));
    graph_data.output_dtypes.push_back(ge::DT_FLOAT);
    std::cout << "GetCompiledGraphSummary finish,"
        << " output shape is: " << DebugString(graph_data.outputs_shape[0])
        << " output dtype is: " << graph_data.output_dtypes[0]
        << std::endl;

    //构建hlo modules,通过JAX的校验
    //TF_RETURN_IF_ERROR(pjrt::MaybeDumpCompileInputs(options, module, topology_));
    XlaComputation xla_computation;
    ExecutableBuildOptions& exec_build_options = options.executable_build_options;
    TF_RETURN_IF_ERROR(MlirToXlaComputation(
        module, xla_computation,
        /*use_tuple_args=*/options.parameter_is_tupled_arguments,
        /*return_tuple=*/false, &exec_build_options));

    TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> arg_layout_modes,
                      GetArgLayoutModes(module));
    TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> out_layout_modes,
                        GetOutputLayoutModes(module));
    TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> arg_memory_spaces,
                        GetArgMemoryKinds(module));
    TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> out_memory_spaces,
                        GetOutputMemoryKinds(module));
    // If auto-sharding modifies shapes of arguments and/or result,
    // we get a callback to restore the layouts. Let us restore the layouts
    // according to the attributes we parsed from MLIR.
    auto layout_callback = [&arg_layout_modes, &out_layout_modes,
                            &arg_memory_spaces,
                            &out_memory_spaces](const HloModule& module)
        -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
      XlaComputation xla_computation(XlaComputation(module.ToProto()));
      return LayoutModesToXlaShapes(
          xla_computation, arg_layout_modes, out_layout_modes, arg_memory_spaces,
          out_memory_spaces, &LayoutUtil::GetWithDefaultLayout);
    };

    // This call will update result_layout in options.executable_build_options.
    TF_ASSIGN_OR_RETURN(
        auto arg_layouts_and_pointers,
        LayoutModesToXla(xla_computation, arg_layout_modes, out_layout_modes,
                         arg_memory_spaces, out_memory_spaces,
                         &LayoutUtil::GetWithDefaultLayout,
                         options.executable_build_options));
  return CompileInternal(xla_computation, arg_layouts_and_pointers.second,
                         layout_callback, options, graph_data);

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
    RunId run_id(options.launch_id);
    std::cout << "Executing computation " << name() << std::endl;

    // only one device
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> wrapped_results(1);
    if (returned_futures.has_value()) {
        returned_futures->resize(1);
    }

    // Fast-path if there is only one device — run the computation on the
    // current thread.
    auto statusor = ExecuteHelper(
        argument_handles[0], num_replicas_, num_partitions_, run_id, options,
        /*last_collective_launch_event=*/{}, returned_futures.has_value());

    if (!statusor.ok()) {
      return std::move(statusor).status();
    }

    wrapped_results[0] = std::move(statusor->buffers);
    if (returned_futures.has_value()) {
      (*returned_futures)[0] = std::move(*statusor->future);
    }

    return wrapped_results;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtNpuExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
    return absl::UnimplementedError("PjRtNpuExecutable::ExecuteSharded is not supported.");
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtNpuExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
    return absl::UnimplementedError("PjRtNpuExecutable::ExecutePortable is not supported.");
}

absl::StatusOr<PjRtLoadedExecutable::Result> PjRtNpuExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const RunId& run_id, const ExecuteOptions& options,
    PjRtNpuClient::CollectiveLaunchEvent last_collective_launch_event,
    bool fill_future, PjRtNpuDevice* device) const {
    std::shared_ptr<DeviceAssignment> device_assignment;
    if (device == nullptr) {
        // only device 0
        const int64_t device_id = 0;
        PjRtGlobalDeviceId global_device_id(device_id);
        TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device,
                            client_->LookupDevice(global_device_id));
        device = tsl::down_cast<PjRtNpuDevice*>(pjrt_device);
    }

    CHECK_EQ(device->process_index(), client_->process_index());

    // Handle inputs.
    // `execute_event` indicates whether npu computation is complete and whether
    // there was an error.
    auto execute_event = tsl::MakeConstructedAsyncValueRef<NpuEvent>();
    // Asynchronously call needed
    MarkEventReadyOnExit ready_on_exit(execute_event);
    auto execute_usage_event = tsl::MakeRef<NpuTrackedDeviceEvent>(execute_event);

    absl::InlinedVector<std::pair<bool, TrackedNpuDeviceBuffer*>, 4> tracked_buffers;
    tracked_buffers.reserve(argument_handles.size());
    
    // To avoid clobbering inputs, we must ensure that
    //   `extra_deps` = inputs' definition events + donated inputs' usage events.
    // This also ensures that the returned `execute_event` dominates all inputs'
    // events, and thus output buffer only need to contain `execute_event` as the
    // single definition event.
    std::vector<tsl::RCReference<tsl::AsyncValue>> input_deps;
    input_deps.reserve(argument_handles.size());

    for (int i = 0; i < argument_handles.size(); ++i) {
        PjRtBuffer* handle = argument_handles[i];
        auto* npu_buffer = tsl::down_cast<CommonPjRtBuffer*>(handle);
        if (npu_buffer->device() != device) {
            return InvalidArgument(
                "Buffer passed to Execute() as argument %d to replica %d is on "
                "device %s, but replica is assigned to device %s.",
                i, replica, npu_buffer->device()->DebugString(),
                device->DebugString());
        }

        TrackedNpuDeviceBuffer* tracked_buffer;
        auto get_buffer = [&](int i) -> absl::Status {
            CommonPjRtBuffer::ScopedHold usage_transaction =
            npu_buffer->GetBufferWithHold(CommonPjRtBuffer::ScopedHold::kUsage);
            if (!usage_transaction.ok()) {
                return InvalidArgument(
                    "Invalid buffer passed: buffer has been deleted or donated.");
            }
            tracked_buffer = tensorflow::down_cast<TrackedNpuDeviceBuffer*>(
                usage_transaction.buffer());
            usage_transaction.ConvertUsageHold(execute_usage_event);
            tracked_buffers.emplace_back(/*can_donate=*/false, tracked_buffer);
            
            return absl::OkStatus();
        };
        
        TF_RETURN_IF_ERROR(get_buffer(i));

        // Definition events are never modified after buffer construction. If they
        // are available and have no error, they can be skipped in input deps.
        // In contrast, already known errors in the input are taken as deps so that
        // they can poison output buffers.
        const auto& definition_event = tracked_buffer->definition_event();
        if (!definition_event.IsAvailable() || definition_event.IsError()) {
            input_deps.push_back(definition_event.CopyRCRef());
        }
    }

    for (auto& tracked_buffer: tracked_buffers) {
        std::cout << "argument handle buffer addr=" << tracked_buffer.second->buffer().get().untyped_data()
            << ", size="<< tracked_buffer.second->buffer().get().size_bytes() << std::endl;
    }

    if (!input_deps.empty()) {
        std::cout << "input is not ready" << std::endl;
    }

    auto npu_executable = tsl::down_pointer_cast<NpuExecutable>(npu_executable_);
    ge::Session* ge_session = client_->GetGlobalGeSession();
    const GraphData& graph_data = npu_executable->GetGraphData();

    // transfer to ge tensor
    std::vector<ge::Tensor> inputs;
    for (int i = 0; i < tracked_buffers.size(); i++) {
        ge::TensorDesc desc;
        desc.SetOriginFormat(ge::FORMAT_ND);
        desc.SetOriginShape(graph_data.inputs_shape[i]);
        // desc.SetShape(graph_data.inputs_shape[i]);
        desc.SetPlacement(ge::Placement::kPlacementDevice);
        desc.SetFormat(ge::FORMAT_ND);
        desc.SetDataType(ge::DataType::DT_FLOAT);
        ge::Tensor input;
        const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t* data) {};
        input.ResetData(
            (uint8_t*)tracked_buffers[i].second->buffer().get().untyped_data(),
            tracked_buffers[i].second->buffer().get().size_bytes(),
            kDoNothing);
        input.SetShapeDimNum(1);
        input.SetShapeDim(0, 3);

        // ge::Tensor input = ge::Tensor(desc, (uint8_t*)tracked_buffers[i].second->buffer().get().untyped_data(), tracked_buffers[i].second->buffer().get().size_bytes());
        std::cout << "input tensor " << i << " data addr=" << tracked_buffers[i].second->buffer().get().untyped_data()
            << ", data size=" << tracked_buffers[i].second->buffer().get().size_bytes() << std::endl;
        inputs.emplace_back(input);
    }

    /* 运行GE图,使用上面准备好的输出内存,参考cpu的用法使用result_buffers_info */
    std::vector<ge::Tensor> outputs;
    ge::Status ge_status = ge_session->RunGraph(graph_data.id, inputs, outputs);
    if (ge_status != ge::SUCCESS) {
        return absl::InternalError(
            absl::StrCat("run graph id ", graph_data.id, " failed, ge status is ", ge_status));
    }

    /* debug */
    auto data_ptr = outputs[0].GetData();
    auto data_size = outputs[0].GetSize();
    auto dataDesc = outputs[0].GetTensorDesc();
    auto dataShape = dataDesc.GetShape();
    auto dataShapeSize = dataShape.GetShapeSize();
    std::cout << " output shape is: " << DebugString(dataShape) << std::endl;
    std::cout << " output shapeSize is: " << dataShapeSize << std::endl;

    /* prepare output buffer */ 
    // `buffer_alloc` and `buffer_alloc_and_copy` are used to do real memory allocation and copy work.
    absl::InlinedVector<std::pair<bool, TrackedNpuDeviceBuffer*>, 4> output_tracked_buffers;
    output_tracked_buffers.reserve(outputs.size());
    void* output_memory_ptr = NULL;
    auto acl_ret = npu::AclrtMallocHost(&output_memory_ptr, data_size);
    if (acl_ret != ACL_SUCCESS) {
        return absl::InternalError(absl::StrCat("malloc host memory failed, error code: ", acl_ret));
    }
    memcpy(output_memory_ptr, data_ptr, data_size);
    auto deleter = [output_memory_ptr]() {
        npu::AclrtFreeHost(output_memory_ptr);
    };
    tsl::AsyncValueRef<NpuDeviceMemory> output_device_memory = NpuDeviceMemory::CreateForeignMemory(output_memory_ptr, data_size, deleter);
    auto output_tracked_buffer = TrackedNpuDeviceBuffer(
        /*owns_buffers=*/true, std::move(output_device_memory), data_size,
        absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4>{tsl::AsyncValueRef<NpuEvent>()});
    output_tracked_buffers.emplace_back(/*can_donate=*/false, &output_tracked_buffer);
        
    BufferAlloc buffer_alloc; // not used
    // BufferAllocAndCopy buffer_alloc_and_copy;
   
    TF_ASSIGN_OR_RETURN(std::vector<BufferInfo> buffer_table,
        CreateBufferTable(npu_executable->buffer_assignment(), /* npu_executable->constants(), */ output_tracked_buffers, buffer_alloc));
            // buffer_alloc_and_copy,
            // tuple_index_table));
    auto result_buffers_info = CreateResultBufferInfo(result_buffer_indices_, buffer_table);

    // Create output buffers.
    // const Shape& result_shape = npu_executable_->result_shape();
    const Shape& result_shape = ShapeUtil::MakeShape(F32, {3});
    std::vector<std::unique_ptr<PjRtBuffer>> res;
    auto* memory_space = *device->default_memory_space();
    CHECK_EQ(result_buffers_info.size(), 1);
    // Program execution writes to output buffers so it's a definition event.
    auto tracked_device_buffer = std::make_unique<TrackedNpuDeviceBuffer>(
        result_buffers_info[0].owns_buffer,
        std::move(result_buffers_info[0].buffer),
        result_buffers_info[0].buffer_size,
        /*definition_event=*/execute_event);
    auto output_buffer = std::make_unique<CommonPjRtBufferImpl>(
        result_shape, std::move(tracked_device_buffer),
        *device->default_memory_space());
    res.push_back(std::move(output_buffer));

    if (fill_future) {
        auto [promise, future] = Future<>::MakePromise();
        promise.Set(absl::OkStatus());
        // execute_event.AndThen([promise = std::move(promise),
        //                     event = execute_event.CopyRef()]() mutable {
        // if (auto* error = event.GetErrorIfPresent()) {
        //     promise.Set(Internal("Compute error: %s", error->message()));
        // } else {
        //     promise.Set(absl::OkStatus());
        // }
        // });
        return Result({std::move(future), /*buffers=*/std::move(res)});
    }

    return Result({/*future=*/std::nullopt, std::move(res)});
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