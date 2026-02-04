#include "xla/xla_npu/pjrt/npu/npu_client.h"

#include <llvm/IR/DataLayout.h>

#include <functional>
#include <map>
#include <memory>
#include <numeric>

#include "tsl/profiler/lib/traceme.h"
#include "xla/pjrt/dump/dump.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/thread_pool_async_work_runner.h"
#include "xla/pjrt/utils.h"
#include "xla/service/dump.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_npu/backend/cpu_compiler_type.npu_buffer_allocations.h"
#include "xla/xla_npu/backend/cpu_compiler_type.npu_thunk.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/pjrt/npu/abstract_npu_buffer.h"
#include "xla/xla_npu/pjrt/npu/npu_device.h"
#include "xla/xla_npu/pjrt/npu/npu_raw_buffer.h"
#include "xla/xla_npu/pjrt/npu_host_memory_spaces.h"
#include "xla/xla_npu/pjrt/plugin/npu_client_options.h"
#include "xla/xla_npu/service/cpu_compiler_type.npu_compiler.h"
#include "xla/xla_npu/service/cpu_compiler_type.npu_executable.h"

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
            auto status = NpuDeviceMemory::AllocateInto(allocation_sizes[i], buffers[i].AsPtr(), allocator);
            if (!status.ok()) {
                buffers[i].SetError(status);
                return;
            }
            ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(buffers[i]->untyped_data(), allocation_sizes[i]);
        }
    }
};

struct BufferAllocAndCopy {
    // All data members should have the same size.
    absl::InlinedVector<tsl::AsyncValueRef<NpuDeviceMemory>, 4> src_buffers;
    absl::InlinedVector<tsl::AsyncValueRef<NpuDeviceMemory>, 4> dst_buffers;
    absl::InlinedVector<size_t, 4> allocation_sizes;

    void AllocateAndCopy(const NpuDeviceMemory::Allocator& allocator) {
        for (int i = 0; i < src_buffers.size(); ++i) {
            auto status = NpuDeviceMemory::AllocateInto(allocation_sizes[i], dst_buffers[i].AsPtr(), allocator);
            if (!status.ok()) {
                dst_buffers[i].SetError(status);
                return;
            }
            CHECK(src_buffers[i].IsConcrete());
            std::memcpy(dst_buffers[i]->untyped_data(), src_buffers[i]->untyped_data(), allocation_sizes[i]);
        }
    }
};

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
    const BufferAllocation& allocation, absl::Span<const NpuConstantAllocation> constants,
    absl::Span<std::pair<bool, TrackedNpuDeviceBuffer*> const> arguments, BufferAlloc& buffer_alloc,
    BufferAllocAndCopy& buffer_alloc_and_copy, const tsl::AsyncValueRef<NpuDeviceMemory>& tuple_index_table) {
    BufferInfo buffer_info;
    if (allocation.is_entry_computation_parameter()) {
        bool can_donate = false;
        TrackedNpuDeviceBuffer* arg = nullptr;
        size_t buffer_size;
        tsl::AsyncValuePtr<NpuDeviceMemory> out;
        if (tuple_index_table) {
            if (allocation.param_shape_index().empty()) {
                out = tuple_index_table.AsPtr();
                buffer_size = arguments.size() * sizeof(void*);
            } else if (allocation.param_shape_index().size() == 1) {
                std::tie(can_donate, arg) = arguments[allocation.param_shape_index()[0]];
                out = arg->buffer().AsPtr();
                buffer_size = arg->BufferSize();
            } else {
                return absl::InvalidArgumentError(
                    absl::StrCat("Nested tuples are not supported for argument: ", allocation.parameter_number(),
                                 " at shape index:", allocation.param_shape_index().ToString()));
            }
        } else if (!allocation.param_shape_index().empty()) {
            return absl::InvalidArgumentError(
                absl::StrCat("Nested tuples are not supported for argument: ", allocation.parameter_number(),
                             " at shape index:", allocation.param_shape_index().ToString()));
        } else {
            std::tie(can_donate, arg) = arguments[allocation.parameter_number()];
            out = arg->buffer().AsPtr();
            buffer_size = arg->BufferSize();
        }
        CHECK_EQ(allocation.size(), buffer_size) << "Size mismatch on param " << allocation.parameter_number()
                                                 << " at shape index " << allocation.param_shape_index().ToString();

        // If we don't own the buffer, we can't overwrite it or donate it. For
        // example we might be pointing to a buffer owned by the client whose
        // lifetime will not extend past the lifetime of the donated input buffer.
        if ((!can_donate || (arg && !arg->owns_buffers())) && !allocation.is_readonly()) {
            auto copy = NpuDeviceMemory::CreateDelayedMemory();

            buffer_alloc_and_copy.src_buffers.push_back(out.CopyRef());
            buffer_alloc_and_copy.dst_buffers.push_back(copy);
            buffer_alloc_and_copy.allocation_sizes.push_back(allocation.size());

            buffer_info.buffer = std::move(copy);
            buffer_info.owns_buffer = true;
            buffer_info.buffer_size = allocation.size();
            return buffer_info;
        }

        buffer_info.buffer = out.CopyRef();
        buffer_info.owns_buffer = !arg || arg->owns_buffers();
        buffer_info.buffer_size = buffer_size;
        return buffer_info;

    } else if (allocation.is_constant() && allocation.index() < constants.size()) {
        se::DeviceMemoryBase constant = constants[allocation.index()].AsDeviceMemoryBase();
        buffer_info.buffer = NpuDeviceMemory::CreateConstantMemory(constant.opaque(), constant.size());
        buffer_info.owns_buffer = false;
        buffer_info.buffer_size = constant.size();
        return buffer_info;

    } else if (allocation.is_constant() || allocation.is_thread_local()) {
        buffer_info.buffer = NpuDeviceMemory::CreateConstantMemory(nullptr, 0);
        buffer_info.owns_buffer = true;
        buffer_info.buffer_size = 0;
        return buffer_info;
    }

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
    const BufferAssignment& assignment, absl::Span<const NpuConstantAllocation> constants,
    absl::Span<std::pair<bool, TrackedNpuDeviceBuffer*> const> arguments, BufferAlloc& buffer_alloc,
    BufferAllocAndCopy& buffer_alloc_and_copy, const tsl::AsyncValueRef<NpuDeviceMemory>& tuple_index_table) {
    std::vector<BufferInfo> buffer_table(assignment.Allocations().size());
    for (BufferAllocation::Index i = 0; i < buffer_table.size(); ++i) {
        const BufferAllocation& allocation = assignment.GetAllocation(i);
        TF_ASSIGN_OR_RETURN(buffer_table[i], MemoryForAllocation(allocation, constants, arguments, buffer_alloc,
                                                                 buffer_alloc_and_copy, tuple_index_table));
    }
    return std::move(buffer_table);
}

static absl::InlinedVector<BufferInfo, 4> CreateResultBufferInfo(
    absl::Span<const BufferAllocation::Index> buffer_indices, absl::Span<const BufferInfo> buffer_table) {
    absl::InlinedVector<BufferInfo, 4> output_buffer_info;
    output_buffer_info.reserve(buffer_indices.size());
    for (int i = 0; i < buffer_indices.size(); ++i) {
        output_buffer_info.push_back(buffer_table[buffer_indices[i]]);
    }
    return output_buffer_info;
}

// Find the root instruction of the entry computation.
static const InstructionValueSet& GetRootValueSet(const BufferAssignment& assignment, const HloModule& module) {
    return assignment.dataflow_analysis().GetInstructionValueSet(module.entry_computation()->root_instruction());
}

// Buffer table is indexed by buffer allocation indices. The output buffer is
// made up of a subset of those buffer allocations (for tuple, it includes tuple
// index table). This helper finds the buffer allocation indices in buffer
// assignment that make up for the output buffer. It is used by
// CreateResultShapedBuffer to reconstruct the output buffer from the buffer
// table allocated by MemoryForAllocation.
static absl::StatusOr<absl::InlinedVector<BufferAllocation::Index, 4>> FindResultBufferAllocationIndex(
    const BufferAssignment& assignment, const HloModule& module) {
    absl::InlinedVector<BufferAllocation::Index, 4> buffer_indices;
    const InstructionValueSet& root_value_set = GetRootValueSet(assignment, module);
    const Shape& result_shape = module.result_shape();
    if (!result_shape.IsTuple()) {
        // Find the buffer allocation that corresponds to the output buffer.
        const HloValueSet& sources = root_value_set.element({});
        // The points to set is unambiguous so the set should be a singleton.
        CHECK_EQ(1, sources.values().size());
        const HloValue* value_source = sources.values()[0];
        HloInstruction* src = value_source->instruction();
        TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice, assignment.GetUniqueSlice(src, value_source->index()));
        const BufferAllocation::Index buffer_index = slice.index();
        buffer_indices.push_back(buffer_index);
        return {std::move(buffer_indices)};
    }
    buffer_indices.reserve(result_shape.tuple_shapes().size());
    for (int i = 0; i < result_shape.tuple_shapes().size(); ++i) {
        // Find the buffer allocations that corresponds to the output tuple,
        // including the tuple index table.
        const HloValueSet& sources = root_value_set.element({i});
        // The points to set is unambiguous so the set should be a singleton.
        CHECK_EQ(1, sources.values().size());
        const HloValue* value_source = sources.values()[0];
        HloInstruction* src = value_source->instruction();
        TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice, assignment.GetUniqueSlice(src, value_source->index()));
        const BufferAllocation::Index buffer_index = slice.index();
        buffer_indices.push_back(buffer_index);
    }
    return {std::move(buffer_indices)};
}

static absl::StatusOr<std::unique_ptr<xla::Executable>> CompileAheadOfTime(
    std::unique_ptr<HloModule> hlo_module, const ExecutableBuildOptions& build_options,
    const ExecutionOptions& execution_options, const xla::AotCompilationOptions& compile_options) {
    // TODO: support it
    return absl::UnimplementedError("CompileAheadOfTime on npu is not supported.");

    // // TODO: use npu compile
    // cpu::CpuCompiler compiler;
    // // TODO (basioli): honor build_options.run_backend_only() for AOT.
    // // Compile AOT.
    // TF_ASSIGN_OR_RETURN(
    //     std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
    //     compiler.CompileAheadOfTime(std::move(hlo_module), compile_options));

    // if (aot_results.size() != 1) {
    //     return Internal("Expected 1 AOT compilation result, got %d.",
    //                     aot_results.size());
    // }

    // // Technically not needed, but it makes sense so that we know serialization
    // // and deserialization works.
    // TF_ASSIGN_OR_RETURN(std::string serialized_aot_result,
    //                     aot_results[0]->SerializeAsString());
    // TF_ASSIGN_OR_RETURN(std::unique_ptr<AotCompilationResult> aot_result,
    //                     compiler.LoadAotCompilationResult(serialized_aot_result));

    // return std::move(*aot_result).LoadExecutable(&compiler, /*executor=*/nullptr);
}

static absl::StatusOr<std::unique_ptr<xla::Executable>> JitCompile(
    std::unique_ptr<HloModule> hlo_module, const ExecutableBuildOptions& build_options,
    const ExecutionOptions& execution_options, const xla::Compiler::CompileOptions& compile_options) {
    VLOG(3) << "Unoptimized HLO module: " << hlo_module->ToString();
    static constexpr char kBeforeOptimizationsDumpName[] = "before_optimizations";
    DumpHloModuleIfEnabled(*hlo_module, kBeforeOptimizationsDumpName);

    // RunHloPasses and RunBackend both look at the LLVM command line options.
    auto llvm_options =
        llvm_ir::ExtractXlaBackendExtraOptions(hlo_module->config().debug_options().xla_backend_extra_options());
    llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_options);

    // Run Hlo Passes
    NpuCompiler compiler;
    if (!build_options.run_backend_only()) {
        TF_ASSIGN_OR_RETURN(hlo_module,
                            compiler.RunHloPasses(std::move(hlo_module), /*stream_exec=*/nullptr, compile_options));
    }

    // Run backend.
    return compiler.RunBackend(std::move(hlo_module), /*stream_exec=*/nullptr, compile_options);
}

// Returns the NPU devices from the given PjRtCpuDevices.
// Precondition: `devices` doesn't contain nullptr.
static std::vector<NpuTopology::NpuDevice> GetNpuDevices(absl::Span<const std::unique_ptr<PjRtNpuDevice>> devices) {
    std::vector<NpuTopology::NpuDevice> npu_devices;
    npu_devices.reserve(devices.size());
    for (const auto& device : devices) {
        npu_devices.push_back(NpuTopology::NpuDevice{device->process_index(), device->local_hardware_id().value()});
    }
    return npu_devices;
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtNpuClient(NpuClientOptions options) {
    npu::Log::Print(npu::Log::Level::DEBUG, "Start create npu client");

    // aclError aclStatus = npu::aclInit(nullptr);
    // if (aclStatus != ACL_SUCCESS) {
    //     return absl::UnavailableError("ACL init failed.");
    // }
    uint32_t real_device_count = -1;
    aclError aclStatus = npu::aclrtGetDeviceCount(&real_device_count);
    if (aclStatus != ACL_SUCCESS) {
        return absl::UnavailableError("ACL get device count failed.");
    }
    if (real_device_count <= 0) {
        return absl::UnavailableError("Threre is no valid device.");
    }
    npu::Log::Print(npu::Log::Level::DEBUG, "Get npu device count %d", real_device_count);

    // Need at least NpuDeviceCount threads to launch one collective.
    int npu_device_count = options.npu_device_count;
    size_t num_threads = std::max(DefaultThreadPoolSize(), npu_device_count);

    std::vector<std::unique_ptr<PjRtNpuDevice>> devices;
    for (int i = 0; i < std::min(npu_device_count, (int)real_device_count); ++i) {
        aclStatus = npu::aclrtSetDevice(i);
        if (aclStatus != ACL_SUCCESS) {
            return absl::UnavailableError("ACL set device failed.");
        }
        auto device = std::make_unique<PjRtNpuDevice>(options.process_id,
                                                      /*local_device_id=*/i);
        // options.max_inflight_computations_per_device);
        devices.push_back(std::move(device));
    }

    // std::unique_ptr<NpuDeviceMemory::Allocator> allocator =
    //     options.allocator ? std::make_unique<CustomAllocator>(options.allocator)
    //                         : NpuDeviceMemory::MakeDefaultAllocator();

    std::unique_ptr<NpuDeviceMemory::Allocator> allocator = NpuDeviceMemory::MakeDefaultAllocator();

    return std::unique_ptr<PjRtClient>(new PjRtNpuClient(options.process_id, std::move(devices), std::move(allocator),
                                                         // std::move(options.collectives),
                                                         num_threads, options.asynchronous,
                                                         std::move(options.customize_hlo_module_config)));
}

PjRtNpuClient::PjRtNpuClient(int process_index, std::vector<std::unique_ptr<PjRtNpuDevice>> devices,
                             std::shared_ptr<NpuDeviceMemory::Allocator> allocator,
                             // std::shared_ptr<cpu::CpuCollectives> collectives,
                             size_t num_threads, bool asynchronous,
                             std::function<void(HloModuleConfig&)> customize_hlo_module_config)
    : process_index_(process_index),
      owned_devices_(std::move(devices)),
      computation_placer_(std::make_unique<ComputationPlacer>()),
      allocator_(std::move(allocator)),
      //   last_collective_launch_event_(tsl::MakeAvailableAsyncValueRef<NpuEvent>()),
      transpose_cache_(1024),
      //   collectives_(std::move(collectives)),
      topology_(platform_id(), platform_name(), platform_version(), GetNpuDevices(owned_devices_),
                std::vector<std::string>{}),
      asynchronous_(asynchronous),
      customize_hlo_module_config_(std::move(customize_hlo_module_config)),
      //   eigen_intraop_pool_(new tsl::thread::ThreadPool(
      //       tsl::Env::Default(), GetThreadOptions(), "XLAEigen",
      //       std::min(num_threads, kMaxIntraOpThreads))),
      //   eigen_intraop_device_(
      //       new Eigen::ThreadPoolDevice(eigen_intraop_pool_->AsEigenThreadPool(),
      //       eigen_intraop_pool_->NumThreads())),
      pjrt_client_thread_pool_(
          new tsl::thread::ThreadPool(tsl::Env::Default(), GetThreadOptions(), "XLAPjRtNpuClient", num_threads)),
      async_work_runner_(MakeThreadPoolAsyncWorkRunner(pjrt_client_thread_pool_.get())) {
    for (const std::unique_ptr<PjRtNpuDevice>& device : owned_devices_) {
        devices_.push_back(device.get());
        CHECK(id_to_device_.insert({device->global_device_id(), device.get()}).second)
            << "Duplicate device id: " << device->global_device_id();

        device->SetClient(this);
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
        // PjRtNpuDevice, so attach the device memory space first.
        auto npu_device_memory_space = std::make_unique<NpuDeviceMemorySpace>(id * 3 + 0, device);
        npu_device->AttachMemorySpace(npu_device_memory_space.get());
        memory_spaces_.push_back(npu_device_memory_space.get());
        owned_memory_spaces_.push_back(std::move(npu_device_memory_space));

        auto pinned_memory_space = std::make_unique<PinnedHostMemorySpace>(id * 3 + 1, device);
        npu_device->AttachMemorySpace(pinned_memory_space.get());
        memory_spaces_.push_back(pinned_memory_space.get());
        owned_memory_spaces_.push_back(std::move(pinned_memory_space));

        auto unpinned_memory_space = std::make_unique<UnpinnedHostMemorySpace>(id * 3 + 2, device);
        npu_device->AttachMemorySpace(unpinned_memory_space.get());
        memory_spaces_.push_back(unpinned_memory_space.get());
        owned_memory_spaces_.push_back(std::move(unpinned_memory_space));
    }

    npu::Log::Print(npu::Log::Level::DEBUG, "PjRtNpuClient created.");
}

PjRtNpuClient::~PjRtNpuClient() {
    auto ret = npu::aclFinalize();
    if (ret != ACL_SUCCESS) {
        npu::Log::Print(npu::Log::Level::ERROR, "Acl finalize failed, error code %d", ret);
    } else {
        npu::Log::Print(npu::Log::Level::DEBUG, "Acl finalize success");
    }
}

absl::StatusOr<PjRtDevice*> PjRtNpuClient::LookupDevice(PjRtGlobalDeviceId global_device_id) const {
    auto it = id_to_device_.find(global_device_id);
    if (it != id_to_device_.end()) {
        return it->second;
    }
    return InvalidArgument("No matching device found for device_id %d", global_device_id.value());
}

absl::StatusOr<PjRtDevice*> PjRtNpuClient::LookupAddressableDevice(PjRtLocalDeviceId local_device_id) const {
    for (auto* device : addressable_devices_) {
        if (local_device_id == device->local_device_id()) {
            return device;
        }
    }
    return InvalidArgument("No matching device found for local_device_id %d", local_device_id.value());
}

absl::Span<PjRtMemorySpace* const> PjRtNpuClient::memory_spaces() const { return memory_spaces_; }

absl::StatusOr<int64_t> PjRtNpuClient::GetOnDeviceBytesCount(PjRtMemorySpace* memory_space, const Shape& shape) const {
    auto pointer_size = llvm::DataLayout(DataLayout()).getPointerSize(0 /* default address space */);
    if (shape.IsTuple() || shape.is_static()) {
        return ShapeUtil::ByteSizeOf(shape, pointer_size);
    }
    int64_t metadata_size = sizeof(int32_t) * shape.dimensions().size();
    return ShapeUtil::ByteSizeOf(shape, pointer_size) + metadata_size;
}

absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>> PjRtNpuClient::AllocateRawBuffer(
    PjRtMemorySpace* memory_space, size_t on_device_bytes_count, bool retry_on_oom,
    tsl::AsyncValueRef<bool> allocate_after) {
    return xla::NpuRawBuffer::Allocate(memory_space, on_device_bytes_count);
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> PjRtNpuClient::LinearizeHostBufferInto(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides, HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer, const xla::Shape& device_shape,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) {
    return tsl::down_cast<NpuRawBuffer*>(raw_buffer.get())
        ->CopyFromHostBuffer(data, type, dims, byte_strides, host_buffer_semantics, std::move(on_done_with_host_buffer),
                             device_shape, async_work_runner(), &transpose_mu_, &transpose_cache_);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> PjRtNpuClient::DefineBuffer(
    const Shape& on_device_shape, tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
    absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4> definition_device_events, bool raw_buffer_is_mutable) {
    absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4> definition_events;
    for (auto& ev : definition_device_events) {
        definition_events.push_back(tsl::down_cast<NpuTrackedDeviceEvent*>(ev.get())->event());
    }
    return std::unique_ptr<PjRtBuffer>(std::make_unique<CommonPjRtBufferImpl>(
        on_device_shape,
        std::make_unique<TrackedNpuDeviceBuffer>(
            /*owns_buffers=*/raw_buffer_is_mutable, tsl::down_cast<NpuRawBuffer*>(raw_buffer.get())->buffer(),
            ShapeUtil::ByteSizeOf(on_device_shape), std::move(definition_events)),
        raw_buffer->memory_space()));
}

absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>, tsl::RCReference<PjRtDeviceEvent>>>
PjRtNpuClient::CreateLinkedEventPromise(PjRtMemorySpace* memory_space, absl::string_view debug_info) {
    auto definition_event_promise = tsl::MakeIndirectAsyncValue();
    auto definition_event = tsl::MakeRef<NpuTrackedDeviceEvent>(tsl::AsyncValueRef<NpuEvent>(definition_event_promise));
    return std::make_pair(tsl::MakeRef<NpuTrackedDeviceEventPromise>(std::move(definition_event_promise)),
                          std::move(definition_event));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> PjRtNpuClient::CompileInternal(
    const XlaComputation& computation, const std::vector<const Shape*>& argument_layout_pointers,
    LayoutCanonicalizationCallback layout_canonicalization_callback, CompileOptions options,
    const AotCompilationOptions* absl_nullable aot_options) {
    tsl::profiler::TraceMe traceme("PjRtNpuClient::Compile");
    auto input_options = options;

    TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());

    if (layout_canonicalization_callback) {
        options.executable_build_options.set_layout_canonicalization_callback(layout_canonicalization_callback);
    }

    int num_replicas;
    int num_partitions;
    std::shared_ptr<DeviceAssignment> device_assignment;
    TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
        options.compile_portable_executable, &options.executable_build_options,
        [this](int num_replicas, int num_partitions) {
            return this->GetDefaultDeviceAssignment(num_replicas, num_partitions);
        },
        &num_replicas, &num_partitions, &device_assignment));

    // if (collectives_ == nullptr && device_assignment) {
    //     for (int replica = 0; replica < device_assignment->replica_count(); ++replica) {
    //         for (int computation = 0; computation < device_assignment->computation_count(); ++computation) {
    //             PjRtGlobalDeviceId id((*device_assignment)(replica, computation));
    //             if (UnpackCpuProcessIndex(id) != process_index()) {
    //                 // TODO(phawkins): improve this error message when we're ready to
    //                 // publicize that multiprocess collectives exist.
    //                 return InvalidArgument("Multiprocess computations aren't implemented on the CPU backend.");
    //             }
    //         }
    //     }
    // }

    ExecutableBuildOptions& build_options = options.executable_build_options;
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds> addressable_device_logical_ids;
    std::vector<PjRtDevice*> addressable_devices;
    if (device_assignment != nullptr) {
        addressable_device_logical_ids.reserve(num_replicas * num_partitions);
        addressable_devices.reserve(num_replicas * num_partitions);
        for (int replica = 0; replica < num_replicas; ++replica) {
            for (int partition = 0; partition < num_partitions; ++partition) {
                PjRtGlobalDeviceId device_id((*device_assignment)(replica, partition));
                // if (UnpackCpuProcessIndex(device_id) != process_index()) {
                //     VLOG(3) << "Non-local device: " << device_id;
                //     continue;
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

    TF_ASSIGN_OR_RETURN(ProgramShape program_shape, computation.GetProgramShape());

    std::unique_ptr<Executable> npu_executable;
    ExecutionOptions execution_options = CreateExecutionOptions(build_options, &program_shape);

    // Unoptimized HloModuleConfig.
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModuleConfig> hlo_module_config,
                        CreateModuleConfig(program_shape, argument_layout_pointers, &execution_options,
                                           execution_options.num_replicas(),
                                           // eigen_intraop_device()->getPool()->NumThreads(),
                                           1, aot_options));

    // // Apply the user-provided callback to customize the HloModuleConfig.
    // if (customize_hlo_module_config_) {
    //     customize_hlo_module_config_(*hlo_module_config);
    // }

    // Unoptimized HloModule.
    const xla::HloModuleProto& hlo_module_proto = computation.proto();
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                        xla::HloModule::CreateFromProto(hlo_module_proto, *hlo_module_config));

    NpuTool::PrintHloModule(hlo_module.get(), "xla hlo module");

    // hlo instruction to ptx
    if (aot_options) {
        TF_ASSIGN_OR_RETURN(npu_executable,
                            CompileAheadOfTime(std::move(hlo_module), build_options, execution_options, *aot_options));
    } else {
        xla::Compiler::CompileOptions compile_options{build_options.device_allocator(),
                                                      build_options.compile_thread_pool(),
                                                      build_options.layout_canonicalization_callback()};
        if (!compile_options.thread_pool) {
            compile_options.thread_pool = pjrt_client_thread_pool();
        }
        TF_ASSIGN_OR_RETURN(npu_executable,
                            JitCompile(std::move(hlo_module), build_options, execution_options, compile_options));
    }

    auto npu_executable_ptr = tsl::down_cast<NpuExecutable*>(npu_executable.get());

    // `result_buffer_indices` has the buffer allocation indices that make up the
    // output buffer (could be tuple).
    TF_ASSIGN_OR_RETURN(
        auto result_buffer_indices,
        FindResultBufferAllocationIndex(npu_executable_ptr->buffer_assignment(), npu_executable_ptr->module()));

    std::unique_ptr<HloModule> unoptimized_hlo_module = nullptr;

    // const bool xla_dump_hlo_unoptimized_snapshots =
    //     options.executable_build_options.has_debug_options() &&
    //     options.executable_build_options.debug_options()
    //         .xla_dump_hlo_unoptimized_snapshots();

    // if (xla_dump_hlo_unoptimized_snapshots) {
    //     TF_ASSIGN_OR_RETURN(
    //         unoptimized_hlo_module,
    //         HloModule::CreateFromProto(computation.proto(),
    //                                 npu_executable->module().config()));
    // }

    auto executable = std::make_unique<PjRtNpuExecutable>(
        num_replicas, num_partitions, std::move(device_assignment), options.parameter_is_tupled_arguments,
        std::move(input_options), std::move(npu_executable), std::move(result_buffer_indices),
        std::move(addressable_device_logical_ids), std::move(addressable_devices), this,
        std::move(unoptimized_hlo_module));
    // TF_RETURN_IF_ERROR(
    //     executable->SetUpDonation(options.parameter_is_tupled_arguments));

    return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<DeviceAssignment> PjRtNpuClient::GetDefaultDeviceAssignment(int num_replicas, int num_partitions) const {
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

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> PjRtNpuClient::CompileAndLoad(mlir::ModuleOp module,
                                                                                    CompileOptions options) {
    tsl::profiler::TraceMe traceme("PjRtNpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options)");

    NpuTool::PrintModuleOp(module, "stablehlo module");

    TF_RETURN_IF_ERROR(pjrt::MaybeDumpCompileInputs(options, module, topology_));

    XlaComputation xla_computation;
    ExecutableBuildOptions& exec_build_options = options.executable_build_options;
    // chlo => mhlo => xla hlo
    TF_RETURN_IF_ERROR(MlirToXlaComputation(module, xla_computation,
                                            /*use_tuple_args=*/options.parameter_is_tupled_arguments,
                                            /*return_tuple=*/false, &exec_build_options));

    // if (options.argument_layouts) {
    //     return CompileAndLoad(xla_computation, options);
    // }

    TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> arg_layout_modes, GetArgLayoutModes(module));
    TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> out_layout_modes, GetOutputLayoutModes(module));
    TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> arg_memory_spaces, GetArgMemoryKinds(module));
    TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> out_memory_spaces, GetOutputMemoryKinds(module));

    // If auto-sharding modifies shapes of arguments and/or result,
    // we get a callback to restore the layouts. Let us restore the layouts
    // according to the attributes we parsed from MLIR.
    auto layout_callback = [&arg_layout_modes, &out_layout_modes, &arg_memory_spaces, &out_memory_spaces](
                               const HloModule& module) -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
        XlaComputation xla_computation(XlaComputation(module.ToProto()));
        return LayoutModesToXlaShapes(xla_computation, arg_layout_modes, out_layout_modes, arg_memory_spaces,
                                      out_memory_spaces, &LayoutUtil::GetWithDefaultLayout);
    };

    // This call will update result_layout in options.executable_build_options.
    TF_ASSIGN_OR_RETURN(
        auto arg_layouts_and_pointers,
        LayoutModesToXla(xla_computation, arg_layout_modes, out_layout_modes, arg_memory_spaces, out_memory_spaces,
                         &LayoutUtil::GetWithDefaultLayout, options.executable_build_options));

    return CompileInternal(xla_computation, arg_layouts_and_pointers.second, layout_callback, options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> PjRtNpuClient::CompileAndLoad(const XlaComputation& computation,
                                                                                    CompileOptions options) {
    return absl::UnimplementedError("CompileAndLoad with XlaComputation is not supported.");
}

PjRtNpuExecutable::PjRtNpuExecutable(int num_replicas, int num_partitions,
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
      addressable_device_logical_ids_(std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)),
      unoptimized_hlo_module_(std::move(unoptimized_hlo_module)) {}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> PjRtNpuExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles, const ExecuteOptions& options,
    std::optional<std::vector<Future<>>>& returned_futures) const {
    RunId run_id(options.launch_id);
    tsl::profiler::TraceMe trace_me("PjRtNpuExecutable::Execute");
    if (device_assignment_ == nullptr) {
        return InvalidArgument("Execute expects a non-null device_assignment");
    }
    const int num_addressable_devices = addressable_devices_.size();

    if (argument_handles.size() != num_addressable_devices) {
        return InvalidArgument(
            "Attempted to execute with %d argument lists when local device "
            "count is %d (total replica count: %d, partition count: %d)",
            argument_handles.size(), num_addressable_devices, num_replicas(), num_partitions());
    }

    VLOG(1) << "Executing computation " << name() << "; num_replicas=" << num_replicas()
            << " num_partitions=" << num_partitions() << " num_addressable_devices=" << num_addressable_devices;

    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> wrapped_results(num_addressable_devices);
    if (returned_futures.has_value()) {
        returned_futures->resize(num_addressable_devices);
    }
    if (num_addressable_devices == 1) {
        // Fast-path if there is only one device — run the computation on the
        // current thread.
        const int replica = addressable_device_logical_ids_[0].replica;
        const int partition = addressable_device_logical_ids_[0].partition;

        // // Dump once before running, in case there's a crash.
        // MaybeDumpHloSnapshot(npu_executable_->module(), run_id, argument_handles[0], {});
        // if (unoptimized_hlo_module_ != nullptr) {
        //     HloUnoptimizedSnapshot hlo_snapshot;
        //     *hlo_snapshot.mutable_hlo_module() = unoptimized_hlo_module_->ToProto();
        //     for (const auto& argument_handle : argument_handles) {
        //         HloInputs hlo_inputs;
        //         for (const auto& buffer : argument_handle) {
        //         TF_ASSIGN_OR_RETURN(auto literal, buffer->ToLiteralSync());
        //         *hlo_inputs.add_arguments() = literal->ToProto();
        //         }
        //         *hlo_snapshot.add_partitions() = std::move(hlo_inputs);
        // }

        // DumpHloUnoptimizedSnapshotIfEnabled(
        //     hlo_snapshot, npu_executable_->module().config().debug_options());
        // }
        auto statusor = ExecuteHelper(argument_handles[0], replica, partition, run_id, options,
                                      /*last_collective_launch_event=*/{}, returned_futures.has_value());

        if (!statusor.ok()) {
            return std::move(statusor).status();
        }

        wrapped_results[0] = std::move(statusor->buffers);
        if (returned_futures.has_value()) {
            (*returned_futures)[0] = std::move(*statusor->future);
        }

        // MaybeDumpHloSnapshot(npu_executable_->module(), run_id, argument_handles[0], wrapped_results[0]);
    } else {
        return absl::UnimplementedError("PjRtNpuExecutable::device more than one is not supported.");
    }

    VLOG(1) << "Replicated execution complete.";

    return wrapped_results;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> PjRtNpuExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device, const ExecuteOptions& options,
    std::optional<Future<>>& returned_future, bool fill_future) const {
    return absl::UnimplementedError("PjRtNpuExecutable::ExecuteSharded is not supported.");
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> PjRtNpuExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device, const ExecuteOptions& options,
    std::optional<Future<>>& returned_future, bool fill_future) const {
    return absl::UnimplementedError("PjRtNpuExecutable::ExecutePortable is not supported.");
}

absl::StatusOr<PjRtLoadedExecutable::Result> PjRtNpuExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition, const RunId& run_id,
    const ExecuteOptions& options, PjRtNpuClient::CollectiveLaunchEvent last_collective_launch_event, bool fill_future,
    PjRtNpuDevice* device) const {
    tsl::profiler::TraceMe traceme([&]() {
        return tsl::profiler::TraceMeEncode("PjRtNpuExecutable::ExecuteHelper", {
                                                                                    {"run_id", run_id.ToInt()},
                                                                                    {"replica", replica},
                                                                                    {"partition", partition},
                                                                                });
    });

    std::shared_ptr<DeviceAssignment> device_assignment;
    if (device == nullptr) {
        CHECK(device_assignment_ != nullptr);
        // const int64_t device_id = (*device_assignment_)(replica, partition);
        // only device 0
        const int64_t device_id = 0;
        PjRtGlobalDeviceId global_device_id(device_id);
        TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device, client_->LookupDevice(global_device_id));
        device = tsl::down_cast<PjRtNpuDevice*>(pjrt_device);
        device_assignment = device_assignment_;
    } else {
        CHECK(device_assignment_ == nullptr);
        CHECK_EQ(replica, 0);
        CHECK_EQ(partition, 0);
        CHECK(addressable_devices_.empty());
        device_assignment = std::make_shared<DeviceAssignment>(1, 1);
        (*device_assignment)(0, 0) = device->id();
    }

    CHECK_EQ(device->process_index(), client_->process_index());

    // Handle inputs.
    // `execute_event` indicates whether npu computation is complete and whether
    // there was an error.
    auto execute_event = tsl::MakeConstructedAsyncValueRef<NpuEvent>();
    // Asynchronously call needed
    MarkEventReadyOnExit ready_on_exit(execute_event);
    auto execute_usage_event = tsl::MakeRef<NpuTrackedDeviceEvent>(execute_event);

    // absl::InlinedVector<CommonPjRtBuffer::ScopedHold, 4> donation_transactions;

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
                i, replica, npu_buffer->device()->DebugString(), device->DebugString());
        }

        TrackedNpuDeviceBuffer* tracked_buffer;
        auto get_buffer = [&](int i) -> absl::Status {
            // bool must_donate = donate_it != parameters_that_must_be_donated_.end() &&
            //                     *donate_it == i &&
            //                     !options.non_donatable_input_indices.contains(i);
            // TF_RETURN_IF_ERROR(TestBufferDonationClashes(
            //     cpu_buffer, donation_clashes, must_donate, i, replica, partition));
            // if (must_donate) {
            //     ++donate_it;
            //     CommonPjRtBuffer::ScopedHold donation_transaction =
            //         cpu_buffer->GetBufferWithHold(
            //             CommonPjRtBuffer::ScopedHold::kDonation);
            //     // On CPU, we allow donation to succeed by introducing a copy. This was
            //     // added when enabling buffer donation on CPU since it turned out that a
            //     // number of users were holding external references to buffers that were
            //     // supposed to be donated. We may wish to tighten those semantics in the
            //     // future.
            //     if (donation_transaction.ok()) {
            //     tracked_buffer = tensorflow::down_cast<TrackedCpuDeviceBuffer*>(
            //         donation_transaction.buffer());
            //     // After acquiring the buffer for donation, we retrieve the dependent
            //     // usage events. Note that we don't need any locking here as
            //     // AcquireDonation() is supposed to synchronize with other usages.
            //     for (const auto& ev : tracked_buffer->UsageEvents()) {
            //         if (!ev.IsAvailable()) {
            //         input_deps.push_back(ev.CopyRCRef());
            //         }
            //     }
            //     tracked_buffers.emplace_back(/*can_donate=*/true, tracked_buffer);
            //     donation_transactions.push_back(std::move(donation_transaction));
            //     return absl::OkStatus();
            //     }
            // }

            CommonPjRtBuffer::ScopedHold usage_transaction =
                npu_buffer->GetBufferWithHold(CommonPjRtBuffer::ScopedHold::kUsage);
            if (!usage_transaction.ok()) {
                return InvalidArgument("Invalid buffer passed: buffer has been deleted or donated.");
            }
            tracked_buffer = tensorflow::down_cast<TrackedNpuDeviceBuffer*>(usage_transaction.buffer());
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

    // TF_RETURN_IF_ERROR(CheckBufferCompatibilities(tracked_buffers));

    // Tuplize the inputs if compiler expects a single tuple argument but runtime
    // gets many inputs that are not yet tupled.
    tsl::AsyncValueRef<NpuDeviceMemory> tuple_index_table;
    // if (parameter_is_tupled_arguments_) {
    //     absl::InlinedVector<tsl::AsyncValueRef<CpuDeviceMemory>, 4> leaf_buffers;
    //     leaf_buffers.reserve(tracked_buffers.size());
    //     for (const auto& tracked_buffer : tracked_buffers) {
    //     leaf_buffers.push_back(tracked_buffer.second->buffer());
    //     }
    //     tuple_index_table = CpuDeviceMemory::CreateDelayedMemory();
    //     tsl::RunWhenReady(
    //         absl::MakeConstSpan(leaf_buffers),
    //         [buffers = leaf_buffers, tuple_index_table,
    //         allocator = client()->allocator()]() mutable {
    //         size_t index_table_byte_size = buffers.size() * sizeof(void*);
    //         // We assume tuple table allocations will not fail.
    //         CHECK_OK(CpuDeviceMemory::AllocateInto(
    //             index_table_byte_size, tuple_index_table.AsPtr(), *allocator));
    //         uintptr_t* index_table =
    //             reinterpret_cast<uintptr_t*>(tuple_index_table->untyped_data());
    //         for (int i = 0; i < buffers.size(); ++i) {
    //             index_table[i] =
    //                 absl::bit_cast<uintptr_t>(buffers[i]->untyped_data());
    //         }
    //         });
    // }

    for (auto& tracked_buffer : tracked_buffers) {
        npu::Log::Print(npu::Log::Level::DEBUG, "show argument handle buffer, addr %p, size %d",
                        tracked_buffer.second->buffer().get().untyped_data(),
                        tracked_buffer.second->buffer().get().size_bytes());
    }

    auto npu_executable = tsl::down_pointer_cast<NpuExecutable>(npu_executable_);
    // `buffer_alloc` and `buffer_alloc_and_copy` are used to do real memory
    // allocation and copy work.
    BufferAlloc buffer_alloc;
    BufferAllocAndCopy buffer_alloc_and_copy;
    TF_ASSIGN_OR_RETURN(std::vector<BufferInfo> buffer_table,
                        CreateBufferTable(npu_executable->buffer_assignment(), npu_executable->constants(),
                                          tracked_buffers, buffer_alloc, buffer_alloc_and_copy, tuple_index_table));
    auto result_buffers_info = CreateResultBufferInfo(result_buffer_indices_, buffer_table);

    // // The choice of where we wait is arbitrary; the reason for the wait is
    // // pacing to avoid problems such as memory fragmentation and running ahead
    // // too far, not for correctness. Placing it before the executable launch
    // // allows the inputs for the next executable to be fetched even if the
    // // launch is delayed.
    // auto compute_reservation = std::make_unique<Semaphore::ScopedReservation>(
    //     device->max_inflight_computations_semaphore().ScopedAcquire(1));

    ExecutableRunOptions run_options;
    run_options.set_run_id(run_id);
    // Need to keep device_assignment alive until execution completes.
    run_options.set_device_assignment(device_assignment.get());
    // run_options.set_intra_op_thread_pool(client_->eigen_intraop_device());

    // auto cpu_run_options = std::make_shared<cpu::CpuExecutableRunOptions>();
    // run_options.set_cpu_executable_run_options(cpu_run_options.get());

    // const CpuExecuteContext* cpu_execute_context =
    //     options.context == nullptr
    //         ? nullptr
    //         : dynamic_cast<const CpuExecuteContext*>(options.context);
    // if (cpu_execute_context != nullptr && cpu_execute_context->process_index().has_value()) {
    //     run_options.set_device_ordinal(
    //         PackCpuDeviceId(*cpu_execute_context->process_index(),
    //                         UnpackCpuLocalDeviceId(device->global_device_id())).value());
    // } else {
    run_options.set_device_ordinal(device->global_device_id().value());
    // }
    // if (cpu_execute_context != nullptr && cpu_execute_context->collectives() != nullptr) {
    //     cpu_run_options->set_collectives(cpu_execute_context->collectives());
    // } else {
    // cpu_run_options->set_collectives(client_->collectives_.get());
    // }

    // // Schedule only one collective at a time.
    // bool is_a_collective_launch = static_cast<bool>(last_collective_launch_event.first);
    // // Add additional dependency conditioned on whether this is a collective
    // // launch or not.
    // if (is_a_collective_launch) {
    //     input_deps.push_back(std::move(last_collective_launch_event.first));
    // } else {
    //     // This is a non-parallel computation. Add the last enqueue event as a
    //     // dependency with any error cleared.
    //     auto last_enqueue_event = device->stream_event_map()->GetLastEnqueueEvent(options.execution_stream_id);
    //     if (!last_enqueue_event.IsAvailable()) {
    //         auto last_enqueue_done_event = tsl::MakeUnconstructedAsyncValueRef<CpuEvent>();
    //         last_enqueue_event.AndThen(
    //             [last_enqueue_done_event = last_enqueue_done_event.CopyRef()]() {
    //                 last_enqueue_done_event.emplace();
    //             });
    //         input_deps.push_back(std::move(last_enqueue_done_event));
    //     }
    // }
    // if (options.context != nullptr) {
    //     run_options.set_ffi_execution_context(&options.context->ffi_context());
    // }

    bool execute_inline = true;
    // bool execute_inline = cheap_computation_ || !client_->asynchronous_ || ThisThreadIsInsideHostCallback();

    // // Overwrite `execute_inline` if it is specified in the ExecuteOptions.
    // if (options.execution_mode == ExecuteOptions::ExecutionMode::kAsynchronous) {
    //     execute_inline = false;
    // } else if (options.execution_mode == ExecuteOptions::ExecutionMode::kSynchronous) {
    //     execute_inline = true;
    // }

    if (input_deps.empty() && execute_inline) {
        // Synchronously call generated function or thunk sequence.

        // // Set denormal and rounding behavior to match the default TF
        // // ThreadPool behavior.
        // tsl::port::ScopedFlushDenormal flush;
        // tsl::port::ScopedSetRound round(FE_TONEAREST);

        // Execution status for npu thunk runtime.
        tsl::AsyncValueRef<NpuThunk::ExecuteEvent> thunks_execute_event;

        // Immediately allocate memory and prepare for computation.
        buffer_alloc.Allocate(*client()->allocator());
        buffer_alloc_and_copy.AllocateAndCopy(*client()->allocator());
        for (const auto& buffer_info : buffer_table) {
            CHECK(buffer_info.buffer.IsAvailable());
            if (buffer_info.buffer.IsError()) {
                return buffer_info.buffer.GetError();
            }
        }

        if (npu_executable->has_thunks()) {
            // Call interpreted thunk sequence implementing XLA executable.
            absl::InlinedVector<MaybeOwningDeviceMemory, 8> buffer_device_mem;
            buffer_device_mem.reserve(buffer_table.size());
            for (const auto& buffer_info : buffer_table) {
                buffer_device_mem.emplace_back(
                    se::DeviceMemoryBase(buffer_info.buffer->untyped_data(), buffer_info.buffer->size_bytes()));
            }

            NpuBufferAllocations allocations(buffer_device_mem);

            // TF_ASSIGN_OR_RETURN(NpuThunk::CollectiveExecuteParams collective_params,
            //                     NpuThunk::CollectiveExecuteParams::Create(&run_options));

            TF_ASSIGN_OR_RETURN(NpuThunk::CustomCallExecuteParams custom_call_execute_params,
                                NpuThunk::CustomCallExecuteParams::Create(&run_options));

            // std::optional<NpuThunk::XnnParams> xnn_params;
            // if (npu_executable->has_xnn_fusions()) {
            //     TF_ASSIGN_OR_RETURN(xnn_params, NpuThunk::XnnParams::Create(&run_options));
            // }

            // std::optional<NpuThunk::YnnParams> ynn_params;
            // if (npu_executable->has_ynn_fusions()) {
            //     TF_ASSIGN_OR_RETURN(ynn_params, NpuThunk::YnnParams::Create(&run_options));
            // }

            // cpu::ThreadPoolTaskRunner task_runner(run_options.intra_op_thread_pool()->getPool()); // TODO 实现

            NpuThunk::ExecuteParams execute_params = {
                nullptr,
                // npu_executable->function_library(),
                &allocations,
                // cpu::GetXfeedManager(run_options.device_ordinal()),
                run_options.intra_op_thread_pool(),
                // &task_runner,
                nullptr,
                // &collective_params,
                &custom_call_execute_params,
                // xnn_params ? &*xnn_params : nullptr,
                // ynn_params ? &*ynn_params : nullptr,
                run_options.run_id().ToInt(),
                run_options.device_ordinal(),
            };

            thunks_execute_event = npu_executable->thunks().Execute(execute_params);

            tsl::profiler::TraceMe trace([&] {
                return tsl::profiler::TraceMeEncode(
                    "ThunkExecutor::Execute (wait for completion)",
                    {{"run_id", run_options.run_id().ToInt()}, {"device_ordinal", run_options.device_ordinal()}});
            });
            tsl::BlockUntilReady(thunks_execute_event);

        } else {
            return Internal("NpuExecutable has no thunks.");
        }

        // for (auto& donation_transaction : donation_transactions) {
        //     std::move(donation_transaction).ConfirmDonation();
        // }

        if (thunks_execute_event.IsError()) {
            return thunks_execute_event.GetError();
        }

    } else {
        return absl::UnimplementedError("PjRtNpuExecutable::ExecuteHelper: not support execute async now");
    }

    // Create output buffers.
    const Shape& result_shape = npu_executable_->result_shape();
    std::vector<std::unique_ptr<PjRtBuffer>> res;
    if (result_shape.IsTuple()) {
        res.reserve(result_buffers_info.size());
        for (int i = 0; i < result_buffers_info.size(); ++i) {
            // Program execution writes to output buffers so it's a definition event.
            absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4> definition_events;
            definition_events.push_back(execute_event.CopyRef());
            auto leaf_tracked_device_buffer = std::make_unique<TrackedNpuDeviceBuffer>(
                result_buffers_info[i].owns_buffer, std::move(result_buffers_info[i].buffer),
                result_buffers_info[i].buffer_size, std::move(definition_events));
            auto leaf_buffer = std::make_unique<CommonPjRtBufferImpl>(
                result_shape.tuple_shapes(i), std::move(leaf_tracked_device_buffer), *device->default_memory_space());
            res.push_back(std::move(leaf_buffer));
        }
    } else {
        CHECK_EQ(result_buffers_info.size(), 1);
        // Program execution writes to output buffers so it's a definition event.
        auto tracked_device_buffer = std::make_unique<TrackedNpuDeviceBuffer>(result_buffers_info[0].owns_buffer,
                                                                              std::move(result_buffers_info[0].buffer),
                                                                              result_buffers_info[0].buffer_size,
                                                                              /*definition_event=*/execute_event);
        auto output_buffer = std::make_unique<CommonPjRtBufferImpl>(result_shape, std::move(tracked_device_buffer),
                                                                    *device->default_memory_space());
        res.push_back(std::move(output_buffer));
    }

    if (fill_future) {
        auto [promise, future] = Future<>::MakePromise();
        execute_event.AndThen([promise = std::move(promise), event = execute_event.CopyRef()]() mutable {
            if (auto* error = event.GetErrorIfPresent()) {
                promise.Set(Internal("Compute error: %s", error->message()));
            } else {
                promise.Set();
            }
        });
        return Result({std::move(future), /*buffers=*/std::move(res)});
    }

    return Result({/*future=*/std::nullopt, std::move(res)});
}

void PjRtNpuExecutable::Delete() {}

bool PjRtNpuExecutable::IsDeleted() const { return false; }

absl::StatusOr<std::string> PjRtNpuExecutable::SerializeExecutable() const {
    return absl::UnimplementedError("PjRtNpuExecutable::SerializeExecutable is not supported.");
}

absl::StatusOr<CompiledMemoryStats> PjRtNpuExecutable::GetCompiledMemoryStats() const {
    return absl::UnimplementedError("PjRtNpuExecutable::GetCompiledMemoryStats is not supported.");
}

}  // namespace xla