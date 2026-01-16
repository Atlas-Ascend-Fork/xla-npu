#ifndef PJRT_NPU_NPU_CLIENT_H_
#define PJRT_NPU_NPU_CLIENT_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
// #include "xla/pjrt/cpu/cpu_async_execution_tracker.h"
// #include "xla/pjrt/cpu/execution_stream_event_map.h"
#include "tsl/platform/fingerprint.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/transpose.h"
#include "xla/xla_npu/pjrt/npu/npu_device.h"
#include "xla/xla_npu/pjrt/npu/tracked_npu_device_buffer.h"
#include "xla/xla_npu/pjrt/npu_pjrt_common.h"
#include "xla/xla_npu/pjrt/plugin/npu_client_options.h"
#include "xla/xla_npu/pjrt/plugin/npu_device_description.h"
#include "xla/xla_npu/pjrt/plugin/npu_topology_description.h"
#include "xla/xla_npu/service/npu_buffer_assignment.h"
#include "xla/xla_npu/third_party/register/aclInterface.h"

namespace xla {

inline const char* DataLayout() {
    static constexpr char kDataLayout[] = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64";
    return kDataLayout;
}

class PjRtNpuClient final : public CommonPjRtClient {
   public:
    PjRtNpuClient(int process_index, std::vector<std::unique_ptr<PjRtNpuDevice>> devices);

    ~PjRtNpuClient() override = default;

    using CollectiveLaunchEvent = std::pair<tsl::AsyncValueRef<NpuEvent>, tsl::CountDownAsyncValueRef<NpuEvent>>;

    int process_index() const override { return process_index_; }

    int device_count() const override { return devices_.size(); }

    int addressable_device_count() const override { return addressable_devices_.size(); }

    absl::Span<PjRtDevice* const> devices() const override { return devices_; }

    absl::Span<PjRtDevice* const> addressable_devices() const override { return addressable_devices_; }

    PjRtPlatformId platform_id() const override { return tsl::Fingerprint64(NpuName()); }

    absl::string_view platform_name() const override { return NpuName(); }

    absl::string_view platform_version() const override { return NpuName(); }

    // Gets the pointer to the topology description held by the client.
    absl::StatusOr<const PjRtTopologyDescription*> GetTopologyDescription() const override { return &topology_; }

    AsyncWorkRunner* async_work_runner() const override { return async_work_runner_.get(); }

    tsl::thread::ThreadPool* pjrt_client_thread_pool() const { return pjrt_client_thread_pool_.get(); }

    // // Returns a pair of async events:
    // // - async event that signals the completion of the last collective launch
    // // - count down event that must be signalled when each rank completes
    // //   a collective launch
    // CollectiveLaunchEvent GetLastCollectiveLaunchEvent(size_t num_addressable_devices) {
    //     tsl::CountDownAsyncValueRef<NpuEvent> count_down(num_addressable_devices);
    //     absl::MutexLock lock(mu_);
    //     auto last_launch = std::move(last_collective_launch_event_);
    //     last_collective_launch_event_ = count_down.AsRef();
    //     return std::make_pair(std::move(last_launch), std::move(count_down));
    // }

    absl::StatusOr<PjRtDevice*> LookupDevice(PjRtGlobalDeviceId global_device_id) const override;

    absl::StatusOr<PjRtDevice*> LookupAddressableDevice(PjRtLocalDeviceId local_device_id) const override;

    absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

    absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(const XlaComputation& computation,
                                                                         CompileOptions options) override;

    absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(mlir::ModuleOp module,
                                                                         CompileOptions options) override;

    // Computes the memory requirements for storing shape on memory_space.
    absl::StatusOr<int64_t> GetOnDeviceBytesCount(PjRtMemorySpace* memory_space,
                                                  const xla::Shape& shape) const override;

    // Allocates a raw buffer of a particular size after an optional
    // allocate_after. Backends may support retrying allocation on oom which
    // can be controlled via retry_on_oom.
    absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>> AllocateRawBuffer(
        PjRtMemorySpace* memory_space, size_t on_device_bytes_count, bool retry_on_oom,
        tsl::AsyncValueRef<bool> allocate_after) override;

    absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> LinearizeHostBufferInto(
        const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
        std::optional<absl::Span<int64_t const>> byte_strides, HostBufferSemantics host_buffer_semantics,
        absl::AnyInvocable<void() &&> on_done_with_host_buffer, const xla::Shape& device_shape,
        tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) override;

    // Defines a pjrt buffer from a shape, raw_buffer and definition events.
    absl::StatusOr<std::unique_ptr<PjRtBuffer>> DefineBuffer(
        const Shape& on_device_shape, tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
        absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4> definition_device_events,
        bool raw_buffer_is_mutable) override;

    absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>, tsl::RCReference<PjRtDeviceEvent>>>
    CreateLinkedEventPromise(PjRtMemorySpace* memory_space, absl::string_view debug_info) override;

    absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(int num_replicas, int num_partitions) const override;

   private:
    friend class PjRtNpuExecutable;
    friend absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtNpuClient(NpuClientOptions options);

    PjRtNpuClient(int process_index, std::vector<std::unique_ptr<PjRtNpuDevice>> devices,
                  std::shared_ptr<NpuDeviceMemory::Allocator> allocator,
                  // std::shared_ptr<cpu::CpuCollectives> collectives,
                  size_t num_threads, bool asynchronous,
                  std::function<void(HloModuleConfig&)> customize_hlo_module_config);

    absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileInternal(
        const XlaComputation& computation, const std::vector<const Shape*>& argument_layout_pointers,
        LayoutCanonicalizationCallback layout_canonicalization_callback, CompileOptions options,
        const AotCompilationOptions* absl_nullable aot_options = nullptr);

    NpuDeviceMemory::Allocator* allocator() const { return allocator_.get(); }

    int process_index_;
    // Includes all devices, including non-addressable devices.
    std::vector<std::unique_ptr<PjRtNpuDevice>> owned_devices_;
    // Pointers to `owned_devices_`.
    std::vector<PjRtDevice*> devices_;
    // Maps Device::id() to the corresponding Device. Includes all devices.
    absl::flat_hash_map<PjRtGlobalDeviceId, PjRtNpuDevice*> id_to_device_;
    // Addressable devices indexed by core_id.
    std::vector<PjRtDevice*> addressable_devices_;
    std::unique_ptr<ComputationPlacer> computation_placer_;

    // Addressable memory spaces.
    std::vector<std::unique_ptr<PjRtMemorySpace>> owned_memory_spaces_;
    // Pointers to `owned_memory_spaces_`.
    std::vector<PjRtMemorySpace*> memory_spaces_;

    // A memory allocator used to allocate host memory for PjRtBuffers, and
    // temporary allocations passed to XLA:CPU executable.
    std::shared_ptr<NpuDeviceMemory::Allocator> allocator_;

    // Launching collectives are prone to deadlock when we use fixed-sized
    // threadpools since ExecuteHelper will block until all replicas reach the
    // barrier. We ensure that
    // 1. Threadpool size is at least as large as device_count so one collective
    //    launch over all devices can succeed.
    // 2. Gang-schedule each collective by conservatively ensuring a total order
    //    of collectives and launching only one collective at a time to avoid
    //    having no active threads to make progress
    // TODO(zhangqiaorjc): Explore alternatives that allow multiple concurrent
    // collectives.
    mutable absl::Mutex mu_;
    // tsl::AsyncValueRef<NpuEvent> last_collective_launch_event_ ABSL_GUARDED_BY(mu_);

    // A cache for transpose plans. We use transposes to convert
    // (possibly strided) buffers provided to BufferFromHostBuffer into dense
    // major-to-minor layout.
    absl::Mutex transpose_mu_;
    TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);

    // std::shared_ptr<cpu::CpuCollectives> collectives_;

    NpuTopologyDescription topology_;

    // Used to control whether asynchronous computation dispatch is available for
    // this client. Only applies to non-parallel computations.
    bool asynchronous_;

    // A callback to customize the HloModuleConfig for each compiled module.
    std::function<void(HloModuleConfig&)> customize_hlo_module_config_;

    // IMPORTANT: All thread pools must be destroyed first, because thread pool
    // destruction guarantees that all scheduled tasks are completed. Otherwise,
    // we might get use-after-free races when dispatched executables try to access
    // the member variables of this class that are already destroyed.

    // // TODO(zhangqiaorjc): Use tsl::compat::EigenHostContextThreadPool.
    // std::unique_ptr<tsl::thread::ThreadPool> eigen_intraop_pool_;
    // std::unique_ptr<Eigen::ThreadPoolDevice> eigen_intraop_device_;

    // Thread pool for running PjRtClient tasks.
    std::unique_ptr<tsl::thread::ThreadPool> pjrt_client_thread_pool_;
    std::unique_ptr<AsyncWorkRunner> async_work_runner_;
};

class PjRtNpuExecutable final : public PjRtLoadedExecutable {
   public:
    PjRtNpuExecutable(int num_replicas, int num_partitions, std::shared_ptr<DeviceAssignment> device_assignment,
                      bool parameter_is_tupled_arguments, CompileOptions compile_options,
                      std::unique_ptr<Executable> npu_executable,
                      absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
                      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
                      std::vector<PjRtDevice*> addressable_devices, PjRtNpuClient* client,
                      std::unique_ptr<HloModule> unoptimized_hlo_module);

    ~PjRtNpuExecutable() override = default;

    PjRtNpuClient* client() const override { return client_; }

    absl::string_view name() const override { return npu_executable_->shared_module()->name(); }

    int num_replicas() const override { return num_replicas_; }

    int num_partitions() const override { return num_partitions_; }

    int64_t SizeOfGeneratedCodeInBytes() const override { return npu_executable_->SizeOfGeneratedCodeInBytes(); }

    const DeviceAssignment& device_assignment() const override { return *device_assignment_; }

    absl::Span<const LogicalDeviceIds> addressable_device_logical_ids() const override {
        return addressable_device_logical_ids_;
    }

    absl::Span<PjRtDevice* const> addressable_devices() const override { return addressable_devices_; }

    absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules() const override {
        return std::vector<std::shared_ptr<HloModule>>{npu_executable_->shared_module()};
    }

    absl::StatusOr<std::vector<std::vector<absl::string_view>>> GetOutputMemoryKinds() const override {
        return Unimplemented("GetOutputMemoryKinds is not supported.");
    }

    absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override;

    using PjRtLoadedExecutable::Execute;
    absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
        absl::Span<const std::vector<PjRtBuffer*>> argument_handles, const ExecuteOptions& options,
        std::optional<std::vector<Future<>>>& returned_futures) const override;

    using PjRtLoadedExecutable::ExecuteSharded;
    absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
        absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device, const ExecuteOptions& options,
        std::optional<Future<>>& returned_future, bool fill_future) const override;

    using PjRtLoadedExecutable::ExecutePortable;
    absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
        absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device, const ExecuteOptions& options,
        std::optional<Future<>>& returned_future, bool fill_future) const override;

    void Delete() override;

    bool IsDeleted() const override;

    absl::StatusOr<std::string> SerializeExecutable() const override;

    std::shared_ptr<Executable> npu_executable() const { return npu_executable_; }

    absl::StatusOr<std::optional<std::string>> Fingerprint() const { return fingerprint_; }

    absl::StatusOr<std::string> FingerprintExecutable() const override { return fingerprint_; }

    absl::StatusOr<CompileOptions> GetCompileOptions() const override { return compile_options_; }

   private:
    friend class PjRtNpuClient;

    // absl::Status SetUpDonation(bool tuple_inputs);

    // Checks that the input buffers passed in by the user have the correct size
    // on device for the compiled program.
    // absl::Status CheckBufferCompatibilities(
    //     absl::Span<std::pair<bool, TrackedNpuDeviceBuffer*> const> input_buffers)
    //     const;

    absl::StatusOr<Result> ExecuteHelper(absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
                                         const RunId& run_id, const ExecuteOptions& options,
                                         PjRtNpuClient::CollectiveLaunchEvent last_collective_launch_event,
                                         bool fill_future, PjRtNpuDevice* device = nullptr) const;

    PjRtNpuClient* client_;

    int num_replicas_;
    int num_partitions_;
    std::shared_ptr<DeviceAssignment> device_assignment_;
    bool parameter_is_tupled_arguments_;
    CompileOptions compile_options_;

    std::shared_ptr<Executable> npu_executable_;

    // Caching `result_buffer_indices_` to avoid lookup
    // HLO dataflow analysis data structures in program execution critical path.

    // Buffer allocation indices corresponding to each result buffer leaf buffer.
    absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices_;

    // Size on device of each leaf buffer of the compiled program, cached here
    // for performance reasons.
    std::vector<int64_t> input_buffer_sizes_in_bytes_;

    // A sorted vector of parameters that have any aliased buffers and thus must
    // be donated when executing the computation.
    std::vector<int> parameters_that_must_be_donated_;

    // The replica and partition indices of device_assignment_ to be run by this
    // client. On single-host platforms without partitioning, this is all
    // replicas (i.e. addressable_device_logical_ids_[i] = (i, 0)), but this may
    // not be the case on multi-host platforms. If there are 4 replicas and 2
    // partitions on a single host platform, size of
    // addressable_device_logical_ids_ is 4*2 = 8.
    std::vector<LogicalDeviceIds> addressable_device_logical_ids_;

    // addressable_devices_[i] is the Device to which
    // addressable_device_logical_ids_[i] is assigned. shared_ptrs instead of
    // unique_ptrs to play well with the Python bindings (see xla.cc).
    std::vector<PjRtDevice*> addressable_devices_;

    // Cached result of comparing HloCostAnalysis FLOP estimate for execute
    // critical path.
    bool cheap_computation_;

    std::string fingerprint_;

    std::unique_ptr<HloModule> unoptimized_hlo_module_;
};

absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtNpuClient(NpuClientOptions options);

}  // namespace xla

#endif  // PJRT_NPU_NPU_CLIENT_H_