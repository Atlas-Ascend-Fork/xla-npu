#ifndef XLA_BACKEND_GPU_RUNTIME_KERNEL_THUNK_H_
#define XLA_BACKEND_GPU_RUNTIME_KERNEL_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
// #include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_npu/backend/thunk.h"
#include "xla/xla_npu/third_party/register/aclInterface.h"
#include "xla/xla_npu/third_party/register/aclnnopInterface.h"

namespace xla {
namespace gpu {

// TODO(ezhulenev): Unify KernelThunk and CustomKernelThunk as they are very
// similar. XLA:GPU should use more of kernel loading APIs provided by
// StreamExecutor out of the box and less custom kernel loading solutions.
//
// Today KernelThunk is required for lowering to XLA runtime, and
// CustomKernelThunk is only supported for thunk execution.
//
// This class stores everything that StreamExecutor needs for launching a
// kernel. It implements the ExecuteOnStream interface for GpuExecutable to
// invoke the corresponding kernel.
//
// This is thread-compatible.
class KernelThunk : public NpuThunk {
   public:
    // Constructs a thunk for the given kernel.
    //
    // KernelThunk takes args as `BufferAllocation::Slice`s (wrapped in
    // `KernelArgument`s). Each slice directly corresponds to an argument or
    // output of the computation. Also, the values must correspond to each arg
    // directly, not to their base allocation (e.g. they can be the result of an
    // `mlir::memref::ViewOp`).
    KernelThunk(NpuThunk::ThunkInfo thunk_info, std::string kernel_name, const emitters::KernelArguments& kernel_arguments,
                LaunchDimensions launch_dimensions, std::optional<se::ClusterDim> cluster_dim, int64_t shmem_bytes,
                stream_executor::gpu::TmaMetadata tma_metadata);
    KernelThunk(const KernelThunk&) = delete;
    KernelThunk& operator=(const KernelThunk&) = delete;
    ~KernelThunk() override = default;

    absl::StatusOr<std::vector<aclTensor*>> ExtractAclInputOutTensorPtr(const BufferAllocations* allocations,
                                                                        const std::vector<Shape>& input_param_shapes,
                                                                        const std::vector<Shape>& output_param_shapes);

    using AclnnExecFunc = aclnnStatus (*)(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);
    absl::Status ExecuteAclnn(aclrtStream& stream, uint64_t workspaceSize, aclOpExecutor* executor,
                              AclnnExecFunc aclnnExecFunc);

    absl::Status PrintOutputTensor(const BufferAllocations* allocations, const std::vector<Shape>& output_param_shapes);

    std::string ToString(int indent) const override;

    absl::StatusOr<ThunkProto> ToProto() const override;
    static absl::StatusOr<std::unique_ptr<KernelThunk>> FromProto(
        ThunkInfo thunk_info, const KernelThunkProto& proto, absl::Span<const BufferAllocation> buffer_allocations);

    absl::Status Initialize(const InitializeParams& params) override;
    absl::Status ExecuteOnStream(const ExecuteParams& params) override;

    const std::vector<BufferAllocation::Slice>& arguments() const { return args_; }
    const std::vector<bool>& written() const { return written_; }

    const std::string& kernel_name() const { return kernel_name_; }
    const LaunchDimensions& launch_dimensions() const { return launch_dimensions_; }
    const std::optional<se::ClusterDim>& cluster_dim() const { return cluster_dim_; }
    // The shared memory required by the kernel.
    int64_t shmem_bytes() const { return shmem_bytes_; }

    const stream_executor::gpu::TmaMetadata& tma_metadata() const { return tma_metadata_; }

    BufferUses buffer_uses() const override;

   private:
    // Buffer slices passed to the kernel as arguments.
    std::vector<BufferAllocation::Slice> args_;

    // args_[i] is written iff (written_[i] == true).
    std::vector<bool> written_;

    // Entry kernel name for the computation.
    const std::string kernel_name_;

    // The thread and block dimension used to launch the kernel.
    const LaunchDimensions launch_dimensions_;

    // The cluster dimensions used to launch the kernel.
    const std::optional<se::ClusterDim> cluster_dim_;

    int64_t shmem_bytes_;

    // Map of argument index to TmaDescriptor used to create arguments to the
    // kernel.
    stream_executor::gpu::TmaMetadata tma_metadata_;

    // Loaded kernels for each `StreamExecutor`.
    mutable absl::Mutex mutex_;
    absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<se::Kernel>> kernel_cache_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKEND_GPU_RUNTIME_KERNEL_THUNK_H_
