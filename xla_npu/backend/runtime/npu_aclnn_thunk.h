#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_ACLNN_THUNK_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_ACLNN_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
// #include "xla/backends/gpu/runtime/thunk.h"
// #include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/dnn.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_interface.h"

namespace xla {

// Wraps executable cuDNN graph objects.
class NpuAclnnThunk : public NpuThunk {
   public:
    NpuAclnnThunk(std::string fingerprint, ThunkInfo thunk_info, std::vector<BufferAllocation::Slice> args,
                  std::vector<bool> output_args, std::optional<int64_t> sdpa_dropout_seed = std::nullopt);
    NpuAclnnThunk(const NpuAclnnThunk&) = delete;
    NpuAclnnThunk& operator=(const NpuAclnnThunk&) = delete;
    ~NpuAclnnThunk() override = default;

    absl::Status Initialize(const InitializeParams&) override;
    absl::Status ExecuteOnStream(const ExecuteParams&) override;

    std::shared_ptr<se::dnn::LazyDnnGraph> graph() const { return graph_; }
    const std::vector<BufferAllocation::Slice>& arguments() const { return args_; }

    // absl::StatusOr<ThunkProto> ToProto() const override;

    // static absl::StatusOr<std::unique_ptr<NpuAclnnThunk>> FromProto(
    //     ThunkInfo thunk_info, const CudnnThunkProto& proto, absl::Span<const BufferAllocation> buffer_allocations);

    absl::StatusOr<std::vector<aclTensor*>> ExtractAclInputOutTensorPtr(const NpuBufferAllocations* allocations,
                                                                        const std::vector<Shape>& param_shapes);

    using AclnnExecFunc = aclnnStatus (*)(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);
    absl::Status ExecuteAclnn(aclrtStream& stream, uint64_t workspaceSize, aclOpExecutor* executor,
                              AclnnExecFunc aclnnExecFunc);

    absl::Status PrintOutputTensor(const NpuBufferAllocations* allocations,
                                   const std::vector<Shape>& output_param_shapes);

   private:
    absl::once_flag once_flag_;
    std::string fingerprint_;
    std::shared_ptr<se::dnn::LazyDnnGraph> graph_;
    std::vector<BufferAllocation::Slice> args_;
    std::vector<bool> output_args_;
    // Sdpa dropout seed
    std::optional<int64_t> sdpa_dropout_seed_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_ACLNN_THUNK_H_
