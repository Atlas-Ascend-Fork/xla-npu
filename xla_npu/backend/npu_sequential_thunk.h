#ifndef BACKEND_NPU_SEQUENTIAL_THUNK_H_
#define BACKEND_NPU_SEQUENTIAL_THUNK_H_

#include <memory>
#include <string>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
// #include "xla/backends/gpu/runtime/thunk.h"
// #include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/xla_npu/backend/npu_thunk.h"

namespace xla {
namespace gpu {

// A thunk that wraps a list of sub-thunks. Executing this thunk executes all
// the sub-thunks sequentially. This is useful to implement instructions that
// require multiple kernel launches or library calls.
class NpuSequentialThunk : public NpuThunk {
   public:
    NpuSequentialThunk(ThunkInfo thunk_info, NpuThunkSequence thunks);
    NpuSequentialThunk(const NpuSequentialThunk&) = delete;
    NpuSequentialThunk& operator=(const NpuSequentialThunk&) = delete;

    NpuThunkSequence& thunks() { return thunks_; }
    const NpuThunkSequence& thunks() const { return thunks_; }
    std::string ToString(int indent) const override;

    // absl::Status Prepare(const PrepareParams& params, ResourceRequestsInterface& resource_requests) override;
    absl::Status Initialize(const InitializeParams& params) override;
    absl::Status ExecuteOnStream(const ExecuteParams& params) override;

    void ForAllThunks(absl::FunctionRef<void(const NpuThunk*)> fn) const override;
    void ForAllThunksMutable(absl::FunctionRef<void(NpuThunk*)> fn) override;
    absl::Status TransformAllNestedThunks(
        absl::FunctionRef<absl::StatusOr<std::unique_ptr<NpuThunk>>(std::unique_ptr<NpuThunk>)> fn) override;

    // absl::StatusOr<ThunkProto> ToProto() const override;

    // static absl::StatusOr<std::unique_ptr<NpuSequentialThunk>> FromProto(ThunkInfo thunk_info,
    //                                                                      const SequentialThunkProto& thunk_proto,
    //                                                                      const Deserializer& deserializer);

    // Converts a NpuThunk into a NpuSequentialThunk. If the input is already a
    // NpuSequentialThunk, the returned value is the downcasted input.
    //
    // The new thunk, if created, will use a default-initialized ThunkInfo.
    static std::unique_ptr<NpuSequentialThunk> FromThunk(std::unique_ptr<NpuThunk> thunk);

   private:
    // The list of sub-thunks.
    NpuThunkSequence thunks_;
};

}  // namespace gpu
}  // namespace xla

#endif  // BACKEND_NPU_SEQUENTIAL_THUNK_H_
