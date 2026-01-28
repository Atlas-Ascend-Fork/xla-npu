#include "xla/xla_npu/backend/npu_sequential_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/annotation.h"
// #include "xla/backends/gpu/runtime/thunk.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/npu_thunk.h"

namespace xla {
namespace gpu {

NpuSequentialThunk::NpuSequentialThunk(ThunkInfo thunk_info, NpuThunkSequence thunks)
    : NpuThunk(Kind::kSequential, thunk_info), thunks_(std::move(thunks)) {}

std::string NpuSequentialThunk::ToString(int indent) const {
    const std::string indent_str(indent * 2, ' ');
    if (thunks_.empty()) {
        return indent_str + "No thunks.";
    }

    auto thunk_with_longest_kind =
        absl::c_max_element(thunks_, [](const std::unique_ptr<NpuThunk>& a, const std::unique_ptr<NpuThunk>& b) {
            return NpuThunk::KindToString(a->kind()).length() < NpuThunk::KindToString(b->kind()).length();
        });
    int64_t max_thunk_kind_len = NpuThunk::KindToString(thunk_with_longest_kind->get()->kind()).length();
    std::string result;
    for (const std::unique_ptr<NpuThunk>& thunk : thunks_) {
        absl::StrAppend(&result, indent_str);
        absl::StrAppendFormat(&result, "%03d: ", thunk->thunk_info().thunk_id.value());
        // Write out the thunk kind, padded out to max_thunk_kind_len.
        absl::string_view kind_str = NpuThunk::KindToString(thunk->kind());
        absl::StrAppend(&result, kind_str, std::string(max_thunk_kind_len - kind_str.length(), ' '), "\t");
        absl::StrAppend(&result, thunk->ToString(indent + 1));
        absl::StrAppend(&result, "\n");
    }
    return result;
}

// absl::Status NpuSequentialThunk::Prepare(const PrepareParams& params, ResourceRequestsInterface& resource_requests) {
//     for (auto& thunk : thunks_) {
//         TF_RETURN_IF_ERROR(thunk->Prepare(params, resource_requests));
//     }
//     return absl::OkStatus();
// }

absl::Status NpuSequentialThunk::Initialize(const InitializeParams& params) {
    for (auto& thunk : thunks_) {
        TF_RETURN_IF_ERROR(thunk->Initialize(params));
    }
    return absl::OkStatus();
}

absl::Status NpuSequentialThunk::ExecuteOnStream(const ExecuteParams& params) {
    std::optional<tsl::profiler::ScopedAnnotation> seq_annotation = GetKernelAnnotation(profile_annotation());
    for (const std::unique_ptr<NpuThunk>& thunk : thunks_) {
        tsl::profiler::TraceMe trace(thunk->profile_annotation());

        std::optional<tsl::profiler::ScopedAnnotation> annotation = GetKernelAnnotation(thunk->profile_annotation());
        if (params.mock_collectives && thunk->IsCollective()) {
            continue;
        }

        VLOG(1) << "[" << params.stream->parent()->device_ordinal() << "] "
                << "Start NpuSequentialThunk::ExecuteOnStream: " << thunk->profile_annotation();
        TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));
        VLOG(1) << "[" << params.stream->parent()->device_ordinal() << "] "
                << "End NpuSequentialThunk::ExecuteOnStream: " << thunk->profile_annotation();
    }
    return absl::OkStatus();
}

void NpuSequentialThunk::ForAllThunks(absl::FunctionRef<void(const NpuThunk*)> fn) const {
    fn(this);
    for (const std::unique_ptr<NpuThunk>& thunk : thunks_) {
        thunk->ForAllThunks(fn);
    }
}

void NpuSequentialThunk::ForAllThunksMutable(absl::FunctionRef<void(NpuThunk*)> fn) {
    fn(this);
    for (const std::unique_ptr<NpuThunk>& thunk : thunks_) {
        thunk->ForAllThunksMutable(fn);
    }
}

absl::Status NpuSequentialThunk::TransformAllNestedThunks(
    absl::FunctionRef<absl::StatusOr<std::unique_ptr<NpuThunk>>(std::unique_ptr<NpuThunk>)> fn) {
    for (std::unique_ptr<NpuThunk>& thunk : thunks_) {
        TF_RETURN_IF_ERROR(thunk->TransformAllNestedThunks(fn));
        TF_ASSIGN_OR_RETURN(thunk, fn(std::move(thunk)));
    }
    return absl::OkStatus();
}

absl::StatusOr<ThunkProto> NpuSequentialThunk::ToProto() const {
    ThunkProto proto;
    *proto.mutable_thunk_info() = thunk_info().ToProto();

    // This sets the oneof-type to the sequential thunk, even if the thunk list is
    // empty.
    proto.mutable_sequential_thunk();
    for (const auto& thunk : thunks_) {
        TF_ASSIGN_OR_RETURN(*proto.mutable_sequential_thunk()->add_thunks(), thunk->ToProto());
    }
    return proto;
}

absl::StatusOr<std::unique_ptr<NpuSequentialThunk>> NpuSequentialThunk::FromProto(
    ThunkInfo thunk_info, const SequentialThunkProto& thunk_proto, const Deserializer& deserializer) {
    NpuThunkSequence thunk_sequence;
    for (const auto& sub_thunk_proto : thunk_proto.thunks()) {
        TF_ASSIGN_OR_RETURN(std::unique_ptr<NpuThunk> sub_thunk, deserializer(sub_thunk_proto));
        thunk_sequence.push_back(std::move(sub_thunk));
    }

    return std::make_unique<NpuSequentialThunk>(std::move(thunk_info), std::move(thunk_sequence));
}

std::unique_ptr<NpuSequentialThunk> NpuSequentialThunk::FromThunk(std::unique_ptr<NpuThunk> thunk) {
    if (thunk->kind() == NpuThunk::kSequential) {
        return std::unique_ptr<NpuSequentialThunk>(static_cast<NpuSequentialThunk*>(thunk.release()));
    }

    std::vector<std::unique_ptr<NpuThunk>> thunks;
    thunks.push_back(std::move(thunk));
    return std::make_unique<NpuSequentialThunk>(NpuThunk::ThunkInfo(), std::move(thunks));
}

}  // namespace gpu
}  // namespace xla
