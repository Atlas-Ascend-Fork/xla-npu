#include "xla/xla_npu/backend/cpu_compiler_type.npu_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
// #include "xla/backends/cpu/collectives/cpu_collectives.h"
// #include "xla/backends/cpu/collectives/in_process_collectives.h"
// #include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
// #include "xla/backends/cpu/runtime/xnnpack/xnn_threadpool.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"
#include "xla/executable_run_options.h"
// #include "xla/service/cpu/cpu_executable_run_options.h"
#include "xla/service/global_device_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

// #ifdef XLA_YNNPACK
// #include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
// #include "xla/backends/cpu/runtime/ynnpack/ynn_threadpool.h"
// #endif  // XLA_YNNPACK

#include "xla/xla_npu/third_party/register/aclInterface.h"

namespace xla {

// Ok execute event allocated with the static storage duration.
static tsl::internal::AsyncValueStorage<NpuThunk::ExecuteEvent> ok_storage;
absl::NoDestructor<tsl::AsyncValueOwningRef<NpuThunk::ExecuteEvent>> NpuThunk::ok_event_(
    tsl::MakeAvailableAsyncValueRef<ExecuteEvent>(ok_storage));

tsl::AsyncValueRef<NpuThunk::ExecuteEvent> NpuThunk::ErrorExecuteEvent(absl::Status error_status) {
    auto async_value = tsl::MakeAvailableAsyncValueRef<ExecuteEvent>();
    // 设置错误状态（TSL AsyncValue 通用接口）
    async_value.SetError(std::move(error_status));
    return async_value;
}

tsl::AsyncValueRef<NpuThunk::ExecuteEvent> NpuThunk::AclErrorExecuteEvent(int acl_ret, const std::string& context) {
    // 映射 ACL 错误码到 absl::Status
    absl::StatusCode code = absl::StatusCode::kInternal;
    std::string error_detail;
    switch (acl_ret) {
        case ACL_ERROR_INVALID_PARAM:
            code = absl::StatusCode::kInvalidArgument;
            error_detail = "invalid parameter";
            break;
        case ACL_ERROR_API_NOT_SUPPORT:
            code = absl::StatusCode::kUnimplemented;
            error_detail = "operation not supported";
            break;
        case ACL_ERROR_BAD_ALLOC:
            code = absl::StatusCode::kResourceExhausted;
            error_detail = "out of memory";
            break;
        default:
            error_detail = "internal error";
            break;
    }

    // 拼接详细错误信息（含文件、行号、上下文）
    std::string error_msg =
        absl::StrFormat("[%s:%d] %s: %s (ACL error code: %d)", __FILE__, __LINE__, context, error_detail, acl_ret);

    return ErrorExecuteEvent(absl::Status(code, error_msg));
}

absl::string_view NpuThunk::KindToString(Kind kind) {
    switch (kind) {
        case Kind::kCall:
            return "call";
        case Kind::kCollective:
            return "collective";
        case Kind::kConditional:
            return "conditional";
        case Kind::kConvolution:
            return "convolution";
        case Kind::kCopy:
            return "copy";
        case Kind::kCustomCall:
            return "custom-call";
        case Kind::kDot:
            return "dot";
        case Kind::kFft:
            return "fft";
        case Kind::kInfeed:
            return "infeed";
        case Kind::kKernel:
            return "kernel";
        case Kind::kOutfeed:
            return "outfeed";
        case Kind::kPartitionId:
            return "partition-id";
        case Kind::kReplicaId:
            return "replica-id";
        case Kind::kRngGetAndUpdateState:
            return "rng-get-and-update-state";
        case Kind::kSort:
            return "sort";
        case Kind::kTopK:
            return "topk";
        case Kind::kWhile:
            return "while";
        case Kind::kXnnFusion:
            return "xnn-fusion";
        case Kind::kYnnFusion:
            return "ynn-fusion";
        case Kind::kOneDnnFusion:
            return "onednn-fusion";
    }
}

NpuThunk::NpuThunk(Kind kind, Info info) : kind_(kind), info_(std::move(info)) {}

// absl::StatusOr<NpuThunk::CollectiveExecuteParams> NpuThunk::CollectiveExecuteParams::Create(
//     const ExecutableRunOptions* run_options) {
//     // Device ordinal must be set by caller and passed in run options, if not,
//     // we use the device ordinal from the parent StreamExecutor.
//     int32_t device_ordinal = run_options->device_ordinal() >= 0 ? run_options->device_ordinal()
//                                                                 : run_options->stream()->parent()->device_ordinal();

//     // Default implementation of a collectives interface that can execute
//     // collective operations within the same process.
//     static cpu::CpuCollectives* const in_process_collectives = new cpu::InProcessCollectives();

//     // If CPU executable run options are set, use the collectives interface
//     // provided by the executable run options if it is set. Otherwise, use the
//     // in-process collectives interface.
//     const cpu::CpuExecutableRunOptions* cpu_run_options = run_options->cpu_executable_run_options();
//     cpu::CpuCollectives* collectives =
//         cpu_run_options && cpu_run_options->collectives() ? cpu_run_options->collectives() : in_process_collectives;

//     return CollectiveExecuteParams{run_options->run_id(), device_ordinal,
//     GlobalDeviceId(run_options->device_ordinal()),
//                                    run_options->device_assignment(), collectives};
// }

// NpuThunk::CollectiveExecuteParams::CollectiveExecuteParams(RunId run_id, int64_t local_device_ordinal,
//                                                            GlobalDeviceId global_device_id,
//                                                            const DeviceAssignment* device_assignment,
//                                                            cpu::CpuCollectives* collectives)
//     : run_id(run_id),
//       local_device_ordinal(local_device_ordinal),
//       global_device_id(global_device_id),
//       device_assignment(device_assignment),
//       collectives(collectives) {}

absl::StatusOr<NpuThunk::CustomCallExecuteParams> NpuThunk::CustomCallExecuteParams::Create(
    const ExecutableRunOptions* run_options) {
    // Device ordinal must be set by caller and passed in run options, if not,
    // we use the device ordinal from the parent StreamExecutor.
    int32_t device_ordinal = run_options->device_ordinal() >= 0 ? run_options->device_ordinal()
                                                                : run_options->stream()->parent()->device_ordinal();

    return CustomCallExecuteParams{run_options->run_id(), device_ordinal, run_options->intra_op_thread_pool(),
                                   run_options->ffi_execution_context()};
}

NpuThunk::CustomCallExecuteParams::CustomCallExecuteParams(RunId run_id, int32_t device_ordinal,
                                                           const Eigen::ThreadPoolDevice* intra_op_thread_pool,
                                                           const ffi::ExecutionContext* ffi_execution_context)
    : run_id(run_id),
      device_ordinal(device_ordinal),
      intra_op_thread_pool(intra_op_thread_pool),
      ffi_execution_context(ffi_execution_context) {}

// absl::StatusOr<NpuThunk::XnnParams> NpuThunk::XnnParams::Create(
//     const ExecutableRunOptions* run_options) {
//   TF_ASSIGN_OR_RETURN(cpu::XnnThreadpool threadpool,
//                       cpu::CreateXnnThreadpool(run_options->intra_op_thread_pool()));
//   return XnnParams(std::move(threadpool));
// }

// NpuThunk::XnnParams::XnnParams(cpu::XnnThreadpool threadpool)
//     : threadpool(std::move(threadpool)) {}

// #ifdef XLA_YNNPACK
// absl::StatusOr<NpuThunk::YnnParams> NpuThunk::YnnParams::Create(
//     const ExecutableRunOptions* run_options) {
//   TF_ASSIGN_OR_RETURN(cpu::YnnThreadpool threadpool,
//                       cpu::CreateYnnThreadpool(run_options->intra_op_thread_pool()));
//   return YnnParams(std::move(threadpool));
// }

// NpuThunk::YnnParams::YnnParams(cpu::YnnThreadpool threadpool)
//     : threadpool(std::move(threadpool)) {}
// #endif  // XLA_YNNPACK

NpuThunk::ExecuteSession::ExecuteSession(int64_t max_workers, int64_t split_threshold)
    : lock_(std::make_shared<std::nullopt_t>(std::nullopt)),
      max_workers_(max_workers),
      split_threshold_(split_threshold) {}

// Encodes thunk info into the TraceMe compatible format.
std::string NpuThunk::TraceMeEncode(int64_t run_id, int64_t device_ordinal) const {
    return tsl::profiler::TraceMeEncode(info_.op_name, {{"hlo_op", info_.op_name},
                                                        {"hlo_module", info_.module_name},
                                                        {"program_id", info_.module_id},
                                                        {"run_id", run_id},
                                                        {"device_ordinal", device_ordinal}});
}

std::ostream& operator<<(std::ostream& os, NpuThunk::Kind kind) {
    os << NpuThunk::KindToString(kind);
    return os;
}

NpuThunkSequence::NpuThunkSequence(std::unique_ptr<NpuThunk> thunk) { push_back(std::move(thunk)); }

void NpuThunkSequence::Append(NpuThunkSequence other) {
    reserve(size() + other.size());
    for (auto& thunk : other) {
        push_back(std::move(thunk));
    }
}

NpuThunkSequence::BufferUses NpuThunkSequence::buffer_uses() const {
    BufferUses buffer_uses;
    for (auto& thunk : *this) {
        BufferUses uses = thunk->buffer_uses();
        buffer_uses.insert(buffer_uses.end(), uses.begin(), uses.end());
    }
    return buffer_uses;
}

NpuThunkSequence::ResourceUses NpuThunkSequence::resource_uses() const {
    ResourceUses resource_uses;
    for (auto& thunk : *this) {
        ResourceUses uses = thunk->resource_uses();
        resource_uses.insert(resource_uses.end(), uses.begin(), uses.end());
    }
    return resource_uses;
}

static void ForEach(const NpuThunkSequence& sequence, absl::FunctionRef<void(const NpuThunk&)> fn) {
    for (auto& thunk : sequence) {
        fn(*thunk);
        for (auto& [name, nested] : thunk->nested_thunks()) {
            ForEach(*nested, fn);
        }
    }
}

static absl::Status ForEach(const NpuThunkSequence& sequence, absl::FunctionRef<absl::Status(const NpuThunk&)> fn) {
    for (auto& thunk : sequence) {
        TF_RETURN_IF_ERROR(fn(*thunk));
        for (auto& [name, nested] : thunk->nested_thunks()) {
            TF_RETURN_IF_ERROR(ForEach(*nested, fn));
        }
    }
    return absl::OkStatus();
}

void NpuThunkSequence::ForEach(absl::FunctionRef<void(const NpuThunk&)> fn) const { xla::ForEach(*this, fn); }

absl::Status NpuThunkSequence::ForEachWithStatus(absl::FunctionRef<absl::Status(const NpuThunk&)> fn) const {
    return xla::ForEach(*this, fn);
}

}  // namespace xla
