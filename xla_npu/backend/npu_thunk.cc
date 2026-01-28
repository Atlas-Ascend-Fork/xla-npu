#include "xla/xla_npu/backend/npu_thunk.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

#include "xla/xla_npu/common/npu_tool.h"

namespace xla {
namespace gpu {

//===----------------------------------------------------------------------===//
// NpuThunk::CollectiveCliques
//===----------------------------------------------------------------------===//

NpuThunk::CollectiveCliques::CollectiveCliques(AcquiredCliquesMap cliques_map, int32_t num_transient_cliques)
    : cliques_map_(std::move(cliques_map)), num_transient_cliques_(num_transient_cliques) {}

absl::StatusOr<Communicator*> NpuThunk::CollectiveCliques::GetComm(const GpuCliqueKey& clique_key, RankId rank) const {
    // Check that we locked access to a clique for `clique_key`.
    auto clique = cliques_map_.find(clique_key);
    if (clique == cliques_map_.end()) {
        return absl::NotFoundError(absl::StrCat("No clique found for clique key: ", clique_key.ToString()));
    }

    // Check that clique has a communicator for our rank.
    auto communicator = (*clique->second)->comm(rank);
    if (!communicator.has_value()) {
        return absl::InternalError(absl::StrCat("Communicator for rank ", rank.value(), " not found in a NCCL clique ",
                                                clique_key.ToString()));
    }

    return *communicator;
}

absl::StatusOr<bool> NpuThunk::CollectiveCliques::peer_access_enabled(const GpuCliqueKey& clique_key) const {
    // Check that we locked access to a clique for `clique_key`.
    auto clique = cliques_map_.find(clique_key);
    if (clique == cliques_map_.end()) {
        return absl::NotFoundError(absl::StrCat("No clique found for clique key: ", clique_key.ToString()));
    }

    return (*clique->second)->peer_access_enabled();
}

//===----------------------------------------------------------------------===//
// NpuThunk::CollectiveExecuteParams
//===----------------------------------------------------------------------===//

using GlobalDeviceIdMap = NpuThunk::CollectiveExecuteParams::GlobalDeviceIdMap;

// Returns global device id for a local device ordinal or an error if global
// device id map is misconfigured and missing an entry for a local device.
static absl::StatusOr<GlobalDeviceId> GetGlobalDeviceId(const GlobalDeviceIdMap* device_id_map,
                                                        int64_t local_device_ordinal) {
    // No local -> global mapping was provided; assume the identity mapping.
    if (!device_id_map) {
        return GlobalDeviceId(local_device_ordinal);
    }

    // Find a global device id in a global device id map.
    auto it = device_id_map->find(local_device_ordinal);
    if (it == device_id_map->end()) {
        return absl::NotFoundError(
            absl::StrCat("No global device id found for local device ordinal: ", local_device_ordinal));
    }

    return it->second;
}

absl::StatusOr<NpuThunk::CollectiveExecuteParams> NpuThunk::CollectiveExecuteParams::Create(
    const ServiceExecutableRunOptions& run_options, absl::Span<se::Stream* const> async_streams,
    int64_t local_device_ordinal, int64_t collective_max_nchannels, int64_t p2p_max_nchannels) {
    const GpuExecutableRunOptions* gpu_options = run_options.run_options().gpu_executable_run_options();

    GpuCollectives* collectives = nullptr;
        // gpu_options && gpu_options->collectives() ? gpu_options->collectives() : GpuCollectives::Default();

    GlobalDeviceIdMap* device_id_map = nullptr;
        // gpu_options && gpu_options->gpu_global_device_ids() ? &*gpu_options->gpu_global_device_ids() : nullptr;

    CliqueIdCallback* clique_id_callback = nullptr;
        // gpu_options && gpu_options->clique_id_callback() ? &gpu_options->clique_id_callback() : nullptr;

    absl::flat_hash_map<GlobalDeviceId, IncarnationId>* incarnations = nullptr;
        // gpu_options && gpu_options->incarnations().has_value() ? &*gpu_options->incarnations() : nullptr;

    TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id, GetGlobalDeviceId(device_id_map, local_device_ordinal));

    return CollectiveExecuteParams(collectives, run_options.stream()->parent(), run_options.run_options().run_id(),
                                   async_streams, local_device_ordinal, global_device_id,
                                   run_options.run_options().device_assignment(), device_id_map, clique_id_callback,
                                   incarnations, collective_max_nchannels, p2p_max_nchannels);
}

NpuThunk::CollectiveExecuteParams::CollectiveExecuteParams(
    GpuCollectives* collectives, se::StreamExecutor* executor, RunId run_id,
    absl::Span<se::Stream* const> async_streams, int64_t local_device_ordinal, GlobalDeviceId global_device_id,
    const DeviceAssignment* device_assn, const GlobalDeviceIdMap* global_device_id_map,
    const CliqueIdCallback* nccl_clique_id_callback,
    const absl::flat_hash_map<GlobalDeviceId, IncarnationId>* incarnations, int64_t collective_max_nchannels,
    int64_t p2p_max_nchannels)
    : collectives(collectives),
      executor(executor),
      run_id(run_id),
      async_streams(async_streams.begin(), async_streams.end()),
      local_device_ordinal(local_device_ordinal),
      global_device_id(global_device_id),
      device_assn(device_assn),
      global_device_id_map(global_device_id_map),
      nccl_clique_id_callback(nccl_clique_id_callback),
      incarnations(incarnations),
      collective_max_nchannels(collective_max_nchannels),
      p2p_max_nchannels(p2p_max_nchannels) {}

//===----------------------------------------------------------------------===//
// NpuThunk::ExecuteParams
//===----------------------------------------------------------------------===//

NpuThunk::ExecuteParams NpuThunk::ExecuteParams::Create(const ServiceExecutableRunOptions& run_options,
                                                  const BufferAllocations& buffer_allocations, se::Stream* stream,
                                                  se::Stream* command_buffer_trace_stream,
                                                  CollectiveExecuteParams* collective_params,
                                                  CollectiveCliques* collective_cliques,
                                                  ExecutionStreamIdMap additional_compute_streams) {
    return ExecuteParams(&buffer_allocations, stream, command_buffer_trace_stream, collective_params,
                         collective_cliques, run_options.run_options().device_to_host_stream(),
                         run_options.run_options().host_to_device_stream(),
                         run_options.run_options().send_device_memory_function(),
                         run_options.run_options().recv_device_memory_function(),
                         run_options.run_options().ffi_execution_context(), additional_compute_streams,
                         run_options.run_options().gpu_executable_run_options()
                             ? run_options.run_options().gpu_executable_run_options()->enable_mock_collectives()
                             : false,
                         run_options.run_options().run_id().ToInt());
}

NpuThunk::ExecuteParams NpuThunk::ExecuteParams::CloneWithNewAllocations(const NpuThunk::ExecuteParams& params,
                                                                   const BufferAllocations& buffer_allocations) {
    return ExecuteParams(&buffer_allocations, params.stream, params.command_buffer_trace_stream,
                         params.collective_params, params.collective_cliques, params.device_to_host_stream,
                         params.host_to_device_stream, params.send_device_memory_function,
                         params.recv_device_memory_function, params.ffi_execution_context,
                         params.additional_compute_streams);
}

NpuThunk::ExecuteParams::ExecuteParams(const BufferAllocations* buffer_allocations, se::Stream* stream,
                                    se::Stream* command_buffer_trace_stream, CollectiveExecuteParams* collective_params,
                                    CollectiveCliques* collective_cliques, se::Stream* device_to_host_stream,
                                    se::Stream* host_to_device_stream,
                                    SendDeviceMemoryFunction* send_device_memory_function,
                                    RecvDeviceMemoryFunction* recv_device_memory_function,
                                    const ffi::ExecutionContext* ffi_execution_context,
                                    ExecutionStreamIdMap additional_compute_streams, bool mock_collectives,
                                    int64_t execution_id)
    : buffer_allocations(buffer_allocations),
      stream(stream),
      command_buffer_trace_stream(command_buffer_trace_stream),
      collective_params(collective_params),
      collective_cliques(collective_cliques),
      device_to_host_stream(device_to_host_stream),
      host_to_device_stream(host_to_device_stream),
      send_device_memory_function(send_device_memory_function),
      recv_device_memory_function(recv_device_memory_function),
      ffi_execution_context(ffi_execution_context),
      additional_compute_streams(additional_compute_streams),
      mock_collectives(mock_collectives),
      execution_id(execution_id) {}

//===----------------------------------------------------------------------===//

/*static*/ absl::string_view NpuThunk::KindToString(NpuThunk::Kind kind) {
#define CASE(x)    \
    case NpuThunk::x: \
        return #x
    switch (kind) {
        // # go/keep-sorted start
        CASE(kAllGather);
        CASE(kAllGatherDone);
        CASE(kAllGatherStart);
        CASE(kAllReduce);
        CASE(kAllReduceDone);
        CASE(kAllReduceStart);
        CASE(kAllToAll);
        CASE(kAllToAllDone);
        CASE(kAllToAllStart);
        CASE(kBuffersDebugChecksum);
        CASE(kBuffersDebugFloatCheck);
        CASE(kCollectiveBroadcast);
        CASE(kCollectiveBroadcastDone);
        CASE(kCollectiveBroadcastStart);
        CASE(kCollectiveKernel);
        CASE(kCollectivePermute);
        CASE(kCollectivePermuteDone);
        CASE(kCollectivePermuteStart);
        CASE(kCommandBuffer);
        CASE(kConditional);
        CASE(kConvolution);
        CASE(kConvolutionReorder);
        CASE(kCopy);
        CASE(kCopyDone);
        CASE(kCuDnn);
        CASE(kCubSort);
        CASE(kCublasLtMatmul);
        CASE(kCustomCall);
        CASE(kCustomKernel);
        CASE(kDynamicSlice);
        CASE(kFft);
        CASE(kGemm);
        CASE(kGroupDone);
        CASE(kGroupStart);
        CASE(kHostExecuteDone);
        CASE(kHostExecuteStart);
        CASE(kHostRecv);
        CASE(kHostRecvDone);
        CASE(kHostSend);
        CASE(kHostSendDone);
        CASE(kInfeed);
        CASE(kKernel);
        CASE(kMemset32BitValue);
        CASE(kMemzero);
        CASE(kNorm);
        CASE(kNvshmemAllReduceDone);
        CASE(kNvshmemAllReduceStart);
        CASE(kNvshmemCollectivePermute);
        CASE(kNvshmemCollectivePermuteDone);
        CASE(kNvshmemCollectivePermuteStart);
        CASE(kNvshmemRecv);
        CASE(kNvshmemRecvDone);
        CASE(kNvshmemSend);
        CASE(kNvshmemSendDone);
        CASE(kOutfeed);
        CASE(kPartitionId);
        CASE(kRaggedAllToAll);
        CASE(kRaggedAllToAllDone);
        CASE(kRaggedAllToAllStart);
        CASE(kRecv);
        CASE(kRecvDone);
        CASE(kReduceScatter);
        CASE(kReduceScatterDone);
        CASE(kReduceScatterStart);
        CASE(kReplicaId);
        CASE(kSelectK);
        CASE(kSend);
        CASE(kSendDone);
        CASE(kSequential);
        CASE(kTriangularSolve);
        CASE(kWaitForStreams);
        CASE(kWhile);
        // # go/keep-sorted end
    }
}

/*static*/
absl::StatusOr<se::Stream*> NpuThunk::GetStreamForExecution(NpuExecutionStreamId stream_id, const ExecuteParams& params) {
    if (stream_id == kDefaultExecutionStreamId) {
        return params.stream;
    }
    auto iter = params.additional_compute_streams.find(stream_id);
    if (iter == params.additional_compute_streams.end()) {
        return absl::InvalidArgumentError("Invalid execution stream id.");
    }
    return iter->second;
}

std::ostream& operator<<(std::ostream& os, NpuThunk::Kind kind) { return os << NpuThunk::KindToString(kind); }

bool IsReductionCollective(NpuThunk::Kind kind) {
    return kind == NpuThunk::kAllReduce || kind == NpuThunk::kAllReduceStart || kind == NpuThunk::kReduceScatter ||
           kind == NpuThunk::kReduceScatterStart || kind == NpuThunk::kNvshmemAllReduceStart;
    ;
}

absl::StatusOr<NpuThunk::ThunkInfo> NpuThunk::ThunkInfo::FromProto(const ThunkInfoProto& proto) {
    TF_RET_CHECK(proto.execution_stream_id() >= 0)
        << "The thunk execution stream ID must be non-negative, but got " << proto.execution_stream_id() << ".";
    NpuThunk::ThunkInfo thunk_info;
    thunk_info.profile_annotation = proto.profile_annotation();
    thunk_info.execution_stream_id = proto.execution_stream_id();
    thunk_info.thunk_id = ThunkId(proto.thunk_id());
    return thunk_info;
}

NpuThunk::ThunkInfo NpuThunk::ThunkInfo::WithProfileAnnotation(const HloInstruction* instr, ThunkId thunk_id) {
    ThunkInfo thunk_info;
    thunk_info.profile_annotation = instr->name();
    thunk_info.thunk_id = thunk_id;
    auto gpu_backend_config = instr->backend_config<GpuBackendConfig>();
    if (gpu_backend_config.ok()) {
        thunk_info.execution_stream_id =
            std::max<uint64_t>(kDefaultExecutionStreamId.value(), gpu_backend_config->operation_queue_id());
    }

    HloPrintOptions options = NpuTool::GetDetailedPrintOptions();
    thunk_info.hloInstrDesc = instr->ToString(options);
    thunk_info.op_name = std::string(instr->name());
    thunk_info.op_code = instr->opcode();
    for (int i = 0; i < instr->operand_count(); ++i) {
        thunk_info.param_shapes.push_back(instr->operand(i)->shape());
    }
    thunk_info.param_shapes.push_back(instr->shape());

    return thunk_info;
}

bool NpuThunk::IsCollective() const {
    switch (kind()) {
        // go/keep-sorted start
        case kAllGather:
        case kAllGatherDone:
        case kAllGatherStart:
        case kAllReduce:
        case kAllReduceDone:
        case kAllReduceStart:
        case kAllToAll:
        case kAllToAllDone:
        case kAllToAllStart:
        case kCollectiveBroadcast:
        case kCollectiveBroadcastDone:
        case kCollectiveBroadcastStart:
        case kCollectivePermute:
        case kCollectivePermuteDone:
        case kCollectivePermuteStart:
        case kGroupDone:
        case kGroupStart:
        case kRaggedAllToAll:
        case kRaggedAllToAllDone:
        case kRaggedAllToAllStart:
        case kRecv:
        case kRecvDone:
        case kReduceScatter:
        case kReduceScatterDone:
        case kReduceScatterStart:
        case kSend:
        case kSendDone:
            // go/keep-sorted end
            return true;
        default:
            return false;
    }
}

void NpuThunk::ForAllThunks(absl::FunctionRef<void(const NpuThunk*)> fn) const { fn(this); }

void NpuThunk::ForAllThunksMutable(absl::FunctionRef<void(NpuThunk*)> fn) { fn(this); }

absl::StatusOr<ThunkProto> NpuThunk::ToProto() const {
    return absl::UnimplementedError(
        absl::StrFormat("Proto serialization for thunk of type %s is not implemented", typeid(*this).name()));
}

ThunkMetadataProto NpuThunk::ToMetadataProto() const {
    ThunkMetadataProto metadata_proto;
    *metadata_proto.mutable_thunk_info() = thunk_info_.ToProto();
    metadata_proto.set_thunk_kind(KindToString(kind_));
    return metadata_proto;
}

ThunkMetadataListProto GetMetadataListProtoFromThunkGraph(const NpuThunk& root_thunk) {
    ThunkMetadataListProto metadata_list_proto;
    root_thunk.ForAllThunks([&metadata_list_proto](const NpuThunk* thunk) {
        *metadata_list_proto.add_thunk_metadata() = thunk->ToMetadataProto();
    });
    return metadata_list_proto;
}

absl::StatusOr<GpuCollectives* absl_nonnull> NpuThunk::GetGpuCollectives(CollectiveExecuteParams const& params) {
    if (params.collectives == nullptr) {
        return Internal("Collectives API is not provided");
    }
    return params.collectives;
}

ThunkInfoProto NpuThunk::ThunkInfo::ToProto() const {
    ThunkInfoProto proto;
    proto.set_profile_annotation(profile_annotation);
    proto.set_execution_stream_id(execution_stream_id.value());
    proto.set_thunk_id(thunk_id.value());
    return proto;
}

}  // namespace gpu
}  // namespace xla
