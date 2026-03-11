/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/global_device_id.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "xla/util.h"
#include "xla/xla_npu/backend/runtime/npu_thunk_id.h"
#include "xla/xla_npu/service/npu_buffer_allocations.h"

namespace xla {

// Execution stream id allows to specify what Gpu stream NpuThunk should be
// using for launching device work (kernels, library calls, etc.). By default
// all thunks use stream #0, which is the default compute stream of an XLA
// executable.
//
// Stream synchronizations are explicit and represented as WaitForStreams thunk
// in a NpuThunkSequence. When NpuThunkSequence converted to CommandBuffer,
// execution streams mapped to concurrent execution scopes and barriers between
// them.
//
// IMPORTANT: Async execution semantics and execution stream id
//
// For async thunks (i.e. thunks corresponding to `all-reduce-start` and
// `all-reduce-done`) execution stream id means NOT a stream where the async
// operation must execute, but a stream that async operation must be
// synchronized with:
//
//   - Start operation must wait for the completion of all launched work on the
//     execution stream id (usually by adding a stream wait) and after that
//     launch async work on implementation defined extra stream (can be borrowed
//     from a pool)
//
//   - Corresponding Done operation must synchronize execution stream id with
//     an implementation defined stream that is running async work, again
//     usually by adding a stream wait.
//
TSL_LIB_GTL_DEFINE_INT_TYPE(NpuExecutionStreamId, uint64_t);

// Unique identifier for async events. The same identifier is expected to be
// shared between a pair of StartThunk and corresponding DoneThunk. It is used
// to collect async regions for a CommandBufferThunk.
TSL_LIB_GTL_DEFINE_INT_TYPE(NpuAsyncEventsUniqueId, uint64_t);

// NpuThunk acts as the bridge between IrEmitter and GpuExecutable. It stores
// the metadata IrEmitter generates for GpuExecutable to invoke an
// HloInstruction.
//
// NpuThunk provides the Initialize and ExecuteOnStream interface for
// GpuExecutable to initialize and execute the invocation respectively. Its
// subclasses are supposed to override these interfaces to launch a generated
// kernel or call an external library function (such as operations in cuBLAS).
//
// Thunks have three execution stages:
//
// (1) Prepare: at this stage NpuThunk can request shared resources required at
// run
//     time, i.e. collective thunks request collective cliques. Executable(s)
//     will coordinate resource acquisition.
//
// (2) Initialize: at this stage NpuThunk must initialize all internal state
//     required for execution, maybe using resources requested at prepare stage.
//
// (3) Execute: at this stage NpuThunk must launch "work" on underlying device
//     using given stream, and it's expected that all expensive initialization
//     is completed at earlier stages.
//
// This is thread-compatible. NpuThunk implementation should expect that it will
// be called concurrently from multiple threads, for different run ids and for
// different devices (stream executors). For partitioned XLA programs the
// expectation is that all local participants execute simultaneously on
// different threads and coordinate resource acquisition via rendezvous.
class NpuThunk {
 public:
  using ExecutionStreamIdMap =
      absl::flat_hash_map<NpuExecutionStreamId, se::Stream*>;

  // When default execution stream id is used, operations launched by a thunk
  // must be synchronized with a stream passed in ExecuteOptions.
  static constexpr auto kDefaultExecutionStreamId = NpuExecutionStreamId(0);

  enum Kind {
    // # go/keep-sorted start
    kAllGather,
    kAllGatherDone,
    kAllGatherStart,
    kAllReduce,
    kAllReduceDone,
    kAllReduceStart,
    kAllToAll,
    kAllToAllDone,
    kAllToAllStart,
    kBuffersDebugChecksum,
    kBuffersDebugFloatCheck,
    kCollectiveBroadcast,
    kCollectiveBroadcastDone,
    kCollectiveBroadcastStart,
    kCollectiveKernel,
    kCollectivePermute,
    kCollectivePermuteDone,
    kCollectivePermuteStart,
    kCall,
    kCommandBuffer,
    kConditional,
    kConvolution,
    kConvolutionReorder,
    kCopy,
    kCopyDone,
    kCuDnn,
    kCubSort,
    kCublasLtMatmul,
    kCustomCall,
    kCustomKernel,
    kDynamicSlice,
    kFft,
    kGemm,
    kGroupDone,
    kGroupStart,
    kHostExecuteDone,
    kHostExecuteStart,
    kHostRecv,
    kHostRecvDone,
    kHostSend,
    kHostSendDone,
    kInfeed,
    kKernel,
    kMemset32BitValue,
    kMemzero,
    kNorm,
    kNvshmemAllReduceDone,
    kNvshmemAllReduceStart,
    kNvshmemCollectivePermute,
    kNvshmemCollectivePermuteDone,
    kNvshmemCollectivePermuteStart,
    kNvshmemRecv,
    kNvshmemRecvDone,
    kNvshmemSend,
    kNvshmemSendDone,
    kOutfeed,
    kPartitionId,
    kRaggedAllToAll,
    kRaggedAllToAllDone,
    kRaggedAllToAllStart,
    kRecv,
    kRecvDone,
    kReduceScatter,
    kReduceScatterDone,
    kReduceScatterStart,
    kReplicaId,
    kSelectK,
    kSend,
    kSendDone,
    kSequential,
    kTriangularSolve,
    kWaitForStreams,
    kWhile
    // go/keep-sorted end
  };

  // TODO(ezhulenev): This should become a part of StreamExecutor library, but
  // for now we keep it here as a NpuThunk implementation detail. It's not yet
  // clear what else should become a part of "executable source", we likely
  // need to keep some information about available symbols and signatures.
  struct ExecutableSource {
    absl::string_view text;            // PTX for NVIDIA backend
    absl::Span<const uint8_t> binary;  // CUBIN for NVIDIA backends
                                       // BinaryMap dnn_compiled_graphs;
  };

  // Metadata associated with a NpuThunk,
  // including profiling and stream execution info.
  struct ThunkInfo {
    ThunkInfo() = default;  // Disable implicit constructors.

    static ThunkInfo WithProfileAnnotation(const HloInstruction* instr,
                                           NpuThunkId thunk_id);

    std::string profile_annotation;

    NpuExecutionStreamId execution_stream_id = kDefaultExecutionStreamId;

    NpuThunkId thunk_id = NpuThunkId{0};

    // 提供给aclnn使用
    std::string hloInstrDesc;  // 保留hlo instruction描述,便于debug
    std::string op_name;
    HloOpcode op_code;
    std::vector<Shape>
        param_shapes;  // 与instruction中参数顺序一致, 最后一个位置表示出参shape
    std::vector<int64_t> dimensions;  // 用于concatenate等需要维度的算子
  };

  //===--------------------------------------------------------------------===//
  // ResourceRequests
  //===--------------------------------------------------------------------===//

  // Each individual thunk can request various resources required for execution
  // at prepare stage. XLA executable is responsible for allocating them before
  // initializing and executing thunks.
  class ResourceRequestsInterface {
   public:
    virtual ~ResourceRequestsInterface() = default;
  };

  //===--------------------------------------------------------------------===//
  // PrepareParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to Prepare. At thunk prepare time we do not launch any
  // work or do any expensive initialization and only pass resource requirements
  // back to executable, i.e. request collective cliques required at run time.
  struct PrepareParams {
  };

  //===--------------------------------------------------------------------===//
  // InitializeParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to Initialize. At thunk initialization time we do not
  // launch any "work" on device and only initialize thunks for execution, i.e.
  // we pre-load kernels on device and instantiate all command buffers.
  struct InitializeParams {
    se::StreamExecutor* executor = nullptr;
    ExecutableSource src;

    const NpuBufferAllocations* buffer_allocations = nullptr;

    // Main compute stream that will be used, passed via `ExecuteParams` to
    // `ExecuteOnStream`. It can be used to initialize on-device "state" (i.e.
    // various control structures) at command buffer recording time (we use it
    // to initialize NCCL execution plans on device when we trace NCCL
    // operations into command buffers);
    se::Stream* stream = nullptr;

    // Auxiliary stream for tracing command buffers. We use a separate stream to
    // avoid accidental tracing of unrelated activities on a main stream.
    se::Stream* command_buffer_trace_stream = nullptr;

    // XLA FFI execution context.
    const ffi::ExecutionContext* ffi_execution_context = nullptr;

    // Total local device count.
    int local_device_count = 0;
  };

  //===--------------------------------------------------------------------===//
  // ExecuteParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to ExecuteOnStream. ExecuteOnStream is responsible for
  // launching "work" on device, i.e. it launches kernels, executes command
  // buffers and calls into libraries (cuBLAS, cuDNN etc.).
  struct ExecuteParams {
    // Constructs execute parameters from an executable run options. Return
    // error if run options are misconfigured.
    static ExecuteParams Create(
        const ServiceExecutableRunOptions& run_options,
        const NpuBufferAllocations& buffer_allocations, se::Stream* stream,
        se::Stream* command_buffer_trace_stream,
        ExecutionStreamIdMap additional_compute_streams = {});

    // Constructs execute parameters from an existing parameters but with
    // different buffer allocations.
    static ExecuteParams CloneWithNewAllocations(
        const ExecuteParams& params,
        const NpuBufferAllocations& buffer_allocations);

    const NpuBufferAllocations* buffer_allocations;  // never null

    // Main compute stream on which thunks launch operations.
    se::Stream* stream;

    // Auxiliary stream for tracing command buffers. We use a separate stream to
    // avoid accidental tracing of unrelated activities on a main stream.
    se::Stream* command_buffer_trace_stream;

    // Streams for moving data between host and device.
    se::Stream* device_to_host_stream;
    se::Stream* host_to_device_stream;

    // Send/Recv callbacks passed to XLA from PjRt.
    SendDeviceMemoryFunction* send_device_memory_function;
    RecvDeviceMemoryFunction* recv_device_memory_function;

    // XLA FFI execution context.
    const ffi::ExecutionContext* ffi_execution_context;

    // Additional compute streams on which thunks launch operations.
    ExecutionStreamIdMap additional_compute_streams;

    bool mock_collectives = false;

    int64_t execution_id = 0;

   private:
    friend class CommandBufferThunk;

    ExecuteParams(const NpuBufferAllocations* buffer_allocations,
                  se::Stream* stream, se::Stream* command_buffer_trace_stream,
                  se::Stream* device_to_host_stream,
                  se::Stream* host_to_device_stream,
                  SendDeviceMemoryFunction* send_device_memory_function,
                  RecvDeviceMemoryFunction* recv_device_memory_function,
                  const ffi::ExecutionContext* ffi_execution_context,
                  ExecutionStreamIdMap additional_compute_streams = {},
                  bool mock_collectives = false, int64_t execution_id = 0);
  };

  //===--------------------------------------------------------------------===//

  NpuThunk(Kind kind, ThunkInfo thunk_info)
      : kind_(kind), thunk_info_(std::move(thunk_info)) {}
  virtual ~NpuThunk() = default;
  NpuThunk(const NpuThunk&) = delete;
  NpuThunk& operator=(const NpuThunk&) = delete;

  virtual std::string ToString(int indent) const { return ""; }
  Kind kind() const { return kind_; }
  absl::string_view profile_annotation() const {
    return thunk_info_.profile_annotation;
  }
  const ThunkInfo& thunk_info() const { return thunk_info_; }

  // Prepares thunk for execution.
  //
  // This may be called multiple times. Its main purpose is to pass resource
  // requests up to the parent executable so it can acquire them before
  // initialization and execution.
  virtual absl::Status Prepare(const PrepareParams& params,
                               ResourceRequestsInterface& resource_requests) {
    return absl::OkStatus();
  }

  // Initializes thunk for execution.
  //
  // This may be called multiple times. Its main purpose is to give us a chance
  // to do initialization outside of ExecuteOnStream() so that the
  // time spent initializing doesn't count towards our execution profile.
  //
  // Precondition: Prepare(initialize_params) has been called.
  virtual absl::Status Initialize(const InitializeParams& params) {
    return absl::OkStatus();
  }

  // 在 Initialize 阶段处理所有常量 buffer 中的特殊值 (-inf, +inf, nan)
  // 这些特殊值只在图分配时写入，计算过程中不会产生，所以在初始化阶段统一处理
  // 参数：params - 初始化参数，args - thunk 的 buffer slices
  absl::Status InitializeSpecialValues(
      const InitializeParams& params,
      const std::vector<BufferAllocation::Slice>& args);

  // Executes thunk on the given stream. This method must be called after
  // Initialize and can be called multiple times over NpuThunk's lifetime.
  //
  // Precondition: Initialize(initialize_params) has been called.
  virtual absl::Status ExecuteOnStream(const ExecuteParams& params) = 0;

  using BufferUses = absl::InlinedVector<BufferUse, 4>;

  // Returns all device buffers used by the thunk.
  //
  // Does not propagate buffers from nested thunks.
  //
  // The order of the buffers in returned vector is consistent across calls.
  virtual BufferUses buffer_uses() const { return {}; }

  static absl::string_view KindToString(NpuThunk::Kind kind);

  NpuExecutionStreamId execution_stream_id() const {
    return thunk_info_.execution_stream_id;
  }
  void set_execution_stream_id(NpuExecutionStreamId execution_stream_id) {
    thunk_info_.execution_stream_id = execution_stream_id;
  }

  static absl::StatusOr<se::Stream*> GetStreamForExecution(
      NpuExecutionStreamId stream_id, const ExecuteParams& params);

  // Returns `true` if this thunk requires inter-GPU communication.
  bool IsCollective() const;

  // Returns any communicators used during execution.
  virtual absl::StatusOr<std::vector<Communicator*>> GetCommunicators(
      const ExecuteParams& params) const {
    return std::vector<Communicator*>();
  }

  // Invokes `fn` with this thunk and all nested thunks.
  virtual void ForAllThunks(absl::FunctionRef<void(const NpuThunk*)> fn) const;

  // Invokes `fn` with this thunk and all nested thunks.
  virtual void ForAllThunksMutable(absl::FunctionRef<void(NpuThunk*)> fn);

  // Recursively replaces all nested thunks with the result of applying `fn` to
  // them.
  // An error will leave the transformation in invalid state.
  // InternalError should be used for status.
  virtual absl::Status TransformAllNestedThunks(
      absl::FunctionRef<
          absl::StatusOr<std::unique_ptr<NpuThunk>>(std::unique_ptr<NpuThunk>)>
          fn) {
    return absl::OkStatus();
  }

  void add_control_predecessor(const NpuThunk* control_predecessor) {
    control_predecessors_.push_back(control_predecessor);
  }

  std::vector<const NpuThunk*> control_predecessors() const {
    return control_predecessors_;
  }

  virtual std::optional<NpuAsyncEventsUniqueId> GetAsyncEventsUniqueId() const {
    return std::nullopt;
  }

  virtual bool IsAsyncStart() const { return false; }

  virtual bool IsAsyncDone() const { return false; }

 protected:
 private:
  Kind kind_;
  ThunkInfo thunk_info_;

  // The list of control predecessors of the thunk.
  // NpuThunk needs to maintain the control dependency information because
  // when it is executed by command buffer, and command buffer may execute the
  // sequence in concurrent mode, and we should make sure that it does not
  // violate the control dependency in the original computation.
  std::vector<const NpuThunk*> control_predecessors_;
};

// A sequence of thunks.
using NpuThunkSequence = std::vector<std::unique_ptr<NpuThunk>>;

std::ostream& operator<<(std::ostream& os, NpuThunk::Kind kind);

// Returns if the thunk implements a reduction collective (all-reduce or
// reduce-scatter).
bool IsReductionCollective(NpuThunk::Kind kind);

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_H_
