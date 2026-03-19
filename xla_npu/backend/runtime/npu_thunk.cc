/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/backend/runtime/npu_thunk.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
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
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/global_device_id.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_npu/backend/runtime/npu_thunk_id.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/kernel/npu_kernel_config.h"
#include "xla/xla_npu/service/npu_buffer_allocations.h"

namespace xla {

//===----------------------------------------------------------------------===//
// NpuThunk::ExecuteParams
//===----------------------------------------------------------------------===//

NpuThunk::ExecuteParams NpuThunk::ExecuteParams::Create(
    const ServiceExecutableRunOptions& run_options,
    const NpuBufferAllocations& buffer_allocations, se::Stream* stream,
    se::Stream* command_buffer_trace_stream,
    ExecutionStreamIdMap additional_compute_streams) {
  return ExecuteParams(&buffer_allocations, stream, command_buffer_trace_stream,
                       run_options.run_options().device_to_host_stream(),
                       run_options.run_options().host_to_device_stream(),
                       run_options.run_options().send_device_memory_function(),
                       run_options.run_options().recv_device_memory_function(),
                       run_options.run_options().ffi_execution_context(),
                       additional_compute_streams, false,
                       run_options.run_options().run_id().ToInt());
}

NpuThunk::ExecuteParams NpuThunk::ExecuteParams::CloneWithNewAllocations(
    const NpuThunk::ExecuteParams& params,
    const NpuBufferAllocations& buffer_allocations) {
  return ExecuteParams(
      &buffer_allocations, params.stream, params.command_buffer_trace_stream,
      params.device_to_host_stream, params.host_to_device_stream,
      params.send_device_memory_function, params.recv_device_memory_function,
      params.ffi_execution_context, params.additional_compute_streams);
}

NpuThunk::ExecuteParams::ExecuteParams(
    const NpuBufferAllocations* buffer_allocations, se::Stream* stream,
    se::Stream* command_buffer_trace_stream, se::Stream* device_to_host_stream,
    se::Stream* host_to_device_stream,
    SendDeviceMemoryFunction* send_device_memory_function,
    RecvDeviceMemoryFunction* recv_device_memory_function,
    const ffi::ExecutionContext* ffi_execution_context,
    ExecutionStreamIdMap additional_compute_streams, bool mock_collectives,
    int64_t execution_id)
    : buffer_allocations(buffer_allocations),
      stream(stream),
      command_buffer_trace_stream(command_buffer_trace_stream),
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
#define CASE(x)     \
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
    CASE(kCall);
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
absl::StatusOr<se::Stream*> NpuThunk::GetStreamForExecution(
    NpuExecutionStreamId stream_id, const ExecuteParams& params) {
  if (stream_id == kDefaultExecutionStreamId) {
    return params.stream;
  }
  auto iter = params.additional_compute_streams.find(stream_id);
  if (iter == params.additional_compute_streams.end()) {
    return absl::InvalidArgumentError("Invalid execution stream id.");
  }
  return iter->second;
}

std::ostream& operator<<(std::ostream& os, NpuThunk::Kind kind) {
  return os << NpuThunk::KindToString(kind);
}

bool IsReductionCollective(NpuThunk::Kind kind) {
  return kind == NpuThunk::kAllReduce || kind == NpuThunk::kAllReduceStart ||
         kind == NpuThunk::kReduceScatter ||
         kind == NpuThunk::kReduceScatterStart ||
         kind == NpuThunk::kNvshmemAllReduceStart;
  ;
}

NpuThunk::ThunkInfo NpuThunk::ThunkInfo::WithProfileAnnotation(
    const HloInstruction* instr, NpuThunkId thunk_id) {
  ThunkInfo thunk_info;
  thunk_info.profile_annotation = instr->name();
  thunk_info.thunk_id = thunk_id;

  HloPrintOptions options = NpuTool::GetDetailedPrintOptions();
  thunk_info.hloInstrDesc = instr->ToString(options);
  thunk_info.op_name = std::string(instr->name());
  thunk_info.op_code = instr->opcode();
  for (int i = 0; i < instr->operand_count(); ++i) {
    thunk_info.param_shapes.push_back(instr->operand(i)->shape());
  }
  thunk_info.param_shapes.push_back(instr->shape());

  // 保存dimensions信息,用于concatenate等算子
  // 只有HloDimensionsInstruction类型的指令才有dimensions属性
  if (HloDimensionsInstruction::ClassOf(instr)) {
    auto dimensions_span = instr->dimensions();
    thunk_info.dimensions =
        std::vector<int64_t>(dimensions_span.begin(), dimensions_span.end());
  }

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

void NpuThunk::ForAllThunks(absl::FunctionRef<void(const NpuThunk*)> fn) const {
  fn(this);
}

void NpuThunk::ForAllThunksMutable(absl::FunctionRef<void(NpuThunk*)> fn) {
  fn(this);
}

// 在 Initialize 阶段处理所有常量 buffer 中的特殊值
// 这些特殊值 (-inf, +inf, nan) 只在图分配时写入，计算过程中不会产生
absl::Status NpuThunk::InitializeSpecialValues(
    const InitializeParams& params,
    const std::vector<BufferAllocation::Slice>& args) {
  const auto* allocations = params.buffer_allocations;

  // 遍历所有 args，检查并替换特殊值
  // 注意：我们只处理标量浮点类型的 buffer
  for (const auto& slice : args) {
    // 获取 buffer 对应的 shape (从 thunk_info 中)
    // 由于我们没有直接的 shape 信息，这里使用一个简单的策略:
    // 检查所有小的 buffer (< 16 bytes)，这些很可能是标量常量
    auto addr = allocations->GetDeviceAddress(slice);
    void* device_addr = addr.opaque();
    size_t size = addr.size();
    // 只处理小 buffer (可能是标量)
    if (size > sizeof(float) * 4) {
      continue;
    }

    // 只处理 4 字节或 8 字节的 buffer (float 或 double)
    if (size != sizeof(float) && size != sizeof(double)) {
      continue;
    }

    // 尝试作为 float 处理
    if (size == sizeof(float)) {
      float value_host = 0.0f;
      aclError ret = npu::aclrtMemcpy(&value_host, sizeof(float), device_addr,
                                      sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
      if (ret != ACL_SUCCESS) {
        continue;  // 跳过读取失败的
      }

      // 检查并替换特殊值
      bool need_replace = false;
      float replacement_value = 0.0f;

      if (std::isinf(value_host) && value_host < 0) {
        need_replace = true;
        replacement_value = -3.4028235e+38f;  // -FLT_MAX
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Initialize: Detected -inf in buffer (slice "
                             "%d:%d), replacing with -FLT_MAX",
                             slice.index(), slice.offset());
      } else if (std::isinf(value_host) && value_host > 0) {
        need_replace = true;
        replacement_value = 3.4028235e+38f;  // FLT_MAX
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Initialize: Detected +inf in buffer (slice "
                             "%d:%d), replacing with FLT_MAX",
                             slice.index(), slice.offset());
      } else if (std::isnan(value_host)) {
        need_replace = true;
        replacement_value = 0.0f;
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Initialize: Detected nan in buffer (slice "
                             "%d:%d), replacing with 0.0f",
                             slice.index(), slice.offset());
      }

      if (need_replace) {
        ret = npu::aclrtMemcpy(device_addr, sizeof(float), &replacement_value,
                               sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret == ACL_SUCCESS) {
          NPU_LOG_FORMAT_PRINT(
              DEBUG,
              "Initialize: Successfully replaced special value in buffer");
        }
      }
    }
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===n// Print Functions for Debugging
//===----------------------------------------------------------------------===n
/*static*/ void NpuThunk::PrintKernelConfig(const NpuKernelConfig* kernel_config) {
  if (kernel_config == nullptr) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Kernel config is null");
    return;
  }
  
  NPU_LOG_FORMAT_PRINT(DEBUG, "---- NPU Kernel Config: ----");
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Kernel type: %d", 
                       static_cast<int>(kernel_config->npu_kernel_type));
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Artifact name: %s", 
                       kernel_config->npu_artifact_info.name.c_str());
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Artifact dir: %s", 
                       kernel_config->npu_artifact_info.dir.c_str());
}

/*static*/ void NpuThunk::PrintComputationGraphs(
    const std::unordered_map<std::string, std::string>& computation_hlo_map,
    const std::string& entry_computation_name) {
  if (computation_hlo_map.empty()) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "No computation graphs available");
    return;
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "---- Computation Graphs (%d total): ----",
                       computation_hlo_map.size());
  for (const auto& [name, hlo_text] : computation_hlo_map) {
    bool is_entry = (name == entry_computation_name);
    PrintComputationHlo(hlo_text, name, is_entry);
  }
}

/*static*/ void NpuThunk::PrintComputationHlo(const std::string& hlo_text,
                                               const std::string& name,
                                               bool is_entry) {
  if (hlo_text.empty()) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Computation %s HLO is empty", name.c_str());
    return;
  }

  const char* entry_marker = is_entry ? " [ENTRY]" : "";
  NPU_LOG_FORMAT_PRINT(DEBUG, "==== Computation: %s%s ====",
                       name.c_str(), entry_marker);

  // Print HLO text line by line
  size_t start = 0;
  size_t end = hlo_text.find('\n');
  while (end != std::string::npos) {
    std::string line = hlo_text.substr(start, end - start);
    if (!line.empty()) {
      NPU_LOG_FORMAT_PRINT(DEBUG, "  %s", line.c_str());
    }
    start = end + 1;
    end = hlo_text.find('\n', start);
  }
  // Print last line if any
  if (start < hlo_text.length()) {
    std::string line = hlo_text.substr(start);
    if (!line.empty()) {
      NPU_LOG_FORMAT_PRINT(DEBUG, "  %s", line.c_str());
    }
  }
  NPU_LOG_FORMAT_PRINT(DEBUG, "==== End of Computation: %s ====",
                       name.c_str());
}

}  // namespace xla
