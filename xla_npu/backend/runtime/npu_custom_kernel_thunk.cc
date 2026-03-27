/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/backend/runtime/npu_custom_kernel_thunk.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/tool/config_loader.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/kernel/npu_custom_kernel.h"

namespace xla {

NpuCustomKernelThunk::NpuCustomKernelThunk(
    NpuThunk::ThunkInfo thunk_info, NpuCustomKernel custom_kernel,
    const emitters::KernelArguments& kernel_arguments)
    : NpuThunk(Kind::kCustomKernel, std::move(thunk_info)),
      args_(kernel_arguments.GetArgumentBufferSlices()),
      written_(kernel_arguments.GetArgumentOutputFlags()),
      custom_kernel_(std::move(custom_kernel)) {}

std::string NpuCustomKernelThunk::ToString(int indent) const {
  return custom_kernel_.ToString();
}

absl::Status NpuCustomKernelThunk::Initialize(const InitializeParams& params) {
  absl::MutexLock lock(mutex_);

  if (!kernel_cache_.contains(params.executor)) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::Kernel> kernel,
        params.executor->LoadKernel(custom_kernel_.kernel_spec()));
    kernel_cache_.emplace(params.executor, std::move(kernel));
  }

  return absl::OkStatus();
}

void NpuCustomKernelThunk::PrintKernelLaunchInfo(int device_ordinal,
                                                 se::Kernel* kernel) const {
  NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
  NPU_LOG_FORMAT_PRINT(
      DEBUG,
      "==== Launch npu custom fusion kernel with %d params ====", args_.size());
  NPU_LOG_FORMAT_PRINT(DEBUG, "---- Corresponding hlo instruction: %s ----",
                       thunk_info().hloInstrDesc);
  NPU_LOG_FORMAT_PRINT(
      DEBUG, "Device id: %d, launching %s as npu fusion kernel %s",
      device_ordinal, custom_kernel_.ToString(), kernel->name());
}

void NpuCustomKernelThunk::PrintKernelSpec() const {
  NPU_LOG_FORMAT_PRINT(DEBUG, "---- Kernel spec: ----");
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Kernel name: %s",
                       custom_kernel_.kernel_spec().kernel_name().c_str());
  NPU_LOG_FORMAT_PRINT(
      DEBUG, "  Thread dims: (%d, %d, %d)", custom_kernel_.thread_dims().x,
      custom_kernel_.thread_dims().y, custom_kernel_.thread_dims().z);
  NPU_LOG_FORMAT_PRINT(
      DEBUG, "  Block dims: (%d, %d, %d)", custom_kernel_.block_dims().x,
      custom_kernel_.block_dims().y, custom_kernel_.block_dims().z);
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Shared memory: %d bytes",
                       custom_kernel_.shared_memory_bytes());

  // Print NPU kernel config and computation graphs using base class functions
  auto* kernel_config = custom_kernel_.npu_kernel_config_ptr();
  NpuThunk::PrintKernelConfig(kernel_config);
  if (kernel_config != nullptr) {
    NpuThunk::PrintComputationGraphs(kernel_config->computation_hlo_map,
                                     kernel_config->npu_artifact_info.name);
  }
}

void NpuCustomKernelThunk::PrintParamShapes() const {
  NPU_LOG_FORMAT_PRINT(DEBUG, "---- Param shapes (%d total): ----",
                       thunk_info().param_shapes.size());
  for (size_t i = 0; i < thunk_info().param_shapes.size(); ++i) {
    const std::string& param_type =
        (i < thunk_info().param_shapes.size() - 1) ? "input" : "output";
    NPU_LOG_FORMAT_PRINT(DEBUG, "  Param %d (%s): %s", i, param_type.c_str(),
                         thunk_info().param_shapes[i].ToString(true).c_str());
  }
}

void NpuCustomKernelThunk::PrintParamInfoBeforeExecute(
    int device_ordinal, const NpuBufferAllocations* buffer_allocations,
    absl::InlinedVector<se::DeviceMemoryBase, 4>& buffer_args) {
  int id = 0;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf = buffer_allocations->GetDeviceAddress(arg);
    buffer_args.push_back(buf);
    if (IsPrintIOValuesEnabled()) {
      const std::string& param_type = written_[id] ? "output" : "input";
      PrintParamInfo(id, param_type, device_ordinal, arg,
                     thunk_info().param_shapes[id], buf.opaque());
    }
    id++;
  }
}

void NpuCustomKernelThunk::PrintParamInfoAfterExecute(
    int device_ordinal, const NpuBufferAllocations* buffer_allocations) {
  int id = 0;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf = buffer_allocations->GetDeviceAddress(arg);
    const std::string& param_type = written_[id] ? "output" : "input";
    PrintParamInfo(id, param_type, device_ordinal, arg,
                   thunk_info().param_shapes[id], buf.opaque());
    id++;
  }
}

absl::Status NpuCustomKernelThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  if (params.stream == nullptr) {
    const std::string& errMsg = absl::StrFormat(
        "Xla stream need to be created or init before execute npu thunk.");
    NPU_LOG_FORMAT_PRINT(ERROR, errMsg);
    return absl::InternalError(errMsg);
  }
  se::StreamExecutor* executor = params.stream->parent();
  se::Kernel* kernel = [&] {
    absl::MutexLock lock(mutex_);
    return kernel_cache_[executor].get();
  }();
  int device_ordinal = executor->device_ordinal();
  // Print HLO computation info similar to npu_aclnn_thunk.cc
  PrintKernelLaunchInfo(device_ordinal, kernel);
  PrintKernelSpec();
  PrintParamShapes();
  absl::InlinedVector<se::DeviceMemoryBase, 4> buffer_args;
  PrintParamInfoBeforeExecute(device_ordinal, params.buffer_allocations,
                              buffer_args);
  se::KernelArgsDeviceMemoryArray args(buffer_args,
                                       custom_kernel_.shared_memory_bytes());
  auto ret =
      kernel->Launch(custom_kernel_.thread_dims(), custom_kernel_.block_dims(),
                     custom_kernel_.cluster_dims(), params.stream, args);

  // 使用统一的流同步函数
  TF_RETURN_IF_ERROR(SyncStreamIfEnabled(params.stream));

  // 根据环境变量控制是否打印输入输出值(包含d2h操作,影响性能)
  // 打印输出值需要先同步流,确保数据已从设备复制到主机
  if (NPU_CONFIG_GET_BOOL(NpuConfig::Key::ENABLE_PRINT_IO_VALUES)) {
    // 先同步流,确保算子执行完成
    TF_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
    PrintParamInfoAfterExecute(device_ordinal, params.buffer_allocations);
  }

  return ret;
}

NpuThunk::BufferUses NpuCustomKernelThunk::buffer_uses() const {
  NpuThunk::BufferUses buffers;
  buffers.reserve(args_.size());
  for (int i = 0; i < args_.size(); ++i) {
    // We assume that any buffer is either an input or an output of the
    // kernel, and inout buffers are represented as 2 separate arguments.
    if (written_[i]) {
      buffers.push_back(BufferUse::Write(args_[i]));
    } else {
      buffers.push_back(BufferUse::Read(args_[i]));
    }
  }
  return buffers;
}

}  // namespace xla
