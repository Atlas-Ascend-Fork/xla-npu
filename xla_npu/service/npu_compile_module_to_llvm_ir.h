/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_NPU_COMPILE_MODULE_TO_LLVM_IR_H_
#define XLA_XLA_NPU_SERVICE_NPU_COMPILE_MODULE_TO_LLVM_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
// #include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
// #include "xla/service/gpu/alias_info.h"
// #include "xla/service/gpu/execution_stream_assignment.h"
// #include "xla/service/gpu/gpu_executable.h"
// #include "xla/service/gpu/ir_emitter_context.h"
// #include "xla/service/gpu/kernel_reuse_cache.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/util.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"
#include "xla/xla_npu/service/npu_alias_info.h"
#include "xla/xla_npu/service/npu_executable.h"
#include "xla/xla_npu/service/npu_execution_stream_assignment.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

namespace xla {

struct NpuCompileModuleResults {
    std::unique_ptr<llvm::Module> llvm_module;
    std::unique_ptr<llvm::Module> llvm_module_constants;
    std::unique_ptr<BufferAssignment> buffer_assignment;
    std::unique_ptr<NpuExecutionStreamAssignment> execution_stream_assignment;
    std::vector<BufferAllocation> allocations;
    std::unique_ptr<NpuSequentialThunk> executable;
    std::vector<NpuExecutable::ConstantInfo> constants;
    absl::flat_hash_map<ShapeIndex, NpuExecutable::OutputInfo> output_info;
    Shape output_shape;
    std::string module_name;
    // CompilationCacheProto kernel_compilation_cache;

    // If true, the compiled module uses buffer allocations owned by
    // buffer_assignment. Otherwise the compiled module uses buffer allocations
    // stored in allocations.
    bool use_original_allocations;
};

// absl::Status NpuLoadCache(NpuIrEmitterContext& ir_emitter_context, absl::string_view cache_file_path);

absl::StatusOr<NpuCompileModuleResults> NpuCompileModuleToLlvmIr(
    const HloModule* hlo_module, llvm::LLVMContext* llvm_context, const std::string& target_triple,
    const std::string& data_layout, const se::Platform* platform, const se::DeviceDescription& device_desc,
    const NpuAliasInfo* alias_info, BufferValue::SizeFunction buffer_size_bytes_function,
    bool split_constants_module = false);

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_COMPILE_MODULE_TO_LLVM_IR_H_
