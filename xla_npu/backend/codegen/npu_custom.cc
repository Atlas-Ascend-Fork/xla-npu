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

#include "xla/xla_npu/backend/codegen/npu_custom.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <mutex>
#include <unordered_map>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/ptrvec.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/call_inliner.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/tools/hlo_extractor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_npu/backend/codegen/npu_fusion_emitter.h"
#include "xla/xla_npu/backend/runtime/npu_custom_kernel_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/kernel/npu_custom_kernel.h"
#include "xla/xla_npu/service/kernel/npu_kernel_config.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

#if NPU_AUTO_FUSE_BACKEND == 1
#include "xla/xla_npu/service/convert_to_afir/ascir_backend.h"
#elif NPU_AUTO_FUSE_BACKEND == 2
#include "xla/xla_npu/service/convert_to_npuir/npuir_backend.h"
#endif

namespace xla {

namespace {

// 编译结果缓存结构 - 直接使用 computation 名称 (calls 后的名称) 作为 key
struct CompileCache {
  // 全局编译缓存 (线程安全)
  static std::unordered_map<std::string, NpuCompileResult>& GetCache() {
    static std::unordered_map<std::string, NpuCompileResult> cache;
    return cache;
  }

  // 缓存锁
  static std::mutex& GetMutex() {
    static std::mutex cache_mutex;
    return cache_mutex;
  }
};

}  // namespace

namespace {

absl::StatusOr<std::unique_ptr<NpuThunk>> BuildCustomKernelThunkForFusion(
    NpuIrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    NpuCustomKernel custom_kernel) {
  TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                      emitters::KernelArguments::Create(
                          ir_emitter_context.buffer_assignment(),
                          NpuConstant::GetDefaultBufferAlignment(), &fusion));

  NpuThunk::ThunkInfo thunk_info = NpuThunk::ThunkInfo::WithProfileAnnotation(
      &fusion, ir_emitter_context.GetNextThunkId());
  return std::make_unique<NpuCustomKernelThunk>(std::move(thunk_info),
                                                std::move(custom_kernel),
                                                std::move(kernel_arguments));
}

}  // namespace

absl::StatusOr<NpuFusionEmissionResult> NpuCustomFusion::Emit(
    NpuIrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  const HloComputation* fused_computation =
      fusion.fused_instructions_computation();

  NpuCompileResult npu_compile_result;
  const std::string entry_computation_key_name =
      NpuConstant::entryComputationKeyName;

#if NPU_AUTO_FUSE_BACKEND == 1
  // 当前autofuse处理的规则是非Entry开头的子computation分别做融合
  NPU_LOG_FORMAT_PRINT(
      INFO,
      "Npu auto fuse switch is set open, run fusion compile based on afir.");

  // 构建 computation map: key 为 "main" 表示 entry computation
  // 如果有 reduce 指令，将其 to_apply computation 也加入 map
  std::unordered_map<std::string, const HloComputation*> computation_map;
  computation_map[entry_computation_key_name] = fused_computation;

  // 遍历所有指令,查找 reduce 类型指令
  for (const HloInstruction* instr : fused_computation->instructions()) {
    if (instr->opcode() == HloOpcode::kReduce) {
      const HloComputation* reduce_computation = instr->to_apply();
      if (reduce_computation != nullptr) {
        std::string reduce_comp_name = std::string(reduce_computation->name());
        computation_map[reduce_comp_name] = reduce_computation;
        NPU_LOG_FORMAT_PRINT(
            INFO, "Found reduce instruction with to_apply computation: %s",
            reduce_comp_name.c_str());
      }
    }
  }

  // 生成缓存 key - 直接使用 fusion computation 的名称 (calls 后的名称)
  std::string cache_key = std::string(fused_computation->name());

  // 检查缓存
  bool cache_hit = false;
  {
    std::lock_guard<std::mutex> lock(CompileCache::GetMutex());
    auto cache_iter = CompileCache::GetCache().find(cache_key);
    if (cache_iter != CompileCache::GetCache().end()) {
      NPU_LOG_FORMAT_PRINT(INFO, "Cache hit for fusion computation: %s",
                           cache_key.c_str());
      npu_compile_result = cache_iter->second;
      cache_hit = true;
    }
  }

  // 如果缓存未命中，执行编译
  if (!cache_hit) {
    NPU_LOG_FORMAT_PRINT(INFO,
                         "Cache miss for fusion computation: %s, compiling...",
                         cache_key.c_str());

    AscirBackend ascir_backend;
    AscirBackendOptions ascir_backend_compile_option;
    ascir_backend_compile_option.output_dir = ir_emitter_context.GetOutputDir();
    ascir_backend_compile_option.soc_version = "Ascend910B1";
    TF_ASSIGN_OR_RETURN(
        npu_compile_result,
        ascir_backend.Compile(computation_map, ascir_backend_compile_option));

    // 更新缓存
    {
      std::lock_guard<std::mutex> lock(CompileCache::GetMutex());
      CompileCache::GetCache()[cache_key] = npu_compile_result;
      NPU_LOG_FORMAT_PRINT(
          INFO, "Cached compilation result for fusion computation: %s",
          cache_key.c_str());
    }
  }
#elif NPU_AUTO_FUSE_BACKEND == 2
  NPU_LOG_FORMAT_PRINT(
      INFO,
      "Npu auto fuse switch is set open, run fusion compile based on npuir.");
  mlir::hfusion::GraphGenTester::test();
#endif

  auto artifact_infos = npu_compile_result.artifact_infos;

  CHECK_COND_AND_PRINT_NPU_LOG(
      artifact_infos.count(entry_computation_key_name) == 0,
      "Can't find computation %s in compile result",
      entry_computation_key_name);

  auto npu_kernel_config_ptr = std::make_unique<NpuKernelConfig>();
  npu_kernel_config_ptr->npu_kernel_type =
      NpuKernelConfig::NpuKernelType::FUSION;
  npu_kernel_config_ptr->npu_artifact_info =
      artifact_infos[entry_computation_key_name];
  se::BlockDim block_dims(
      0);  // autofuse算子的block dim由tiling计算得到, 不使用这里设置的值
  size_t shared_memory_bytes = 0;
  std::string computation_name = std::string(
      fused_computation
          ->name());  // 此处kernel的名称为computation的名称,即calls后面跟的名称
  NpuCustomKernel kernel(std::move(npu_kernel_config_ptr), computation_name,
                         block_dims, shared_memory_bytes);
  TF_ASSIGN_OR_RETURN(
      auto thunk, BuildCustomKernelThunkForFusion(ir_emitter_context, fusion,
                                                  std::move(kernel)));

  NpuFusionEmissionResult result;
  result.thunks.push_back(std::move(thunk));
  return result;
}

}  // namespace xla
