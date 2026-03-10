/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/service/npu_compile_module_to_llvm_ir.h"

#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
// #include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/dump.h"
// #include "xla/service/gpu/alias_info.h"
// #include "xla/service/gpu/execution_stream_assignment.h"
// #include "xla/service/gpu/gpu_constants.h"
// #include "xla/service/gpu/gpu_executable.h"
// #include "xla/service/gpu/gpu_memory_space_assignment.h"
// #include "xla/service/gpu/ir_emitter_context.h"
// #include "xla/service/gpu/ir_emitter_unnested.h"
// #include "xla/service/gpu/metrics.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
// #include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
// #include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "tsl/platform/path.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/npu_alias_info.h"
#include "xla/xla_npu/service/npu_executable.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"
#include "xla/xla_npu/service/npu_ir_emitter_unnested.h"

namespace xla {

namespace {

using tsl::profiler::ScopedAnnotation;

// Prints mlir diagnostic messages to VLOG level 2.
static mlir::LogicalResult DiagnosticHandler(mlir::Diagnostic& diag) {
    VLOG(2) << diag.str();
    return mlir::failure();
}

// // Removes all globals from the given module that are both uninitialized and
// // have no uses within that module.
// void RemoveUnusedAndUninitializedGlobals(se::Platform::Id platform_id, const DebugOptions& options,
//                                          llvm::Module* llvm_module,
//                                          const std::vector<NpuExecutable::ConstantInfo>& constants) {
//     bool supports_runtime_managed_constants =
//         platform_id != se::rocm::kROCmPlatformId && options.xla_gpu_enable_shared_constants();
//     if (!supports_runtime_managed_constants) {
//         return;
//     }

//     for (const auto& info : constants) {
//         // Empty content means the constant is initialized in the LLVM IR, so we
//         // must not remove it.
//         if (!info.content.span().empty()) {
//             llvm::GlobalVariable* global = llvm_module->getGlobalVariable(info.symbol_name);
//             CHECK(global != nullptr);
//             if (global->use_empty()) {
//                 global->eraseFromParent();
//             }
//         }
//     }
// }

NpuCompileModuleResults InitializeResults(const HloModule* hlo_module, llvm::LLVMContext* llvm_context,
                                          const std::string& target_triple, const std::string& data_layout,
                                          const bool split_constants_module) {
    absl::string_view module_name = hlo_module->name();
    NpuCompileModuleResults results;
    results.module_name = module_name;
    results.llvm_module = std::make_unique<llvm::Module>(module_name, *llvm_context);
    results.llvm_module->setTargetTriple(llvm::Triple(target_triple));
    results.llvm_module->setDataLayout(data_layout);

    if (split_constants_module) {
        // Constants are emitted into a separate module to avoid caching them.
        results.llvm_module_constants =
            std::make_unique<llvm::Module>(absl::StrCat(module_name, "_consts"), *llvm_context);
        results.llvm_module_constants->setTargetTriple(llvm::Triple(target_triple));
        results.llvm_module_constants->setDataLayout(data_layout);
    }

    results.use_original_allocations = true;
    results.execution_stream_assignment = std::make_unique<NpuExecutionStreamAssignment>(hlo_module);
    return results;
}

// std::string GetDumpName(const se::DeviceDescription& device_desc) {
//     std::string prefix;
//     if (auto* cc = device_desc.gpu_compute_capability().cuda_compute_capability()) {
//         prefix = absl::StrCat("sm_", cc->ToString());
//     } else if (auto* cc = device_desc.gpu_compute_capability().rocm_compute_capability()) {
//         prefix = cc->gfx_version();
//     }
//     return absl::StrCat(prefix, "_gpu_", kAfterOptimizationsDumpName);
// }

std::unique_ptr<mlir::MLIRContext> CreateMlirContext() {
    mlir::DialectRegistry registry;
    // Disable MLIR multi-threading to prevent creating too many threads when
    // compiling XLA executables concurrently (e.g. during auto-tuning).
    auto mlir_context = std::make_unique<mlir::MLIRContext>(registry, mlir::MLIRContext::Threading::DISABLED);
    mlir_context->getDiagEngine().registerHandler(DiagnosticHandler);
    return mlir_context;
}

std::string Phase(absl::string_view phase_name, const HloModule* module) {
    return absl::StrFormat("%s:#module=%s,program_id=%d#", phase_name, module->name(), module->unique_id());
}

bool UseCache(const DebugOptions& options, bool split_constants_module) {
    return split_constants_module && options.xla_gpu_enable_llvm_module_compilation_parallelism() &&
           !options.xla_gpu_kernel_cache_file().empty();
}

absl::StatusOr<std::unique_ptr<NpuSequentialThunk>> LowerHlo(const HloModule* hlo_module,
                                                             NpuIrEmitterContext& ir_emitter_context,
                                                             llvm::Module* llvm_module_constants,
                                                             se::Platform::Id platform_id, bool use_cache) {
    const DebugOptions& options = hlo_module->config().debug_options();
    ScopedAnnotation annotation(Phase("XlaEmitLlvmIr", hlo_module));
    uint64_t start_usecs = tsl::Env::Default()->NowMicros();

    // if (use_cache) {
    //     TF_RETURN_IF_ERROR(NpuLoadCache(ir_emitter_context, options.xla_gpu_kernel_cache_file()));
    // }
    std::unique_ptr<NpuIrEmitterUnnested> ir_emitter = NpuIrEmitterUnnested::Create(&ir_emitter_context);
    {
        XLA_SCOPED_LOGGING_TIMER(absl::StrCat("NpuCompiler::RunBackend - IR emission for ", hlo_module->name()));

        TF_RETURN_IF_ERROR(ir_emitter->EmitHloComputation(hlo_module->entry_computation()));

        // RemoveUnusedAndUninitializedGlobals(platform_id, options, ir_emitter_context.llvm_module_constants(),
        //                                     ir_emitter_context.constants());

        // // This won't record values for calls that error out (because if they error
        // // out we have no way of telling how far through the process we got).
        // uint64_t end_usecs = tsl::Env::Default()->NowMicros();
        // RecordHloToLlvmDuration(end_usecs - start_usecs);
    }
    return ir_emitter->ConsumeThunkSequence();
}

}  // namespace

// absl::Status NpuLoadCache(NpuIrEmitterContext& ir_emitter_context, absl::string_view cache_file_path) {
//     tsl::profiler::TraceMe traceme("NpuLoadCache");
//     CHECK(!cache_file_path.empty());
//     std::string resolved_path;
//     if (!tsl::io::ResolveTestPrefixes(cache_file_path, resolved_path)) {
//         return FailedPrecondition("File path can not be resolved: %s", cache_file_path);
//     }
//     if (tsl::Env::Default()->FileExists(resolved_path).ok()) {
//         std::string serialized;
//         TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), resolved_path, &serialized));
//         CompilationCacheProto proto;
//         if (!proto.ParseFromString(serialized)) {
//             return Internal("Failed to parse serialized CompilationCacheProto.");
//         }
//         // Register all cached kernel names with the name uniquer to avoid
//         // naming conflicts.
//         for (const auto& [name, _] : proto.entries()) {
//             TF_RET_CHECK(ir_emitter_context.name_uniquer()->GetUniqueName(name) == name)
//                 << "Failed registering " << name << "in NameUniquer.";
//         }
//         TF_RETURN_IF_ERROR(ir_emitter_context.kernel_cache().Load(proto));
//     } else {
//         VLOG(1) << "Compilation cache file does not exist: " << resolved_path;
//     }
//     return absl::OkStatus();
// }

absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
    const HloModule* module, const NpuAliasInfo* alias_info, BufferValue::SizeFunction buffer_size_bytes_function) {
    // ScopedAnnotation annotation(Phase("XlaBufferAssignment", module));

    // const DebugOptions& options = module->config().debug_options();

    // std::optional<BufferValue::Color> color =
    //     options.xla_gpu_temp_buffer_use_separate_color()
    //         ? std::optional<BufferValue::Color>((int)MemorySpaceColor::kTempBuffer)
    //         : std::nullopt;
    // std::optional<BufferValue::Color> color = std::nullopt;

    if (!module->has_schedule()) {
        const std::string err_msg = "Module need to be scheduled.";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::UnavailableError(err_msg);
    }

    TF_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> buffer_assignment,
                        BufferAssigner::Run(
                            module, std::make_unique<SequentialHloOrdering>(module->schedule()),
                            std::move(buffer_size_bytes_function), alias_info,
                            /*color_alignment=*/
                            [](LogicalBuffer::Color) { return 256; },
                            /*allocate_buffers_for_constants=*/true
                            // /*colorer=*/CreateColorer(options),
                            // /*must_not_live_out=*/{},
                            // /*preset_assignments*/ {},
                            // /*private_stack*/ {},
                            // /*heap_buffer_interval_compare*/ nullptr,
                            // /*isolation_options*/ std::nullopt,
                            // color
                            ));

    VLOG(1) << "Buffer Assignment Stats for " << module->name() << "\n" << buffer_assignment->StatsString(alias_info);
    return buffer_assignment;
}

absl::StatusOr<NpuCompileModuleResults> NpuCompileModuleToLlvmIr(
    const HloModule* hlo_module, llvm::LLVMContext* llvm_context, const std::string& target_triple,
    const std::string& data_layout, const se::Platform* platform, const se::DeviceDescription& device_desc,
    const NpuAliasInfo* alias_info, BufferValue::SizeFunction buffer_size_bytes_function, bool split_constants_module) {
    tsl::profiler::TraceMe traceme("NpuCompileModuleToLlvmIr");
    const bool use_cache = UseCache(hlo_module->config().debug_options(), split_constants_module);

    NpuCompileModuleResults results =
        InitializeResults(hlo_module, llvm_context, target_triple, data_layout, split_constants_module);

    TF_ASSIGN_OR_RETURN(results.buffer_assignment,
                        RunBufferAssignment(hlo_module, alias_info, std::move(buffer_size_bytes_function)));
    TF_ASSIGN_OR_RETURN(results.output_info, NpuGetOutputInfo(*hlo_module, *results.buffer_assignment));

    // capture the output shape after buffer assignment because it may change
    // during buffer assignment (nevertheless the const hlo_module)
    results.output_shape = hlo_module->result_shape();
    // DumpHloModuleIfEnabled(*hlo_module, *results.buffer_assignment, GetDumpName(device_desc));

    VLOG(1) << "After optimization module fingerprint for " << hlo_module->name() << ": "
            << hlo_module->GetFingerprint128();

    const auto& output_sub_dir_name = NpuTool::generate_unique_folder_name_absl();
    const auto& output_dir = tsl::io::JoinPath(
        NpuConfig::GetInstance().GetTypedValue<std::string>(NpuConfig::Key::NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR),
        output_sub_dir_name);
    std::unique_ptr<mlir::MLIRContext> mlir_context = CreateMlirContext();
    auto symbolic_expr_context = std::make_unique<SymbolicExprContext>(mlir_context.get());
    NpuIrEmitterContext ir_emitter_context(output_dir, hlo_module, results.buffer_assignment.get(),
                                           results.execution_stream_assignment.get(), platform->Name(), device_desc,
                                           mlir_context.get(), results.llvm_module.get(),
                                           results.llvm_module_constants.get(),
                                           /*emit_kernels=*/true);

    TF_ASSIGN_OR_RETURN(results.executable, LowerHlo(hlo_module, ir_emitter_context,
                                                     results.llvm_module_constants.get(), platform->id(), use_cache));

    results.constants = std::move(ir_emitter_context.constants());
    // if (use_cache) {
    //     results.kernel_compilation_cache = ir_emitter_context.kernel_cache().Export();
    // }

    return results;
}

}  // namespace xla
