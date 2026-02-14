#ifndef SERVICE_NPU_COMPILER_H_
#define SERVICE_NPU_COMPILER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
// #include "xla/service/cpu/cpu_aot_compilation_result.h"
// #include "xla/service/cpu/executable.pb.h"
// #include "xla/service/cpu/thunk_emitter.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/service/llvm_compiler.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla_npu/service/cpu_compiler_type.npu_executable.h"
#include "xla/xla_npu/service/cpu_compiler_type.npu_thunk_emitter.h"

namespace mlir {
class DialectRegistry;
}  // namespace mlir

namespace xla {

// NPU-targeting implementation of the XLA Compiler interface.
//
// The compiler translates XLA HLO code into LLVM IR and uses LLVM's JIT
// infrastructure to create an executable "blob" that can then be returned
// wrapped in CpuExecutable and actually invoked.
class NpuCompiler : public LLVMCompiler {
   public:
    NpuCompiler();
    ~NpuCompiler() override = default;

    absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(std::unique_ptr<HloModule> hlo_module,
                                                                     std::vector<se::StreamExecutor*> stream_execs,
                                                                     const CompileOptions& options) override;

    absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(std::unique_ptr<HloModule> module,
                                                            se::StreamExecutor* stream_exec,
                                                            const CompileOptions& options) override;

    absl::StatusOr<std::unique_ptr<Executable>> RunBackend(std::unique_ptr<HloModule> module,
                                                           se::StreamExecutor* stream_exec,
                                                           const CompileOptions& options) override;

    absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>> CompileAheadOfTime(
        std::unique_ptr<HloModule> hlo_module, const AotCompilationOptions& options) override;

    se::Platform::Id PlatformId() const override;

    HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

    //   absl::StatusOr<std::unique_ptr<AotCompilationResult>> Export(
    //       Executable* executable) const override;

    //   // Returns a (deserialized) AotCompilationResult from a serialized
    //   // AotCompilationResult.
    //   absl::StatusOr<std::unique_ptr<AotCompilationResult>>
    //   LoadAotCompilationResult(const std::string& serialized_aot_result) override;

    absl::StatusOr<HloSchedule> CreateHloSchedule(const HloModule& hlo_module) const;

    absl::StatusOr<std::unique_ptr<BufferAssignment>> CreateBufferAssignment(const HloModule& module) const;

   private:
    // Initialize the LLVM target.
    static void InitializeLLVMTarget();

    // Runs the HLO passes which are necessary for both optimizations and
    // correctness.
    absl::Status RunHloPasses(HloModule* module, bool is_aot_compile, llvm::TargetMachine* target_machine,
                              const CompileOptions& compile_options);

    //   // Runs HLO passes up to and including layout assignment.
    //   absl::Status RunHloPassesThroughLayoutAssn(
    //       HloModule* module, bool /*is_aot_compile*/,
    //       TargetMachineFeatures* target_machine_features);

    // Runs HLO passes after layout assignment.
    absl::Status RunHloPassesAfterLayoutAssn(HloModule* module, bool is_aot_compile,
                                             cpu::TargetMachineFeatures* target_machine_features,
                                             const CompileOptions& compile_options);

    absl::StatusOr<std::unique_ptr<NpuExecutable>> CompileNpuExecutable(
        std::unique_ptr<HloModule> module, const NpuThunkEmitter::Options& thunk_emitter_options,
        std::unique_ptr<cpu::IrCompiler> ir_compiler, const llvm::PICLevel::Level& pic_level = llvm::PICLevel::NotPIC,
        const llvm::PIELevel::Level& pie_level = llvm::PIELevel::Default);

    // absl::StatusOr<std::unique_ptr<AotCompilationResult>> CompileAheadOfTimeThunks(
    //     std::unique_ptr<HloModule> module, cpu::IrCompiler::TargetMachineBuilder target_machine_builder,
    //     const cpu::CpuAotCompilationOptions& aot_options, const llvm::Triple& triple,
    //     const llvm::PICLevel::Level& pic_level, const llvm::PIELevel::Level& pie_level);

    NpuCompiler(const NpuCompiler&) = delete;
    NpuCompiler& operator=(const NpuCompiler&) = delete;
};

}  // namespace xla

#endif  // SERVICE_NPU_COMPILER_H_
