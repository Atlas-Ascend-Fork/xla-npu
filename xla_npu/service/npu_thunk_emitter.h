#ifndef SERVICE_NPU_THUNK_EMITTER_H_
#define SERVICE_NPU_THUNK_EMITTER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/sort_thunk.h"
// #include "xla/backends/cpu/runtime/thunk.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/ir_emitter2.h"
#include "xla/service/cpu/parallel_fusion_emitter.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

#include "xla/xla_npu/backend/npu_thunk.h"

namespace xla {

// NpuThunkEmitter is responsible for converting optimized HLO module into a
// sequence of thunks that will launch "work" on the CPU: launch host kernels,
// call into the libraries (oneDNN, Eigen, etc.).
//
// During the thunk emission it emits IR (LLVM IR) for the host kernels via the
// IrEmitter that later will be compiled into the executable binary (one or
// multiple LLVM modules compiled to object files).
class NpuThunkEmitter {
 public:
  struct Options {
    // Whether to compile copy as LLVM kernel. This is used to avoid
    // dependencies on pjrt/transpose for tfcompiled models.
    bool compile_copy_as_llvm_kernel;
    // Wheter the thunk emitter is used for AOT compilation. AOT compiled
    // kernels get linked together and might have to respect certain
    // restrictions, such as having the same module flags.
    bool is_aot_compilation;
  };

  struct EmittedKernel {
    EmittedKernel(absl::string_view name, llvm::orc::ThreadSafeModule module)
        : kernel_name(name), module(std::move(module)) {}

    std::string kernel_name;
    llvm::orc::ThreadSafeModule module;
  };

  NpuThunkEmitter(cpu::IrEmitter2& ir_emitter, tsl::thread::ThreadPool& thread_pool,
               const BufferAssignment& buffer_assignment,
               const cpu::TargetMachineFeatures& target_machine_features,
               const HloModule& hlo_module,
               const Options& options = {/*compile_copy_as_llvm_kernel=*/false,
                                         /*is_aot_compilation=*/false});

  // Emits HLO module entry computation as a sequence of thunks.
  absl::StatusOr<NpuThunkSequence> EmitEntryComputation(const HloModule& module);

  absl::StatusOr<std::vector<EmittedKernel>> ConsumeKernels();

 private:
  struct HostKernelAllocationSlices {
    std::vector<BufferAllocation::Slice> arguments;
    std::vector<BufferAllocation::Slice> results;
  };

  std::optional<cpu::SortThunk::SortDirection> MatchSortDirection(
      const HloComputation* hlo_comparator) const;

  // Returns the buffer allocation slice assigned to the given instruction at
  // the given shape index. Instruction must have a unique slice assigned to it!
  absl::StatusOr<BufferAllocation::Slice> GetAllocationSlice(
      const HloInstruction* instruction, const ShapeIndex& index = {});

  // Returns a token resource corresponding to the given instruction result.
  absl::StatusOr<std::shared_ptr<Resource>> GetTokenResource(
      const HloInstruction* instruction, const ShapeIndex& index = {});

  absl::StatusOr<NpuThunkSequence> EmitHloComputation(
      const HloComputation* computation);

  absl::StatusOr<NpuThunkSequence> EmitHloInstruction(
      const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitCallThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitConcatenateKernelThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitGetDimensionSizeThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitSetDimensionSizeThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitBatchNormGradThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitBatchNormTrainingThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitConvolutionThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitCopyThunk(
//       const HloInstruction* instruction);

  absl::StatusOr<NpuThunkSequence> EmitElementalKernelThunk(
      const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitPadKernelThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitFftThunk(const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitFusionKernelThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitReductionKernelThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitRngThunk(const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitRngBitGeneratorThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitRngGetAndUpdateStateThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitStochasticConvertThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitInfeedThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitOutfeedThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitConditionThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitWhileThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitDotThunk(const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitReplicaIdThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitPartitionIdThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitAllGatherThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitAllReduceThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitAllToAllThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitCollectivePermuteThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitReduceScatterThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitCustomCallThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitSliceToDynamicThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitTopKThunk(
//       const HloCustomCallInstruction* custom_call);

//   absl::StatusOr<NpuThunkSequence> EmitOneDnnOpThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitSliceThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitDynamicUpdateSliceThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitSortThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitXnnFusionThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitYnnFusionThunk(
//       const HloInstruction* instruction);

//   absl::StatusOr<NpuThunkSequence> EmitOneDnnFusionThunk(
//       const HloInstruction* instruction);

  // Returns the list of buffer allocation slices assigned to the given
  // instruction that will be passed to the host kernel as arguments: a
  // flattened list of all the leaf buffers for all operands and result. We do
  // not materialize tuples at run time and only read and write from buffers
  // corresponding to arrays.
  absl::StatusOr<HostKernelAllocationSlices> GetHostKernelAllocationSlices(
      const HloInstruction* instruction);

  // Verifies that the element types of all of the given operand instructions
  // match and are of one of the given supported types.
  absl::Status ElementTypesSameAndSupported(
      const HloInstruction& instruction,
      absl::Span<const HloInstruction* const> operands,
      absl::Span<const PrimitiveType> supported_types);

  // Convenience function that creates a thunk sequence containing given kernel.
  static absl::StatusOr<NpuThunkSequence> MakeKernelThunkSequence(
      const HloInstruction* instruction,
      const NpuThunkEmitter::HostKernelAllocationSlices& buffers,
      const cpu::IrEmitter2::KernelInfo& kernel,
      std::optional<uint64_t> min_alignment = std::nullopt);

  static absl::StatusOr<NpuThunkSequence> MakeKernelThunkSequence(
      const HloInstruction* instruction, const KernelSpec& kernel_spec,
      std::optional<uint64_t> min_alignment = std::nullopt);

  cpu::IrEmitter2& ir_emitter_;
  const BufferAssignment& buffer_assignment_;

  const cpu::TargetMachineFeatures& target_machine_features_;
  const HloModuleConfig& hlo_module_config_;

  const Options options_;

  // A global resource that is used to order all collective operations.
  std::shared_ptr<Resource> communicator_resource_;

  // Token resources that correspond to the token buffer allocation slices. We
  // rely on buffer assignment to assign unique "identity" to each token, and
  // create a separate resource for each unique allocation slice.
  absl::flat_hash_map<BufferAllocation::Slice, std::shared_ptr<Resource>>
      token_resources_;

  std::vector<EmittedKernel> kernels_;

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
  SymbolicExprContext symbolic_expr_context_{mlir_context_.get()};
  cpu::FusionCompiler fusion_compiler_;

  absl::flat_hash_map<std::string, KernelSpec> kernel_spec_cache_;

  cpu::ParallelFusionEmitter parallel_fusion_emitter_;
};

}  // namespace xla

#endif  // SERVICE_NPU_THUNK_EMITTER_H_
