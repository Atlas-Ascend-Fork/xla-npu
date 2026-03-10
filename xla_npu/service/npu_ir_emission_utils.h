/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_NPU_IR_EMISSION_UTILS_H_
#define XLA_XLA_NPU_SERVICE_NPU_IR_EMISSION_UTILS_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
// #include "xla/service/gpu/ir_emission_utils.pb.h"
#include "tsl/platform/protobuf.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

// // <HLO computation fingerprint, serialized compiled object>.
// using BinaryMap = absl::flat_hash_map<std::string, std::string>;

// If a dimensions is smaller than this, untiled transposition may be more
// efficient.
inline constexpr int64_t kNpuMinDimensionToTransposeTiled = 16;
// But if both swap dimensions are larger than 'kNpuMinDimensionToTransposeTiled2',
// and the product of the dimensions to be swapped is larger than
// 'kNpuMinTotalDimensionsToTransposeTiled', tiled transposition may be more
// efficient.
inline constexpr int64_t kNpuMinDimensionToTransposeTiled2 = 8;
inline constexpr int64_t kNpuMinTotalDimensionsToTransposeTiled = 64 * 128;
// As the amount of shared memory is limited, we need to make sure that we don't
// detect 102 transposes that would require too much bytes for the most minor
// dimension.
inline constexpr int64_t kNpuMaxBitsInMostMinorDimension = 8 * 8;

// // Returns true if the given dot is supported by cuBLAS.
// absl::StatusOr<bool> IsCublasSupportedMatMul(
//     const HloInstruction& dot, bool allow_matrix_vector_multiplication);

// inline constexpr int64_t WarpSize(
//     const se::DeviceDescription& gpu_device_info) {
//   return gpu_device_info.threads_per_warp();
// }

// // Fusions that implemented with pre-compiled device kernels have
// // FusionBackendConfig.kind requel to this string.
// inline constexpr absl::string_view kCustomFusionKind = "__custom_fusion";

// // Generic fusions that use Triton have FusionBackendConfig.kind equal to this
// // string. This fusion kind will eventually subsume all usages of
// // kTritonGemmFusionKind.
// inline constexpr absl::string_view kTritonFusionKind = "__triton";

// // Used for fusions that codegen a collective.
// inline constexpr absl::string_view kTritonCollectiveFusionKind =
//     "__triton_collective";

// // Fusions that use Triton have FusionBackendConfig.kind equal to this string.
// inline constexpr absl::string_view kTritonGemmFusionKind = "__triton_gemm";

// // Generic fusions that use Triton have FusionBackendConfig.kind equal to this
// // string. Used for fusions that implement a dot expressed as nested fusions.
// inline constexpr absl::string_view kTritonNestedGemmFusionKind =
//     "__triton_nested_gemm_fusion";

// // Fusions that use Triton have FusionBackendConfig.kind equal to this string.
// // Used for fusions that implement a scaled dot.
// inline constexpr absl::string_view kTritonScaledDotFusionKind =
//     "__triton_scaled_dot_fusion";

// inline constexpr absl::string_view kCuDnnFusionKind = "__cudnn$fusion";

// // Fusions that can be emitted using a dynamic memcpy. A dynamic memcpy depends
// // on some loop induction variable.
// inline constexpr absl::string_view kDynamicMemcpyFusionKind =
//     "__dynamic_memcpy";

// inline constexpr absl::string_view kUncompilableFusion =
//     "__uncompilable_fusion";

// inline constexpr absl::string_view kTopKCustomCallTarget = "__gpu$TopK";

// The number of shared memory banks.
inline constexpr int64_t kNpuNumShmemBanks = 32;

// The bitwidth of a shared memory bank.
inline constexpr int64_t kNpuBankBitwidth = 32;

// // The name of the custom fusion config for dynamic slice fusion with static
// // slices, such that the offset can be computed at compile time.
// inline constexpr absl::string_view
//     kDynamicSliceFusionWithStaticAddressComputationConfigName =
//         "address_computation";
// // The name of the custom fusion config for dynamic slice fusion with dynamic
// // slices, such that the offset is computed at runtime.
// inline constexpr absl::string_view
//     kDynamicSliceFusionWithDynamicAddressComputationConfigName =
//         "dynamic_address_computation";

// // Returns the name of the custom fusion config if the given instruction is a
// // custom fusion and has a custom fusion name, otherwise returns std::nullopt.
// // The custom fusion name is basically the value of
// // instr.backend_config().fusion_backend_config().custom_fusion_config().name().
// // If any of this does not exist in the chain, then we return std::nullopt.
// std::optional<std::string> GetCustomFusionConfigName(
//     const HloInstruction* instr);

// // Returns true if the given instruction is a custom fusion for dynamic slice
// // fusion. This is determined by checking the name of custom fusion config.
// bool IsDynamicSliceFusion(const HloInstruction* instr);

// // Returns true if `hlo` will be implemented as a call to a TopK routine.
// bool IsCustomCallToTopK(const HloInstruction& hlo);

// // Returns true if `hlo` will be implmented as a call to a custom PTX kernel
// // implementation.
// bool IsCustomCallToPtxKernel(const HloInstruction& hlo);

// // Returns true if instruction is a Mosaic GPU collective instruction.
// bool IsCollectiveMosaicGpuInstruction(const HloInstruction& hlo);

// // Returns true if `instr` is a slice (or dynamic slice) instruction and
// // operates on a contiguous slice of the input buffer.
// bool IsContiguousSlice(const HloInstruction& instr);

// // Emits code that determines whether the current thread is thread 0 within
// // block 0 of the kernel.
// llvm::Value* IsBlock0Thread0(llvm::IRBuilderBase* b);

absl::StatusOr<BufferAllocation::Slice> NpuGetAllocationSlice(const BufferAssignment& buffer_assignment,
                                                              const HloInstruction* instr, const ShapeIndex& index);

// Returns the first hero instruction reachable from `instr` as root. Hero
// instruction can be in a different computation if the parent HloFusionAdaptor
// is a producer-consumer fusion.
HloInstructionAdaptor NpuFindNonTrivialHero(const HloInstructionAdaptor& instr);

// Same as above, but fusion is the parent computation of the hlo instruction.
const HloInstruction& NpuFindNonTrivialHero(const HloInstruction& instr);

std::optional<TransposeDescription> NpuGetDescriptionForTiledTransposeEmitter(const HloInstruction& hero);

// Canonical transpose permutes the input shape
// <D_0 x ... x D_n x T2 x D_{n+1} x ... x D_m x A x T1 x B> into
// <D'_0 x ... x D'_n' x T1 x D'_{n'+1} x ... x D'_m x A x T2 x B>.
// Note that the `D` dimensions are batch dimensions. They can also be
// permuted, but they are tiled by 1.
//
// Examples:
// 1. <8x32> -> <32x8> will be canonicalized to <8x1x32x1> -> <32x1x8x1>.
// 2. <8x2x32> -> <32x2x8> will be canonicalized to <8x2x32x1> -> <32x2x8x1>.
// 3. <8x2x32x7x6> -> <6x32x2x7x8> becomes <8x2x32x7x6x1> -> <6x32x2x7x8x1>.

// TODO(b/370690811): Unify this with TransposeDescription.
struct NpuTransposeSpec {
    PrimitiveType elem_type() const { return input_shape().element_type(); }

    const Shape& input_shape() const { return transpose->operand(0)->shape(); }
    const Shape& output_shape() const { return transpose->shape(); }

    int64_t rank() const { return input_shape().dimensions().size(); }
    int64_t canonical_rank() const { return canonical_input_shape.size(); }

    int64_t dim_A() const { return canonical_input_shape[dim_A_id()]; }
    int64_t dim_A_id() const { return canonical_rank() - 3; }

    int64_t dim_B() const { return canonical_input_shape.back(); }
    int64_t dim_B_id() const { return canonical_rank() - 1; }

    int64_t dim_T1() const { return canonical_input_shape[dim_T1_input_id()]; }
    int64_t dim_T1_input_id() const { return canonical_rank() - 2; }
    int64_t dim_T1_output_id() const { return canonical_inv_permutation[canonical_rank() - 2]; }

    int64_t dim_T2() const { return canonical_input_shape[dim_T2_input_id()]; }
    int64_t dim_T2_input_id() const { return canonical_permutation[canonical_rank() - 2]; }
    int64_t dim_T2_output_id() const { return canonical_rank() - 2; }

    std::string ToString() const;

    const HloTransposeInstruction* transpose;

    llvm::SmallVector<int64_t, 3> permutation;
    llvm::SmallVector<int64_t, 3> inv_permutation;

    llvm::SmallVector<int64_t, 3> canonical_output_shape;
    llvm::SmallVector<int64_t, 3> canonical_permutation;
    llvm::SmallVector<int64_t, 3> canonical_inv_permutation;
    llvm::SmallVector<int64_t, 3> canonical_input_shape;
};

NpuTransposeSpec NpuGetTransposeSpec(const HloTransposeInstruction* transpose);

// Returns the default tile sizes for the packed transpose emitter.
absl::StatusOr<absl::InlinedVector<int64_t, 3>> NpuGetPackedTransposeTileSizes(const NpuTransposeSpec& spec);

// // Verify the given module, and crash if it failed.
// void VerifyModule(const llvm::Module& module);

// // Returns the llvm type for the indices used in the kernel that contains the
// // hlo instruction. Such indices include the index for the parallel loop and
// // the indices for the tensors accessed by the kernel. The return type is i32
// // iff the following conditions are met:
// //  . The launch_size of the kernel is within the range of i32.
// //  . The sizes of all the tensors accessed within the kernel are within the
// //    range of i32.
// // Otherwise, the return type is i64.
// llvm::Type* GetIndexTypeForKernel(const HloInstruction* hlo,
//                                   int64_t launch_size, llvm::IRBuilderBase* b);

// This class stores either a non-owning reference or owns data that represents
// a dense array in XLA format. It is used for intermediate storage during IR
// constant emission.
class NpuDenseDataIntermediate {
   public:
    // Creates an instance of NpuDenseDataIntermediate that owns the provided vector.
    static NpuDenseDataIntermediate Own(std::vector<uint8_t> owned) {
        NpuDenseDataIntermediate di;
        di.data_ = std::move(owned);
        return di;
    }

    // Creates an instance of NpuDenseDataIntermediate that aliases the input.
    static NpuDenseDataIntermediate Alias(absl::Span<const uint8_t> aliased) {
        NpuDenseDataIntermediate di;
        di.data_ = aliased;
        return di;
    }

    // Returns a reference to the data this object represents.
    absl::Span<const uint8_t> span() const {
        return data_.index() == 0 ? absl::Span<const uint8_t>(std::get<0>(data_)) : std::get<1>(data_);
    }

    // // Converts `this` into its protobuf representation.
    // // Note that the protobuf message will always contain a copy of the data -
    // // also for non-owning instances of NpuDenseDataIntermediate.
    // DenseDataIntermediateProto ToProto() const;

    // // Constructs a data-owning instance of NpuDenseDataIntermediate from its
    // // protobuf representation.
    // static NpuDenseDataIntermediate FromProto(
    //     const DenseDataIntermediateProto& proto);

   private:
    std::variant<std::vector<uint8_t>, absl::Span<const uint8_t>> data_;
};

absl::StatusOr<NpuDenseDataIntermediate> NpuLiteralToXlaFormat(const Literal& literal);

// // Returns a deterministic encoded string representation of the proto message.
// absl::StatusOr<std::string> GetProtoFingerprint(
//     const tsl::protobuf::MessageLite&);

// // Returns concatenated fingerprint of an HLO instruction without its backend
// // config and its backend config's deterministic fingerprint.
// template <typename ConfigType>
// absl::StatusOr<std::string> FingerprintWithBackendConfig(
//     const HloInstruction& hlo) {
//   TF_ASSIGN_OR_RETURN(const auto config, hlo.backend_config<ConfigType>());
//   TF_ASSIGN_OR_RETURN(const std::string fingerprint,
//                       GetProtoFingerprint(config));
//   return absl::StrCat(hlo.ToString(HloPrintOptions::Fingerprint()),
//                       ", backend_config_fingerprint=", fingerprint);
// }

// struct InductionVariableFunctionalDependency {
//   // The dependency may be via multiple levels of intermediate calls. At each
//   // level, we need to know which parameters to evaluate, since not all of them
//   // may be relevant. The while loop's body is not included here, since the
//   // induction variable is implicitly the only dependency that is allowed.
//   // The size of the value is always the same as the number of parameters in the
//   // computation. We request a single element of inlined space, which will
//   // automatically pick the optimum value (16, usually).
//   absl::flat_hash_map<const HloComputation*,
//                       absl::InlinedVector<bool, 1 /* chosen automatically */>>
//       required_parameters;

//   // The loop and its induction variable that the value depends on.
//   const HloInstruction* loop;
//   const HloInstruction* induction_var;
// };

// // Checks if `instr`'s value is a pure function of a while loop's induction
// // variable. This supports instructions that are inside call, async or fusion
// // instructions. The dependency can be through arbitrary non-side-effecting
// // instructions.
// // Currently, this does not support nested while loops. Only dependencies on the
// // inner-most while loop will successfully be analyzed.
// // Requires `while_loop_trip_count_annotator` to have been run on the loop.
// std::optional<InductionVariableFunctionalDependency>
// ResolveFunctionalDependencyOnInductionVariable(const HloInstruction* instr);

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_IR_EMISSION_UTILS_H_
