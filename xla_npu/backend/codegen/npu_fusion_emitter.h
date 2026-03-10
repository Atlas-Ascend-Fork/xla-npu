/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_BACKEND_CODEGEN_NPU_FUSION_EMITTER_H_
#define XLA_XLA_NPU_BACKEND_CODEGEN_NPU_FUSION_EMITTER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// #include "absl/status/status.h"
// #include "absl/status/statusor.h"
// #include "absl/types/span.h"
// #include "llvm/IR/Function.h"
// #include "llvm/IR/IRBuilder.h"
// #include "mlir/IR/AffineMap.h"
// #include "xla/backends/gpu/runtime/thunk.h"
// #include "xla/codegen/emitters/kernel_arguments.h"
// #include "xla/hlo/analysis/indexing_analysis.h"
// #include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
// #include "xla/service/gpu/ir_emitter_context.h"
// #include "xla/service/gpu/launch_dimensions.h"
// #include "xla/service/llvm_ir/ir_array.h"
// #include "xla/shape.h"
// #include "xla/stream_executor/device_description.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

namespace xla {

struct NpuFusionEmissionResult {
    std::vector<std::unique_ptr<NpuThunk>> thunks;
};

class NpuFusionInterface {
   public:
    virtual ~NpuFusionInterface() = default;

    virtual absl::StatusOr<NpuFusionEmissionResult> Emit(NpuIrEmitterContext& ir_emitter_context,
                                                         const HloFusionInstruction& fusion) const = 0;
};

// // Interface for fusions that are implemented using cuda kernels.
// class NpuKernelFusionInterface : public NpuFusionInterface {
//    public:
//     virtual ~NpuKernelFusionInterface() = default;

//     // Returns the fusion's launch dimensions.
//     virtual LaunchDimensions launch_dimensions() const = 0;

//     // Computes an indexing map from thread to output element(s) of the **hero**.
//     //
//     // The dimensions in the resulting map are
//     //   d0, d1, d2: threadIdx.{x,y,z}
//     //   d3, d4, d5: blockIdx.{x,y,z}
//     // If one thread computes multiple elements, this will be represented using a
//     // symbol.
//     //
//     // Cases where the exact element cannot be statically determined are currently
//     // unsupported (scatter, in-place DUS). Implementations will return nullopt.
//     // Note: Work in progress, not implemented for all emitters.
//     virtual std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(int64_t root_index,
//                                                                        SymbolicExprContext* ctx) const = 0;

//     // Computes indexing maps from thread id to input elements of the root's
//     // **hero**. Note that in many cases this is not computable from the output
//     // indexing. The indexing may only be known for some operands of the hero.
//     virtual std::optional<std::vector<IndexingMap>> ComputeThreadIdToInputIndexing(int64_t root_index,
//                                                                                    SymbolicExprContext* ctx) const =
//                                                                                    0;

//     static constexpr std::array<int, 3> kIndexingMapThreadIdxDims = {0, 1, 2};
//     static constexpr std::array<int, 3> kIndexingMapBlockIdxDims = {3, 4, 5};

//    protected:
//     // Returns the default mapping for the given launch dimensions: linearizes
//     // the thread index and then reshapes it into the given layout.
//     // Populates the ranges for d0, d1, d2, d3, d4, d5 from the thread counts and
//     // block sizes in the given launch dimensions.
//     static IndexingMap GetDefaultThreadIdIndexingMap(const LaunchDimensions& launch_dims, int unroll_factor,
//                                                      const Shape& shape, SymbolicExprContext* ctx);
// };

// absl::StatusOr<llvm::Function*> NpuBuildKernelPrototype(NpuIrEmitterContext& ir_emitter_context,
//                                                         const std::string& impl_fn_name,
//                                                         const std::string& suggested_name,
//                                                         const emitters::KernelArguments& arguments,
//                                                         const LaunchDimensions& launch_dimensions,
//                                                         llvm::IRBuilderBase* builder);

// absl::StatusOr<llvm::Function*> NpuBuildKernelPrototypeFromUniqueName(NpuIrEmitterContext& ir_emitter_context,
//                                                                       const std::string& impl_fn_name,
//                                                                       const std::string& unique_kernel_name,
//                                                                       const emitters::KernelArguments& arguments,
//                                                                       const LaunchDimensions& launch_dimensions,
//                                                                       llvm::IRBuilderBase* builder);

// // Compute the kernel name. The opcode string may contain "-" which cannot be
// // in a PTX function name, so sanitize the name before uniquifying it.
// std::string NpuGetSanitizedUniqueName(NpuIrEmitterContext& ir_emitter_context, const std::string& suggested_name);

// absl::Status NpuAnnotateKernelLaunchDimensions(const se::DeviceDescription& device_info,
//                                                const LaunchDimensions& launch_dims, llvm::Function* kernel,
//                                                llvm::Module* llvm_module);

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_CODEGEN_NPU_FUSION_EMITTER_H_
