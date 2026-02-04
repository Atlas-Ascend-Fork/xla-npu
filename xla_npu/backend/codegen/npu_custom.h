#ifndef BACKEND_NPU_CUSTOM_H_
#define BACKEND_NPU_CUSTOM_H_

#include "absl/status/statusor.h"
// #include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/hlo/ir/hlo_instructions.h"
// #include "xla/service/gpu/hlo_fusion_analysis.h"
// #include "xla/service/gpu/ir_emitter_context.h"
#include "xla/xla_npu/backend/codegen/npu_fusion_emitter.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

namespace xla {

// A wrapper for fusions implemented using the mechanism in
// xla/service/gpu/kernels. See custom_kernel_fusion.h in that folder for
// details.
class NpuCustomFusion : public NpuFusionInterface {
   public:
    absl::StatusOr<NpuFusionEmissionResult> Emit(NpuIrEmitterContext& ir_emitter_context,
                                                 const HloFusionInstruction& fusion) const final;
};

// // Emitter for custom fusions implementing address computation. An address
// // computation contains a custom call hero, with at least one of its operands
// // coming from a static contiguous slice. E.g. operand `%cast` of `%gemm` coming
// // from `%slice`:
// // %address_computation {
// //   %p0 = f32[2, 1024, 1024]
// //   %p1 = f32[1024, 1024]
// //   %slice = f32[1, 1024, 1024] slice(%p0)
// //   %cast = f32[1024, 1024] bitcast(%slice)
// //   ROOT %gemm = custom_call(%cast, %p1) __cublas$Gemm
// // }
// //
// // The goal is to compute the buffer addresses for such operands (`%cast`) at
// // compile-time instead of allocating a new buffer for it at runtime by
// // translating the static slice into offset + size of the original buffer passed
// // into the custom call `%gemm`.
// //
// // It is possible to inscribe the results of the custom call within a larger
// // array. In that case, the affected results are each fed into a
// // `dynamic-update-slice` operation, whose result is one of the fusion's
// // outputs.
// //
// // The pass makes the assumption that, for each one of the custom-call's outputs
// // there is exactly one path to the fusion root. The resulting shape for the
// // dynamic slice fusion may be an unwrapped array, a flat tuple, or even a
// // nested tuple.
// class NpuDynamicSliceFusion : public NpuFusionInterface {
//    public:
//     explicit NpuDynamicSliceFusion(const HloFusionAnalysis& analysis, const CallGraph& call_graph)
//         : analysis_(analysis), call_graph_(call_graph) {}

//     absl::StatusOr<NpuFusionEmissionResult> Emit(NpuIrEmitterContext& ir_emitter_context,
//                                               const HloFusionInstruction& fusion) const final;

//    private:
//     const HloFusionAnalysis& analysis_;
//     const CallGraph& call_graph_;
// };

}  // namespace xla

#endif  // BACKEND_NPU_CUSTOM_H_
