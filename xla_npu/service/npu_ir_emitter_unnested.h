/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_UNNESTED_H_
#define XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_UNNESTED_H_

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/shape_util.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/service/npu_ir_emitter.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

namespace xla {

// Emits LLVM IR for an "unnested computation".
// An unnested computation is an HloComputation which you run by executing one
// or more kernels for each HloInstruction it contains.  Examples of unnested
// computations:
//  - An HloModule's root computation,
//  - The body of an HLO while loop,
//  - The true/false computation of an HLO conditional.
// Note the opportunity for confusion -- the while loop's computation is nested
// within the root computation, but it's emitted using NpuIrEmitterUnnested!  Don't
// think about it too hard.
// Examples of things that are not unnested computations:
//  - The body of a fusion node.  NpuIrEmitterUnnested emits the relevant code
//    within a kernel function using FusedIrEmitter.  (FusedIrEmitter is not
//    really an NpuIrEmitter, but is more an "IR generator generator".)
class NpuIrEmitterUnnested : public NpuIrEmitter {
   public:
    absl::string_view platform_name() const { return ir_emitter_context_->platform_name(); }

    using ValueVector3 = std::array<llvm::Value*, 3>;
    using ValueVector2 = std::array<llvm::Value*, 2>;

    using ConstantGenerator = std::function<llvm::Value*(int64_t)>;

    NpuIrEmitterUnnested(const NpuIrEmitterUnnested&) = delete;
    NpuIrEmitterUnnested& operator=(const NpuIrEmitterUnnested&) = delete;

    static std::unique_ptr<NpuIrEmitterUnnested> Create(NpuIrEmitterContext* ir_emitter_context);

    // Transfers the ownership of thunk_sequence_ out.
    std::unique_ptr<NpuSequentialThunk> ConsumeThunkSequence(NpuThunk::ThunkInfo thunk_info = NpuThunk::ThunkInfo{}) {
        return std::make_unique<NpuSequentialThunk>(thunk_info, std::move(thunk_sequence_));
    }

    // Emits code for the given HLO computation.
    // Also populates related information to 'ir_emitter_context_' for
    // large-constant initializations. Large constants don't get initializers in
    // the generated code and so must be initialized by XLA. The value of these
    // constants will be stored in 'content'. Constants with initializers in the
    // generated code will have empty 'content'.
    absl::Status EmitHloComputation(const HloComputation* computation);

   private:
    explicit NpuIrEmitterUnnested(NpuIrEmitterContext* ir_emitter_context);

    // NpuIrEmitterUnnested handles the following instructions differently from
    // NpuIrEmitter. It also mixes in some special handling for custom kernels
    // via the ThunkEmitter.
    absl::Status EmitCall(const HloInstruction* instr);

    absl::Status EmitConstant(const HloConstantInstruction* instr);

    absl::Status EmitNpuAclnnThunk(const HloInstruction* instr);

    absl::Status EmitFusion(const HloFusionInstruction* instr);

    absl::Status EmitHloInstruction(const HloInstruction* instr);

    absl::Status EmitTargetElementLoop(const HloInstruction& hlo,
                                       const llvm_ir::ElementGenerator& body_emitter) override;

    // Add a owning NpuThunk object to the thunk sequence.
    void AddThunkToThunkSequence(std::unique_ptr<NpuThunk> thunk) {
        if (emit_group_thunks_) {
            scoped_thunk_sequence_.emplace_back(std::move(thunk));
            return;
        }
        thunk_sequence_.emplace_back(std::move(thunk));
    }

    absl::StatusOr<BufferAllocation::Slice> GetAllocationSliceForHlo(const HloInstruction* instr,
                                                                     const ShapeIndex& index = {}) const;
    NpuThunkSequence thunk_sequence_;
    NpuThunkSequence scoped_thunk_sequence_;
    bool emit_group_thunks_ = false;

    std::unique_ptr<CallGraph> call_graph_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_UNNESTED_H_
