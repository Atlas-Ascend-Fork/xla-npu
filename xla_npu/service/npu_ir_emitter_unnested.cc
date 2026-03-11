/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/service/npu_ir_emitter_unnested.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/human_readable_json.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/ffi/attribute_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "xla/xla_npu/backend/codegen/npu_custom.h"
#include "xla/xla_npu/backend/codegen/npu_fusion_emitter.h"
#include "xla/xla_npu/backend/runtime/npu_aclnn_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_call_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_custom_kernel_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/npu_execution_stream_assignment.h"
#include "xla/xla_npu/service/npu_ir_emission_utils.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

namespace xla {
NpuIrEmitterUnnested::NpuIrEmitterUnnested(NpuIrEmitterContext* ir_emitter_context)
    : NpuIrEmitter(ir_emitter_context, /*is_nested=*/false),
      call_graph_(CallGraph::Build(&ir_emitter_context->hlo_module())) {}

std::unique_ptr<NpuIrEmitterUnnested> NpuIrEmitterUnnested::Create(NpuIrEmitterContext* ir_emitter_context) {
    return std::unique_ptr<NpuIrEmitterUnnested>(new NpuIrEmitterUnnested(ir_emitter_context));
}

absl::Status NpuIrEmitterUnnested::EmitConstant(const HloConstantInstruction* instr) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Emit const thunk");

    TF_ASSIGN_OR_RETURN(NpuDenseDataIntermediate content, NpuLiteralToXlaFormat(instr->literal()));

    int element_bytes = primitive_util::ByteWidth(instr->literal().shape().element_type());
    TF_RET_CHECK(content.span().size() % element_bytes == 0);
    // Treat packed constants as a byte constant.
    int num_elements = content.span().size() / element_bytes;

    std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice, GetAllocationSliceForHlo(instr, {}));

    ir_emitter_context_->emit_constant(num_elements, element_bytes, global_name, slice.index(), std::move(content),
                                       &b_);
    return absl::OkStatus();
}

absl::Status NpuIrEmitterUnnested::EmitNpuAclnnThunk(const HloInstruction* instr) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Emit aclnn thunk");

    TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                        emitters::KernelArguments::Create(ir_emitter_context_->buffer_assignment(),
                                                          NpuConstant::GetDefaultBufferAlignment(), instr));
    const std::string fingerprint = "";
    // check if sdpa dropout is enabled
    std::optional<int64_t> dropout_seed = std::nullopt;

    // TODO 运行mmoe目前需要打桩此代码
    AddThunkToThunkSequence(std::make_unique<NpuAclnnThunk>(
        fingerprint, NpuThunk::ThunkInfo::WithProfileAnnotation(instr, ir_emitter_context_->GetNextThunkId()),
        kernel_arguments.GetArgumentBufferSlices(), kernel_arguments.GetArgumentOutputFlags(), dropout_seed));

    return absl::OkStatus();
}

absl::StatusOr<BufferAllocation::Slice> NpuIrEmitterUnnested::GetAllocationSliceForHlo(const HloInstruction* instr,
                                                                                       const ShapeIndex& index) const {
    return NpuGetAllocationSlice(ir_emitter_context_->buffer_assignment(), instr, index);
}

absl::Status NpuIrEmitterUnnested::EmitFusion(const HloFusionInstruction* instr) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Emit fusion thunk");

    // TODO 运行mmoe目前需要打桩此代码
    std::unique_ptr<NpuFusionInterface> emitter = std::make_unique<NpuCustomFusion>();
    TF_ASSIGN_OR_RETURN(auto result, emitter->Emit(*ir_emitter_context_, *instr));

    const NpuExecutionStreamAssignment& stream_assignment = ir_emitter_context_->execution_stream_assignment();
    for (std::unique_ptr<NpuThunk>& thunk : result.thunks) {
        TF_ASSIGN_OR_RETURN(NpuExecutionStreamId execution_stream_id,
                            stream_assignment.GetSyncExecutionStreamId(instr));
        thunk->set_execution_stream_id(execution_stream_id);
        AddThunkToThunkSequence(std::move(thunk));
    }

    return absl::OkStatus();
}

absl::Status NpuIrEmitterUnnested::EmitCall(const HloInstruction* instr) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Emit call thunk");

    const HloComputation* called_computation = instr->called_computations().front();
    auto ir_emitter = NpuIrEmitterUnnested::Create(ir_emitter_context_);
    NPU_LOG_FORMAT_PRINT(DEBUG, "====Start generate sub hlo instruction thunk====");
    TF_RETURN_IF_ERROR(ir_emitter->EmitHloComputation(called_computation));
    NPU_LOG_FORMAT_PRINT(DEBUG, "====End generate sub hlo instruction thunk====");
    std::unique_ptr<NpuSequentialThunk> called_thunk_sequence = ir_emitter->ConsumeThunkSequence(
        NpuThunk::ThunkInfo::WithProfileAnnotation(instr, ir_emitter_context_->GetNextThunkId()));
    AddThunkToThunkSequence(std::make_unique<NpuCallThunk>(
        NpuThunk::ThunkInfo::WithProfileAnnotation(instr, ir_emitter_context_->GetNextThunkId()),
        std::move(called_thunk_sequence->thunks())));
    return absl::OkStatus();
}

static const HloInstruction* FindCanonicalSendRecvStartOp(const HloInstruction* inst) {
    CHECK(inst->opcode() == HloOpcode::kSend || inst->opcode() == HloOpcode::kRecv ||
          inst->opcode() == HloOpcode::kSendDone || inst->opcode() == HloOpcode::kRecvDone);
    // If the instruction is wrapped in an async computation, return
    // the instruction itself.
    if (inst->parent()->IsAsyncComputation()) {
        return inst;
    }

    // Find container while loop and index for the send/recv case or
    // return canonical start op directly.
    const HloInstruction* while_op = nullptr;
    int64_t i = -1;
    if (inst->opcode() == HloOpcode::kSend || inst->opcode() == HloOpcode::kRecv) {
        CHECK_EQ(inst->users().size(), 1);
        const HloInstruction* unique_user = inst->users().front();

        // Return send/recv inst directly if this is a simple send/recv
        // pair.
        if (unique_user->opcode() == HloOpcode::kSendDone || unique_user->opcode() == HloOpcode::kRecvDone) {
            return inst;
        }

        // Find while loop and index, otherwise.
        CHECK(unique_user->opcode() == HloOpcode::kTuple || unique_user->opcode() == HloOpcode::kWhile);
        if (unique_user->IsRoot()) {
            // send/recv op in the loop body.
            auto maybe_while_op = unique_user->parent()->GetUniqueCaller(HloOpcode::kWhile);
            CHECK(maybe_while_op);
            while_op = *maybe_while_op;
            i = unique_user->operand_index(inst);
        } else {
            // send/recv leading into the loop.
            CHECK_EQ(unique_user->users().size(), 1);
            CHECK(unique_user->users().front()->opcode() == HloOpcode::kWhile);
            while_op = unique_user->users().front();
            i = unique_user->operand_index(inst);
        }
    }

    // Find container while loop and index for the send-done/recv-done
    // case or return canonical start op directly.
    if (inst->opcode() == HloOpcode::kSendDone || inst->opcode() == HloOpcode::kRecvDone) {
        const HloInstruction* operand = inst->operand(0);

        // Return send/recv inst directly if this is a simple send/recv
        // pair.
        if (operand->opcode() == HloOpcode::kSend || operand->opcode() == HloOpcode::kRecv) {
            return operand;
        }

        // Find while loop and index, otherwise.
        CHECK(operand->opcode() == HloOpcode::kGetTupleElement);
        const auto* gte = Cast<HloGetTupleElementInstruction>(operand);
        const HloInstruction* iter_tuple = operand->operand(0);
        if (iter_tuple->opcode() == HloOpcode::kParameter) {
            // send-done/recv-done in the loop body.
            CHECK(Cast<HloParameterInstruction>(iter_tuple)->parameter_number() == 0);
            auto maybe_while = iter_tuple->parent()->GetUniqueCaller(HloOpcode::kWhile);
            CHECK(maybe_while);
            while_op = *maybe_while;
            i = gte->tuple_index();
        } else {
            // send-done/recv-done proceeding the loop.
            CHECK(iter_tuple->opcode() == HloOpcode::kWhile);
            while_op = iter_tuple;
            i = gte->tuple_index();
        }
    }

    // Extract canonical start op from while loop's init.
    CHECK(while_op != nullptr);
    CHECK(0 <= i && i < while_op->shape().tuple_shapes().size());
    const HloInstruction* init = while_op->operand(0);
    const HloInstruction* canonical_start_op = init->operand(i);
    CHECK(canonical_start_op->opcode() == HloOpcode::kSend || canonical_start_op->opcode() == HloOpcode::kRecv);
    return canonical_start_op;
}

std::vector<const HloInstruction*> GetRealDependencyInstructions(const HloInstruction* instr) {
    std::vector<const HloInstruction*> real_deps;
    switch (instr->opcode()) {
        case HloOpcode::kSend:
        case HloOpcode::kSendDone:
        case HloOpcode::kRecv:
        case HloOpcode::kRecvDone:
            return {FindCanonicalSendRecvStartOp(instr)};
        case HloOpcode::kAllGatherDone:
        case HloOpcode::kAllReduceDone:
        case HloOpcode::kAsyncDone:
        case HloOpcode::kCollectivePermuteDone:
        case HloOpcode::kCopyDone:
            return {instr->operand(0)};
        case HloOpcode::kAllGatherStart:
        case HloOpcode::kAllReduceStart:
        case HloOpcode::kAsyncStart:
        case HloOpcode::kCollectivePermuteStart:
        case HloOpcode::kCall:
        case HloOpcode::kConditional:
        case HloOpcode::kConstant:
        case HloOpcode::kCustomCall:
        case HloOpcode::kFusion:
        case HloOpcode::kCopy:
        case HloOpcode::kInfeed:
        case HloOpcode::kOutfeed:
        case HloOpcode::kPartitionId:
        case HloOpcode::kFft:
        case HloOpcode::kReplicaId:
        case HloOpcode::kRngGetAndUpdateState:
        case HloOpcode::kSort:
        case HloOpcode::kWhile:
        case HloOpcode::kCopyStart:
            return {instr};
        case HloOpcode::kAddDependency:
        case HloOpcode::kAfterAll:
        case HloOpcode::kTuple:
            for (const HloInstruction* operand : instr->operands()) {
                auto deps = GetRealDependencyInstructions(operand);
                real_deps.insert(real_deps.end(), deps.begin(), deps.end());
            }
            return real_deps;
        case HloOpcode::kBitcast:
        case HloOpcode::kGetTupleElement: {
            auto deps = GetRealDependencyInstructions(instr->operand(0));
            real_deps.insert(real_deps.end(), deps.begin(), deps.end());
        }
            return real_deps;
        default:
            return {};
    }
}

absl::Status NpuIrEmitterUnnested::EmitTargetElementLoop(const HloInstruction& hlo,
                                                         const llvm_ir::ElementGenerator& body_emitter) {
    return Internal("This should be unreachable");
}

absl::Status NpuIrEmitterUnnested::EmitHloInstruction(const HloInstruction* instr) {
    switch (instr->opcode()) {
        case HloOpcode::kAbs:
        case HloOpcode::kAcos:
        case HloOpcode::kAcosh:
        case HloOpcode::kAsin:
        case HloOpcode::kAsinh:
        case HloOpcode::kAdd:
        case HloOpcode::kAnd:
        case HloOpcode::kAtan2:
        case HloOpcode::kAtanh:
        case HloOpcode::kBroadcast:
        case HloOpcode::kBitcastConvert:
        case HloOpcode::kCbrt:
        case HloOpcode::kCeil:
        case HloOpcode::kClamp:
        case HloOpcode::kClz:
        case HloOpcode::kCompare:
        case HloOpcode::kComplex:
        case HloOpcode::kConvert:
        case HloOpcode::kCos:
        case HloOpcode::kCosh:
        case HloOpcode::kDivide:
        case HloOpcode::kErf:
        case HloOpcode::kExp:
        case HloOpcode::kExpm1:
        case HloOpcode::kFloor:
        case HloOpcode::kGather:
        case HloOpcode::kImag:
        case HloOpcode::kIota:
        case HloOpcode::kIsFinite:
        case HloOpcode::kLog1p:
        case HloOpcode::kLog:
        case HloOpcode::kMap:
        case HloOpcode::kMaximum:
        case HloOpcode::kMinimum:
        case HloOpcode::kMultiply:
        case HloOpcode::kNegate:
        case HloOpcode::kNot:
        case HloOpcode::kOr:
        case HloOpcode::kPopulationCount:
        case HloOpcode::kPower:
        case HloOpcode::kReal:
        case HloOpcode::kReducePrecision:
        case HloOpcode::kRemainder:
        case HloOpcode::kReshape:
        case HloOpcode::kReverse:
        case HloOpcode::kRoundNearestAfz:
        case HloOpcode::kRoundNearestEven:
        case HloOpcode::kRsqrt:
        case HloOpcode::kSelect:
        case HloOpcode::kShiftLeft:
        case HloOpcode::kShiftRightArithmetic:
        case HloOpcode::kShiftRightLogical:
        case HloOpcode::kSign:
        case HloOpcode::kSin:
        case HloOpcode::kSinh:
        case HloOpcode::kSqrt:
        case HloOpcode::kSubtract:
        case HloOpcode::kTranspose:
        case HloOpcode::kTan:
        case HloOpcode::kTanh:
        case HloOpcode::kXor:
        case HloOpcode::kDot:  // 自动融合暂不支持matmul,以aclnn处理
        case HloOpcode::kConcatenate:
        case HloOpcode::kReduce: // TODO reduce需要特殊处理
        case HloOpcode::kDynamicSlice:
            return EmitNpuAclnnThunk(instr);

        case HloOpcode::kCall:
            return EmitCall(Cast<HloCallInstruction>(instr));
        case HloOpcode::kConstant:
            return EmitConstant(Cast<HloConstantInstruction>(instr));
        case HloOpcode::kCustomCall:
            // TODO 实现此功能
            return absl::OkStatus();
        case HloOpcode::kFusion:
            return EmitFusion(Cast<HloFusionInstruction>(instr));
        // HLO module is already scheduled, so instructions for ordering
        // are noops.
        case HloOpcode::kAddDependency:
        case HloOpcode::kAfterAll:
        // We don't need to emit thunks for these operations because
        // their semantics are encoded by buffers.
        case HloOpcode::kBitcast:
        case HloOpcode::kGetTupleElement:
        case HloOpcode::kParameter:
        case HloOpcode::kTuple:
            NPU_LOG_FORMAT_PRINT(DEBUG, "Emit no thunk");
            return absl::OkStatus();
        default:
            return Internal("Unsupported instruction opcode: %s", HloOpcodeString(instr->opcode()));
    }

    return Internal("Unhandled HLO instruction");
}

absl::Status NpuIrEmitterUnnested::EmitHloComputation(const HloComputation* computation) {
    const HloSchedule& schedule = computation->parent()->schedule();
    if (!schedule.is_computation_scheduled(computation)) {
        return Internal("Sequence not found for computation: %s", computation->name());
    }

    const HloInstructionSequence& sequence = schedule.sequence(computation);
    absl::flat_hash_map<const HloInstruction*, NpuThunk*> instr_to_thunk;
    int id = 0;
    for (const HloInstruction* instr : sequence.instructions()) {
        // ========== 打印instr的完整信息 ==========
        NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
        NPU_LOG_FORMAT_PRINT(DEBUG, "[FULL Info for HloInstruction %d] ", id++);
        HloPrintOptions options = NpuTool::GetDetailedPrintOptions();
        std::string detailed_str = instr->ToString(options);
        NPU_LOG_FORMAT_PRINT(DEBUG, detailed_str);

        int64_t previous_thunk_size = thunk_sequence_.size();
        TF_RETURN_IF_ERROR(EmitHloInstruction(instr));

        NPU_LOG_FORMAT_PRINT(DEBUG, "  ----------------------------------------");
        NPU_LOG_FORMAT_PRINT(DEBUG, "  Target thunk Info: ");

        if (thunk_sequence_.size() > previous_thunk_size) {
            instr_to_thunk[instr] = thunk_sequence_.back().get();

            const auto& thunk = thunk_sequence_.back();
            NPU_LOG_FORMAT_PRINT(DEBUG, "    Op Code: %s", HloOpcodeString(thunk->thunk_info().op_code));
            NPU_LOG_FORMAT_PRINT(DEBUG, "    Op Name: %s", thunk->thunk_info().op_name);
            NPU_LOG_FORMAT_PRINT(DEBUG, "    Op Shape size: %d", thunk->thunk_info().param_shapes.size());
            for (int i = 0; i < thunk->thunk_info().param_shapes.size(); ++i) {
                NpuTool::PrintXlaShape(thunk->thunk_info().param_shapes[i],
                                       absl::StrFormat("    Op param %d shape: ", i));
            }
        } else {
            // 某些op类型不会分配thunk,比如constant类型
            NPU_LOG_FORMAT_PRINT(DEBUG, "    No thunk info.");
        }
    }
    return absl::OkStatus();
}

}  // namespace xla
