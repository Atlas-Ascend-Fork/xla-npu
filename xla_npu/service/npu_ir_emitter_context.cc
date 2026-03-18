/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/service/npu_ir_emitter_context.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Alignment.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/service/npu_executable.h"
#include "xla/xla_npu/service/npu_ir_emission_utils.h"

namespace xla {

void NpuIrEmitterContext::emit_constant(int64_t num_elements, int64_t bytes_per_element, absl::string_view symbol_name,
                                        int allocation_idx, NpuDenseDataIntermediate content, llvm::IRBuilderBase* b) {
    bool should_emit_initializer = false;

    // Ptxas has issues if the constant allocation is smaller than 64 bytes.
    // TODO(b/253259975): Remove when fixed ptxas version is submitted.
    constexpr int64_t kMinConstAllocationInBytes = 64;
    bool needs_padding = num_elements * bytes_per_element < kMinConstAllocationInBytes;

    llvm::ArrayType* global_type =
        llvm::ArrayType::get(b->getInt8Ty(), std::max(num_elements * bytes_per_element, kMinConstAllocationInBytes));

    NpuExecutable::ConstantInfo info;
    llvm::Constant* initializer = [&]() -> llvm::Constant* {
        if (!should_emit_initializer) {
            info.content = std::move(content);
            return llvm::ConstantAggregateZero::get(global_type);
        }

        std::vector<uint8_t> padded(kMinConstAllocationInBytes, 0);
        absl::c_copy(content.span(), padded.begin());
        return llvm::ConstantDataArray::get<uint8_t>(
            llvm_module_constants()->getContext(),
            needs_padding ? llvm::ArrayRef<uint8_t>(padded)
                          : llvm::ArrayRef<uint8_t>(content.span().data(), content.span().size()));
    }();

    // Explicitly set global addrspace for SPIR backend.
    int addrspace = llvm::Triple(llvm_module_constants()->getTargetTriple()).isSPIR() ? 1 : 0;
    // These globals will be looked up by name by NpuExecutable so we need to
    // give them an external linkage.  Not all of their uses are visible in
    // the LLVM IR so we can't give then a linkage that merely preserves their
    // names (like available_externally), we also need to ensure that they stick
    // around even if they're "unused".
    // We may have to be more clever here in the future if we notice that we're
    // keeping around too many globals because of their linkage.
    llvm::GlobalVariable* global_for_const = new llvm::GlobalVariable(
        global_type, /*isConstant=*/should_emit_initializer, llvm::GlobalValue::ExternalLinkage,
        /*Initializer=*/initializer, symbol_name,
        /*TLMode=*/llvm::GlobalValue::NotThreadLocal,
        /*AddressSpace=*/addrspace,
        /*isExternallyInitialized=*/false);
    global_for_const->setAlignment(llvm::Align(NpuConstant::kConstantBufferAlignBytes));
    llvm_module_constants()->insertGlobalVariable(global_for_const);

    info.symbol_name.assign(symbol_name);
    info.allocation_index = allocation_idx;
    constants_.push_back(std::move(info));
}

}  // namespace xla
