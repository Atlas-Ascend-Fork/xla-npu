/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_SERVICE_NPU_HLO_FUSION_ANALYSIS_H_
#define XLA_XLA_NPU_SERVICE_NPU_HLO_FUSION_ANALYSIS_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/codegen/hlo_fusion_spec.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

// Returns true if the instruction's fusion backend config kind matches the
// given one.
bool IsNpuFusionKind(const HloInstruction& hlo, absl::string_view kind);

class NpuHloFusionAnalysis {
   public:
    // The type of emitted fusion.
    enum class EmitterFusionKind {
        kLoop,
        kCustomFusion,
        kReduction,
        kTranspose,
        kConcatenate,
        kScatter,
        kDynamicMemcpy,
        // NPU ACLNN specific types (using kNpuAclnn prefix):
        kNpuAclnnMatMul,       // NPU ACLNN matrix multiplication
        kNpuAclnnSoftmax,      // NPU ACLNN softmax
        kNpuAclnnLayerNorm,    // NPU ACLNN layer normalization
        kNpuAclnnBatchMatMul,  // NPU ACLNN batch matrix multiplication
        kNpuAclnnGelu,         // NPU ACLNN GELU
        kNpuAclnnConv2d,       // NPU ACLNN convolution
    };

    // Precomputed information about inputs (arguments) and outputs (roots) of the
    // fusion.
    struct InputOutputInfo {
        int smallest_input_dtype_bits;
        int smallest_output_dtype_bits;
    };

    static NpuHloFusionAnalysis Create(
        // FusionBackendConfig backend_config,
        std::unique_ptr<HloFusionAdaptor> fusion, const se::DeviceDescription* device_info);

    // Creates a NpuHloFusionAnalysis that analyzes just instruction as a standalone
    // fusion.
    static NpuHloFusionAnalysis Create(const HloInstruction& instruction, const se::DeviceDescription& device_info);

    // Creates a NpuHloFusionAnalysis that analyzes a hypothetical fusion of producer
    // into consumer.
    static NpuHloFusionAnalysis Create(const HloInstruction& producer, const HloInstruction& consumer,
                                       const se::DeviceDescription& device_info);

    const HloFusionAdaptor& fusion() const { return fusion_spec_.fusion(); }
    const HloFusionSpec& fusion_spec() const { return fusion_spec_; }

    absl::Span<const HloInstructionAdaptor> fusion_roots() const { return fusion_spec_.fusion_roots(); }
    HloInstructionAdaptor fusion_root(int64_t i) const { return fusion_spec_.fusion_root(i); }
    int64_t fusion_root_count() const { return fusion_spec_.fusion_root_count(); }

    absl::Span<const HloInstructionAdaptor> fusion_heroes() const { return fusion_spec_.fusion_heroes(); }
    HloInstructionAdaptor fusion_hero(int64_t i) const { return fusion_spec_.fusion_hero(i); }

    EmitterFusionKind emitter_fusion_kind() const { return emitter_fusion_kind_; }

    // Returns the hero reduction of the computation.
    const HloInstruction* FindHeroReduction() const;

    const se::DeviceDescription& device_info() const { return *device_info_; }

    // Returns the tiled transpose description. Requires that emitter_fusion_kind_
    // is kTranspose.
    const TransposeDescription& tiled_transpose() const {
        CHECK(tiled_transpose_.has_value());
        return *tiled_transpose_;
    }

    const InputOutputInfo& input_output_info() const { return input_output_info_; }

   private:
    NpuHloFusionAnalysis(
        HloFusionSpec fusion_spec, EmitterFusionKind emitter_fusion_kind, const se::DeviceDescription* device_info,
        std::optional<TransposeDescription> tiled_transpose, InputOutputInfo input_output_info);

    bool HasConsistentTransposeHeros() const;

    HloFusionSpec fusion_spec_;
    EmitterFusionKind emitter_fusion_kind_;

    const se::DeviceDescription* device_info_;
    std::optional<TransposeDescription> tiled_transpose_;
    InputOutputInfo input_output_info_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_HLO_FUSION_ANALYSIS_H_
