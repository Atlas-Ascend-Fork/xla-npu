/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/service/npu_hlo_fusion_analysis.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/codegen/hlo_fusion_spec.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/primitive_util.h"
// #include "xla/service/gpu/backend_configs.pb.h"
// #include "xla/service/gpu/ir_emission_utils.h"
// #include "xla/service/gpu/reduction_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_npu/service/npu_ir_emission_utils.h"

namespace xla {

namespace {

// Returns a description of a transpose hero, that is compatible with all roots.
//
// A root is compatible with the transpose hero if:
//   * Either the root has a traspose hero with the same normalized dimensions
//   * Or the root output shape is equal to the the transpose input shape
std::optional<TransposeDescription> FindConsistentTransposeHero(
    const absl::InlinedVector<HloInstructionAdaptor, 2>& hlo_roots,
    const absl::InlinedVector<HloInstructionAdaptor, 2>& heroes) {
    std::optional<TransposeDescription> tiled_transpose_hero;
    std::vector<const HloInstruction*> non_transpose_roots;

    for (auto [root, hero] : llvm::zip(hlo_roots, heroes)) {
        if (auto tr = NpuGetDescriptionForTiledTransposeEmitter(hero.instruction())) {
            if (!tiled_transpose_hero) {
                // First transpose hero found.
                tiled_transpose_hero = tr;
            } else if (!tiled_transpose_hero->IsEquivalent(*tr)) {
                // Transpose heroes have different shape.
                return std::nullopt;
            }
        } else {
            non_transpose_roots.push_back(&root.instruction());
        }
    }

    if (!tiled_transpose_hero) return std::nullopt;

    for (auto* root : non_transpose_roots) {
        // Roots that don't have a transpose hero, should have a shape compatible
        // with the transpose input.
        if (!ShapeUtil::IsReshapeOrTransposeBitcast(root->shape(), tiled_transpose_hero->input_shape(),
                                                    /*ignore_element_type=*/true)) {
            return std::nullopt;
        }
    }

    return tiled_transpose_hero;
}

bool UseConcatenateFusion(absl::Span<const HloInstructionAdaptor> roots,
                          absl::Span<const HloInstructionAdaptor> heroes) {
    if (heroes.size() != 1) return false;
    if (heroes.front().opcode() != HloOpcode::kConcatenate) return false;
    // The concat emitter does not support multiple outputs yet. TODO(csigg): fix.
    if (roots.front().shape().IsTuple()) return false;
    // Limit the number of operands because the concat emitter produces code for
    // each operand, hurting occupancy.
    if (heroes.front().instruction().operand_count() > 4) return false;
    // The loop emitter is faster when warp divergence and occupancy are both low.
    // TODO(csigg): exclude this case.
    return true;
}

NpuHloFusionAnalysis::EmitterFusionKind GetEmitterFusionKind(
    // const FusionBackendConfig& fusion_backend_config,
    absl::Span<const HloInstructionAdaptor> fusion_roots, absl::Span<const HloInstructionAdaptor> fusion_heroes,
    const std::optional<TransposeDescription>& tiled_transpose, const se::DeviceDescription& device_info) {
    // if (fusion_backend_config.kind() == kCustomFusionKind) {
    //     return NpuHloFusionAnalysis::EmitterFusionKind::kCustomFusion;
    // }

    // if (fusion_backend_config.kind() == kTritonFusionKind || fusion_backend_config.kind() == kTritonGemmFusionKind ||
    //     fusion_backend_config.kind() == kTritonNestedGemmFusionKind ||
    //     fusion_backend_config.kind() == kTritonScaledDotFusionKind ||
    //     fusion_backend_config.kind() == kTritonCollectiveFusionKind) {
    //     return NpuHloFusionAnalysis::EmitterFusionKind::kTriton;
    // }

    // if (fusion_backend_config.kind() == kDynamicMemcpyFusionKind) {
    //     return NpuHloFusionAnalysis::EmitterFusionKind::kDynamicMemcpy;
    // }

    // if (fusion_backend_config.kind() == kCuDnnFusionKind) {
    //     return NpuHloFusionAnalysis::EmitterFusionKind::kCuDnn;
    // }

    // std::optional<HloInstructionAdaptor> first_reduce_hero;
    // for (auto [root, hero] : llvm::zip(fusion_roots, fusion_heroes)) {
    //     if (IsRealReductionHero(root.instruction(), hero.instruction(), device_info)) {
    //         first_reduce_hero = hero;
    //         break;
    //     }
    // }
    // if (first_reduce_hero.has_value()) {
    //     bool valid_shapes = true;
    //     Shape hero_operand_shape = first_reduce_hero->GetOperand(0).shape();
    //     for (auto [root, hero] : llvm::zip(fusion_roots, fusion_heroes)) {
    //         if (root == *first_reduce_hero) {
    //             continue;
    //         }
    //         if (!IsRealReductionHero(root.instruction(), hero.instruction(), device_info)) {
    //             // Needs to have a compatible shape to the reduce operand (compatible
    //             // meaning same number of elements).
    //             if (ShapeUtil::ElementsIn(root.shape()) != ShapeUtil::ElementsIn(hero_operand_shape)) {
    //                 valid_shapes = false;
    //                 break;
    //             }
    //         } else if (!AreReductionsMultiOutputFusionCompatible(&hero.instruction(),
    //                                                              &first_reduce_hero->instruction())) {
    //             valid_shapes = false;
    //             break;
    //         }
    //     }
    //     if (valid_shapes) {
    //         return NpuHloFusionAnalysis::EmitterFusionKind::kReduction;
    //     }
    // }

    // We expect that the last dimension is swapped with a different dimension.
    if (tiled_transpose.has_value()) {
        return NpuHloFusionAnalysis::EmitterFusionKind::kTranspose;
    }

    if (fusion_roots.size() > 1) {
        return NpuHloFusionAnalysis::EmitterFusionKind::kLoop;
    }

    if (fusion_roots[0].opcode() == HloOpcode::kScatter) {
        return NpuHloFusionAnalysis::EmitterFusionKind::kScatter;
    }

    if (UseConcatenateFusion(fusion_roots, fusion_heroes)) {
        return NpuHloFusionAnalysis::EmitterFusionKind::kConcatenate;
    }

    return NpuHloFusionAnalysis::EmitterFusionKind::kLoop;
}

const Shape& GetShape(const HloInstructionAdaptor& adaptor) { return adaptor.shape(); }

const Shape& GetShape(const HloInstruction* instruction) { return instruction->shape(); }

template <typename Container>
int SmallestBitWidth(const Container& args) {
    int bits = std::numeric_limits<int>::max();
    for (const auto& operand : args) {
        const Shape& shape = GetShape(operand);
        if (!shape.IsArray()) continue;
        bits = std::min(bits, shape.element_type() == PRED ? 8 : primitive_util::BitWidth(shape.element_type()));
    }
    return bits;
}

}  // namespace

bool IsNpuFusionKind(const HloInstruction& hlo, absl::string_view kind) {
    // auto gpu_config = hlo.backend_config<GpuBackendConfig>();
    // if (!gpu_config.ok()) {
    //     return false;
    // }
    // return gpu_config->fusion_backend_config().kind() == kind;
    return false;  // TODO 实现NPU的fusion instruction类型判断
}

NpuHloFusionAnalysis::NpuHloFusionAnalysis(
    //  FusionBackendConfig fusion_backend_config,
    HloFusionSpec fusion_spec, EmitterFusionKind emitter_fusion_kind, const se::DeviceDescription* device_info,
    std::optional<TransposeDescription> tiled_transpose, NpuHloFusionAnalysis::InputOutputInfo input_output_info)
    :  // fusion_backend_config_(std::move(fusion_backend_config)),
      fusion_spec_(std::move(fusion_spec)),
      emitter_fusion_kind_(emitter_fusion_kind),
      device_info_(device_info),
      tiled_transpose_(tiled_transpose),
      input_output_info_(std::move(input_output_info)) {}

// static
NpuHloFusionAnalysis NpuHloFusionAnalysis::Create(
    // FusionBackendConfig backend_config,
    std::unique_ptr<HloFusionAdaptor> fusion, const se::DeviceDescription* device_info) {
    absl::InlinedVector<HloInstructionAdaptor, 2> roots = fusion->GetRoots();
    absl::InlinedVector<HloInstructionAdaptor, 2> heroes;
    for (auto root : roots) {
        heroes.push_back(NpuFindNonTrivialHero(root));
    }

    InputOutputInfo input_output_info{
        /*smallest_input_dtype_bits=*/SmallestBitWidth(fusion->GetParameters()),
        /*smallest_output_dtype_bits=*/SmallestBitWidth(roots),
    };

    std::optional<TransposeDescription> tiled_transpose_hero = FindConsistentTransposeHero(roots, heroes);

    EmitterFusionKind emitter_fusion_kind = GetEmitterFusionKind(
        // backend_config,
        roots, heroes, tiled_transpose_hero, *device_info);

    // The loop emitter always uses the roots as heroes. Even if we found some
    // non-trivial heroes in the fusion, some logic in GetEmitterFusionKind might
    // still decide to use the loop emitter and in that case we need to reset the
    // heroes to the roots.
    if (emitter_fusion_kind == EmitterFusionKind::kLoop) {
        heroes = roots;
        tiled_transpose_hero = std::nullopt;
    }

    HloFusionSpec fusion_spec(std::move(fusion), std::move(roots), std::move(heroes));

    return NpuHloFusionAnalysis(
        // std::move(backend_config),
        std::move(fusion_spec), emitter_fusion_kind, device_info, tiled_transpose_hero, std::move(input_output_info));
}

// static
NpuHloFusionAnalysis NpuHloFusionAnalysis::Create(const HloInstruction& instruction,
                                                  const se::DeviceDescription& device_info) {
    // absl::StatusOr<GpuBackendConfig> gpu_backend_config = instruction.backend_config<GpuBackendConfig>();

    // FusionBackendConfig fusion_backend_config =
    //     gpu_backend_config.ok() ? gpu_backend_config->fusion_backend_config() :
    //     FusionBackendConfig::default_instance();
    return Create(
        // std::move(fusion_backend_config),
        HloFusionAdaptor::ForInstruction(&instruction), &device_info);
}

// static
NpuHloFusionAnalysis NpuHloFusionAnalysis::Create(const HloInstruction& producer, const HloInstruction& consumer,
                                                  const se::DeviceDescription& device_info) {
    // absl::StatusOr<GpuBackendConfig> gpu_backend_config;

    // if (consumer.has_backend_config()) {
    //     gpu_backend_config = consumer.backend_config<GpuBackendConfig>();
    // }

    // if (!gpu_backend_config.ok() && producer.has_backend_config()) {
    //     gpu_backend_config = producer.backend_config<GpuBackendConfig>();
    // }

    // FusionBackendConfig fusion_backend_config =
    //     gpu_backend_config.ok() ? gpu_backend_config->fusion_backend_config() :
    //     FusionBackendConfig::default_instance();

    return NpuHloFusionAnalysis::Create(
        // std::move(fusion_backend_config),
        HloFusionAdaptor::ForProducerConsumer(&producer, &consumer), &device_info);
}

const HloInstruction* NpuHloFusionAnalysis::FindHeroReduction() const {
    // TODO implement
    return nullptr;

    // if (emitter_fusion_kind_ != EmitterFusionKind::kReduction) {
    //     return nullptr;
    // }
    // const auto& roots = fusion_roots();
    // CHECK(!roots.empty());
    // // We always use the first reduce root that triggers unnested reduction
    // // emitter as the hero reduction, since all the reductions are required to
    // // have the same shape and layout as verified by
    // // `IsFusedReductionOutputConsistent()`.
    // for (auto [root, hero] : llvm::zip(roots, fusion_heroes())) {
    //     if (IsRealReductionHero(root.instruction(), hero.instruction(), *device_info_)) {
    //         return &hero.instruction();
    //     }
    // }
    // LOG(FATAL) << "Did not find a hero reduction";
}

}  // namespace xla
