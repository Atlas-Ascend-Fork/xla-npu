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
// #include "xla/service/gpu/backend_configs.pb.h"
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
        kTriton,
        kReduction,
        kTranspose,
        kConcatenate,
        kScatter,
        kCuDnn,
        kDynamicMemcpy,
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

    // const FusionBackendConfig& fusion_backend_config() const { return fusion_backend_config_; }

    // Returns the tiled transpose description. Requires that emitter_fusion_kind_
    // is kTranspose.
    const TransposeDescription& tiled_transpose() const {
        CHECK(tiled_transpose_.has_value());
        return *tiled_transpose_;
    }

    const InputOutputInfo& input_output_info() const { return input_output_info_; }

   private:
    NpuHloFusionAnalysis(
        // FusionBackendConfig fusion_backend_config,
        HloFusionSpec fusion_spec, EmitterFusionKind emitter_fusion_kind, const se::DeviceDescription* device_info,
        std::optional<TransposeDescription> tiled_transpose, InputOutputInfo input_output_info);

    bool HasConsistentTransposeHeros() const;

    // FusionBackendConfig fusion_backend_config_;

    HloFusionSpec fusion_spec_;
    EmitterFusionKind emitter_fusion_kind_;

    const se::DeviceDescription* device_info_;
    std::optional<TransposeDescription> tiled_transpose_;
    InputOutputInfo input_output_info_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_HLO_FUSION_ANALYSIS_H_
