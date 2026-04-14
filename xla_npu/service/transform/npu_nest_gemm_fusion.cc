/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/xla_npu/service/transform/npu_nest_gemm_fusion.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {

// Default tile sizes for nested GEMM fusion
constexpr int64_t kDefaultTileSizeM = 128;
constexpr int64_t kDefaultTileSizeN = 128;
constexpr int64_t kDefaultTileSizeK = 32;

// Default number of blocks for nested fusion
constexpr int64_t kDefaultNumBlocks = 1;

}  // namespace

// Type aliases for protobuf generated types
using npu_backend_config::FusionBackendConfig;
using npu_backend_config::NpuBackendConfig;
using npu_backend_config::NpuBlockLevelConfig;
using npu_backend_config::NpuGemmConfig;

absl::StatusOr<bool> NpuNestGemmFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool computation_changed,
                        RunOnComputation(computation));
    changed |= computation_changed;
  }
  return changed;
}

absl::StatusOr<bool> NpuNestGemmFusion::RunOnComputation(
    HloComputation* computation) {
  bool changed = false;
  // Get all instructions in post-order
  std::vector<HloInstruction*> instrs = computation->MakeInstructionPostOrder();
  for (HloInstruction* instr : instrs) {
    // Only process custom fusion instructions
    if (instr->opcode() != HloOpcode::kFusion) {
      continue;
    }
    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instr);
    if (!IsNpuGemmFusion(fusion)) {
      continue;
    }
    VLOG(3) << "Transforming NPU GEMM fusion to nested: " << fusion->ToString();
    TF_ASSIGN_OR_RETURN(bool transformed, TransformGemmFusionToNested(fusion));
    changed |= transformed;
  }
  return changed;
}

bool NpuNestGemmFusion::IsNpuGemmFusion(const HloInstruction* fusion) {
  if (fusion->opcode() != HloOpcode::kFusion) {
    return false;
  }

  auto result = fusion->backend_config<NpuBackendConfig>();
  if (!result.ok()) {
    return false;
  }

  const NpuBackendConfig& config = result.value();
  if (!config.has_fusion_backend_config()) {
    return false;
  }

  const FusionBackendConfig& backend_config = config.fusion_backend_config();
  return backend_config.kind() == kNpuAclnnGemmFusionKind;
}

absl::StatusOr<bool> NpuNestGemmFusion::TransformGemmFusionToNested(
    HloFusionInstruction* fusion) {
  // Find the dot instruction in the fusion
  TF_ASSIGN_OR_RETURN(HloInstruction * dot,
                      FindDotInFusion(fusion->called_computation()));
  if (dot == nullptr) {
    VLOG(3) << "No dot found in GEMM fusion, skipping nesting";
    return false;
  }

  // Change the fusion kind to nested GEMM
  TF_ASSIGN_OR_RETURN(auto npu_config,
                      fusion->backend_config<NpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *npu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kNpuNestedGemmFusionKind));

  // Update the config with block-level parameters
  if (backend_config.has_npu_gemm_config()) {
    const NpuGemmConfig& gemm_config = backend_config.npu_gemm_config();

    // Set block level parameters using output_tile_sizes array
    NpuBlockLevelConfig block_config;
    block_config.add_output_tile_sizes(gemm_config.block_m());
    block_config.add_output_tile_sizes(gemm_config.block_n());
    block_config.add_output_tile_sizes(gemm_config.block_k());
    block_config.set_num_blocks(kDefaultNumBlocks);
    *backend_config.mutable_block_config() = block_config;
  }

  TF_RETURN_IF_ERROR(fusion->set_backend_config(npu_config));

  VLOG(3) << "Transformed fusion to nested: " << fusion->ToString();
  return true;
}

absl::StatusOr<HloInstruction*> NpuNestGemmFusion::FindDotInFusion(
    HloComputation* fusion) {
  for (HloInstruction* instr : fusion->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kDot) {
      return instr;
    }
  }
  return nullptr;
}

absl::StatusOr<HloInstruction*> NpuNestGemmFusion::CreateNestedFusionForOperand(
    HloInstruction* operand, HloInstruction* consumer, absl::string_view kind,
    int64_t tile_size) {
  // This is a simplified implementation
  // In a full implementation, we would create nested fusions for complex operand patterns
  return nullptr;
}

absl::Status NpuNestGemmFusion::UpdateNestedFusionConfig(
    HloFusionInstruction* nested_fusion, int64_t tile_size,
    int64_t num_blocks) {
  TF_ASSIGN_OR_RETURN(auto npu_config,
                      nested_fusion->backend_config<NpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *npu_config.mutable_fusion_backend_config();

  NpuBlockLevelConfig block_config;
  block_config.add_output_tile_sizes(tile_size);
  block_config.set_num_blocks(num_blocks);
  *backend_config.mutable_block_config() = block_config;

  TF_RETURN_IF_ERROR(nested_fusion->set_backend_config(npu_config));
  return absl::OkStatus();
}

}  // namespace xla
