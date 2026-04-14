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

#include "xla/xla_npu/service/transform/npu_fusion_block_level_rewriter.h"

#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {

// NPU Unified Buffer (UB) size in bytes (typical for Ascend 910)
constexpr int64_t kNpuUBSize = 256 * 1024;  // 256KB

// Default tile sizes for block-level fusion
constexpr int64_t kDefaultTileSize1D = 1024;
constexpr int64_t kDefaultTileSize2D = 64;
constexpr int64_t kDefaultTileSize3D = 32;
constexpr int64_t kDefaultTileSize4D = 16;

// Default number of blocks for block-level fusion
constexpr int64_t kDefaultNumBlocks = 1;

// Default number of pipeline stages
constexpr int64_t kDefaultNumStages = 2;

// Minimum number of elements to enable block-level fusion
constexpr int64_t kMinElementsForBlockLevel = 1024;

// NPU fusion kind constants
constexpr char kNpuBlockingFusionKind[] = "__npu_blocking";

}  // namespace

absl::StatusOr<bool> NpuFusionBlockLevelRewriter::RunImpl(
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

absl::StatusOr<bool> NpuFusionBlockLevelRewriter::RunOnComputation(
    HloComputation* computation) {
  bool changed = false;

  // Get all instructions in post-order
  std::vector<HloInstruction*> instrs = computation->MakeInstructionPostOrder();

  for (HloInstruction* instr : instrs) {
    // Only process loop or input fusion instructions
    if (instr->opcode() != HloOpcode::kFusion) {
      continue;
    }

    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instr);

    // Skip if already a custom fusion
    if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom) {
      continue;
    }

    // Skip multi-output fusions
    if (fusion->IsMultiOutputFusion()) {
      continue;
    }

    // Check if this fusion can be converted to block-level
    if (!CanConvertToBlockLevel(fusion)) {
      VLOG(4) << "Fusion cannot be converted to block-level: "
              << fusion->ToString();
      continue;
    }

    VLOG(3) << "Converting fusion to block-level: " << fusion->ToString();

    TF_ASSIGN_OR_RETURN(bool converted, ConvertFusionToBlockLevel(fusion));
    changed |= converted;
  }

  return changed;
}

bool NpuFusionBlockLevelRewriter::CanConvertToBlockLevel(
    const HloInstruction* fusion) {
  // Must be a loop or input fusion
  if (fusion->opcode() != HloOpcode::kFusion) {
    return false;
  }

  const auto* fusion_instr = Cast<HloFusionInstruction>(fusion);
  HloInstruction::FusionKind kind = fusion_instr->fusion_kind();
  if (kind != HloInstruction::FusionKind::kLoop &&
      kind != HloInstruction::FusionKind::kInput) {
    return false;
  }

  // Check if the fusion contains only elementwise operations
  if (!ContainsOnlyElementwiseOps(fusion_instr->called_computation())) {
    return false;
  }

  // Check if the output is large enough to benefit from block-level fusion
  int64_t num_elements = ShapeUtil::ElementsIn(fusion->shape());
  if (num_elements < kMinElementsForBlockLevel) {
    return false;
  }

  return true;
}

absl::StatusOr<bool> NpuFusionBlockLevelRewriter::ConvertFusionToBlockLevel(
    HloFusionInstruction* fusion) {
  // Compute optimal tile sizes
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> tile_sizes,
                      ComputeTileSizes(fusion));

  // Change the fusion kind to custom with block kind
  TF_ASSIGN_OR_RETURN(
      auto npu_config,
      fusion->backend_config<npu_backend_config::NpuBackendConfig>());

  // Get backend config or create if not present
  npu_backend_config::FusionBackendConfig backend_config;
  if (npu_config.has_fusion_backend_config()) {
    backend_config = npu_config.fusion_backend_config();
  }

  // Set the fusion kind to NPU block
  backend_config.set_kind(std::string(kNpuBlockingFusionKind));

  // Set up block-level configuration
  TF_RETURN_IF_ERROR(
      SetBlockLevelConfig(fusion, tile_sizes, kDefaultNumBlocks));

  // Save the backend config
  *npu_config.mutable_fusion_backend_config() = backend_config;
  TF_RETURN_IF_ERROR(fusion->set_backend_config(npu_config));

  VLOG(3) << "Converted fusion to block-level: " << fusion->ToString();
  return true;
}

absl::StatusOr<std::vector<int64_t>>
NpuFusionBlockLevelRewriter::ComputeTileSizes(const HloInstruction* fusion) {
  const Shape& shape = fusion->shape();
  int64_t rank = shape.dimensions_size();

  std::vector<int64_t> tile_sizes(rank);

  // Compute tile sizes based on rank
  switch (rank) {
    case 1:
      tile_sizes[0] = kDefaultTileSize1D;
      break;
    case 2:
      tile_sizes[0] = kDefaultTileSize2D;
      tile_sizes[1] = kDefaultTileSize2D;
      break;
    case 3:
      tile_sizes[0] = kDefaultTileSize3D;
      tile_sizes[1] = kDefaultTileSize3D;
      tile_sizes[2] = kDefaultTileSize3D;
      break;
    default:
      // For higher dimensional tensors, use smaller tiles
      for (int i = 0; i < rank; ++i) {
        tile_sizes[i] = kDefaultTileSize4D;
      }
      break;
  }

  // Clamp tile sizes to the actual shape dimensions
  const auto& dimensions = shape.dimensions();
  for (int i = 0; i < rank; ++i) {
    tile_sizes[i] = std::min(tile_sizes[i], dimensions[i]);
  }

  return tile_sizes;
}

bool NpuFusionBlockLevelRewriter::ContainsOnlyElementwiseOps(
    HloComputation* fusion) {
  for (HloInstruction* instr : fusion->MakeInstructionPostOrder()) {
    // Skip parameters
    if (instr->opcode() == HloOpcode::kParameter) {
      continue;
    }

    // Check if it's an elementwise operation
    switch (instr->opcode()) {
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
      case HloOpcode::kClamp:
      case HloOpcode::kAbs:
      case HloOpcode::kNegate:
      case HloOpcode::kExp:
      case HloOpcode::kLog:
      case HloOpcode::kSqrt:
      case HloOpcode::kRsqrt:
      case HloOpcode::kPower:
      case HloOpcode::kTanh:
      case HloOpcode::kLogistic:
      case HloOpcode::kBitcast:
      case HloOpcode::kReshape:
      case HloOpcode::kBroadcast:
        continue;
      default:
        return false;
    }
  }
  return true;
}

absl::Status NpuFusionBlockLevelRewriter::SetBlockLevelConfig(
    HloFusionInstruction* fusion, const std::vector<int64_t>& tile_sizes,
    int64_t num_blocks) {
  TF_ASSIGN_OR_RETURN(
      auto npu_config,
      fusion->backend_config<npu_backend_config::NpuBackendConfig>());

  // Get or create backend config
  npu_backend_config::FusionBackendConfig* backend_config =
      npu_config.mutable_fusion_backend_config();

  npu_backend_config::NpuBlockLevelConfig block_config;
  *block_config.mutable_output_tile_sizes() = {tile_sizes.begin(),
                                               tile_sizes.end()};
  block_config.set_num_blocks(num_blocks);
  block_config.set_num_stages(kDefaultNumStages);
  *backend_config->mutable_block_config() = block_config;

  TF_RETURN_IF_ERROR(fusion->set_backend_config(npu_config));
  return absl::OkStatus();
}

}  // namespace xla
