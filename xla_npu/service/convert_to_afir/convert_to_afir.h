/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_SERVICE_CONVERT_TO_AFIR_H_
#define XLA_SERVICE_CONVERT_TO_AFIR_H_

#include <memory>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Configuration options for HLO to AFIR conversion
struct HloToAfirOptions {
  bool preserve_names = true;
  bool verify_module = true;
  bool emit_afir_dialect = false;  // If true, emit actual AFIR ops
};

// Convert an HloModule to MLIR Module with AFIR dialect
// This function converts an XLA HloModule to MLIR's AFIR (Ascend Frontend IR) dialect.
//       - Converting HLO computations to MLIR functions
//       - Translating HLO instructions to AFIR operations
//       - Preserving module metadata and attributes
// Args: module: The HloModule to convert
//       context: The MLIRContext to use (will load required dialects)
//       options: Conversion options
// Returns: An MLIR ModuleOp containing the converted AFIR code
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToAfir(
    const HloModule& module, mlir::MLIRContext* context,
    const HloToAfirOptions& options = {});

// Convert a single HloComputation to MLIR Module with AFIR dialect
// This function converts a single HLO computation to MLIR's AFIR dialect.
// It also includes the fused/called computations computations that this computation depends on.
// Args: computation: The HloComputation to convert
//       context: The MLIRContext to use (will load required dialects)
//       options: Conversion options
// Returns: An MLIR ModuleOp containing the converted AFIR code for this computation and its dependencies used/called computations.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloComputationToAfir(
    const HloComputation& computation, mlir::MLIRContext* context,
    const HloToAfirOptions& options = {});

// Convenience overload that creates its own MLIRContext
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToAfir(
    const HloModule& module, const HloToAfirOptions& options = {});

}  // namespace xla

#endif  // XLA_SERVICE_CONVERT_TO_AFIR_H_
