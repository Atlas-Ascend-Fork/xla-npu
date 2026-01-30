/* Copyright 2025 The OpenXLA Authors

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

#ifndef XLA_XLA_NPU_SERVICE_CONVERT_TO_AFIR_ASCIR_BACKEND_H_
#define XLA_XLA_NPU_SERVICE_CONVERT_TO_AFIR_ASCIR_BACKEND_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Configuration for AscIR backend compilation
struct AscirBackendOptions {
  // Output directory for generated files
  std::string output_dir = "./ascir_output";

  // Whether to preserve intermediate files
  bool preserve_intermediates = true;

  // SOC version (e.g., "Ascend910B1")
  std::string soc_version = "Ascend910B1";
};

// Result of AscIR backend compilation
struct AscirBackendResult {
  // Path to generated host file
  std::string host_file;

  // Path to generated device file
  std::string device_file;

  // Path to compiled device kernel binary (.bin)
  std::string device_binary_file;

  // Path to generated tiling data file
  std::string tiling_data_file;

  // Path to generated ascir protobuf text file
  std::string ascir_file;

  // Graph name extracted from ascir text
  std::string graph_name;

  // Tiling data bytes
  std::vector<uint8_t> tiling_data;
};

// AscIR Backend Interface
//
// This class provides an interface to compile HLO modules to AscIR kernels
// through the following pipeline:
// 1. HLO -> AFIR (using ConvertHloToAfir)
// 2. AFIR -> AscIR (using Python script)
// 3. AscIR -> Kernel (using Ascend runtime)
//
// Usage example:
//   AscirBackend backend;
//   AscirBackendOptions options;
//   AscirBackendResult result = backend.Compile(*hlo_module, options);
class AscirBackend {
 public:
  AscirBackend();
  ~AscirBackend();

  // Compile an HLO module to AscIR kernel
  //
  // Args:
  //   hlo_module: The HLO module to compile
  //   options: Compilation options
  //
  // Returns:
  //   AscirBackendResult containing paths to generated files
  absl::StatusOr<AscirBackendResult> Compile(
      const HloModule& hlo_module,
      const AscirBackendOptions& options = AscirBackendOptions());

 private:
  // Step 1: Convert HLO to AFIR MLIR module (in-memory)
  absl::StatusOr<std::string> ConvertHloToAfirModule(
      const HloModule& hlo_module,
      const AscirBackendOptions& options);

  // Step 2: Call Python script to convert AFIR to AscIR
  absl::StatusOr<std::string> ConvertAfirToAscir(
      const std::string& afir_string,
      const AscirBackendOptions& options);

  // Step 3: Call Ascend runtime to generate kernel
  absl::StatusOr<AscirBackendResult> GenerateKernel(
      const std::string& ascir_text,
      const AscirBackendOptions& options);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_CONVERT_TO_AFIR_ASCIR_BACKEND_H_
