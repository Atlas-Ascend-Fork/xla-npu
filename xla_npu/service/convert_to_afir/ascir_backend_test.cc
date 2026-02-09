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

#include "xla/xla_npu/service/convert_to_afir/ascir_backend.h"

#include <fstream>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/xla_data.pb.h"

ABSL_FLAG(std::string, hlo_text, "", "HLO text to compile");
ABSL_FLAG(std::string, hlo_file, "", "HLO file to compile");
ABSL_FLAG(std::string, output_dir, "./test_ascir_output", "Output directory");

// Simple HLO module for testing
const char kSimpleHLO[] = R"(
HloModule SimpleAdd

ENTRY main {
  %param0 = f32[4] parameter(0)
  %param1 = f32[4] parameter(1)
  %add = f32[4] add(%param0, %param1)
  %sub = f32[4] subtract(%param0, %param1)
  %mul = f32[4] multiply(%add, %sub)
  ROOT %result = f32[4] add(%add, %mul)
}
)";

// More complex HLO module for testing (from test_simple.hlo)
const char kComplexHLO[] = R"(
HloModule TensorFlowMatMul

ENTRY %main {
  %arg0 = f32[4,4] parameter(0)
  %arg1 = f32[4,4] parameter(1)
  %constant = f32[] constant(0.01)
  %broadcast = f32[4,4] broadcast(%constant), dimensions={}
  %multiply.1 = f32[4,4] multiply(%arg1, %broadcast)
  %dot = f32[4,4] dot(%arg0, %multiply.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %dot.1 = f32[4,4] dot(%dot, %arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

absl::StatusOr<std::unique_ptr<xla::HloModule>> ParseHloString(
    absl::string_view hlo_text,
    const std::string& module_name = "test_module") {
  xla::HloModuleConfig config;
  xla::HloParserOptions options;

  return xla::ParseAndReturnUnverifiedModule(hlo_text, config, options);
}

absl::Status RunTest(absl::string_view hlo_text,
                     const std::string& output_dir) {
  std::cout << "=== HLO to AscIR Backend Test ===" << std::endl;
  std::cout << "\n[Step 1] Parsing HLO text..." << std::endl;

  // Parse HLO text to HloModule
  TF_ASSIGN_OR_RETURN(auto hlo_module, ParseHloString(hlo_text));

  std::cout << "  ✅ HLO module parsed: " << hlo_module->name() << std::endl;
  std::cout << "     Computations: " << hlo_module->computation_count() << std::endl;
  std::cout << "     Entry computation instructions: "
            << hlo_module->entry_computation()->instruction_count() << std::endl;

  std::cout << "\n[Step 2] Configuring AscIR Backend..." << std::endl;

  xla::AscirBackendOptions options;
  options.output_dir = output_dir;
  options.preserve_intermediates = true;
  options.soc_version = "Ascend910B1";

  std::cout << "  Output directory: " << options.output_dir << std::endl;
  std::cout << "  SOC version: " << options.soc_version << std::endl;

  std::cout << "\n[Step 3] Compiling HLO to AscIR Kernel..." << std::endl;

  xla::AscirBackend backend;
  TF_ASSIGN_OR_RETURN(auto result, backend.Compile(*hlo_module, options));

  std::cout << "  ✅ Compilation successful!" << std::endl;

  std::cout << "\n[Step 4] Generated Files:" << std::endl;
  std::cout << "  Total computations compiled: " << result.computation_results.size() << std::endl;

  for (size_t i = 0; i < result.computation_results.size(); ++i) {
    const auto& comp_result = result.computation_results[i];
    std::cout << "\n  Computation " << (i + 1) << ": " << comp_result.computation_name << std::endl;
    std::cout << "    Graph name:      " << comp_result.graph_name << std::endl;
    std::cout << "    ASCIR file:      " << comp_result.ascir_file << std::endl;
    std::cout << "    Host file:       " << comp_result.host_file << std::endl;
    std::cout << "    Device file:     " << comp_result.device_file << std::endl;
    std::cout << "    Tiling data:      " << comp_result.tiling_data_file << std::endl;
    std::cout << "    Tiling size:     " << comp_result.tiling_data.size() << " bytes" << std::endl;
  }

  // Verify files exist
  std::cout << "\n[Step 5] Verifying generated files..." << std::endl;

  int total_files = 0;
  int existing_files = 0;

  for (const auto& comp_result : result.computation_results) {
    total_files += 3;  // host, device, tiling_data
    std::ifstream host_file(comp_result.host_file);
    std::ifstream device_file(comp_result.device_file);

    if (host_file.good()) {
      std::cout << "  ✅ Host file exists for " << comp_result.computation_name << std::endl;
      existing_files++;
    } else {
      std::cout << "  ❌ Host file missing for " << comp_result.computation_name << std::endl;
    }

    if (device_file.good()) {
      std::cout << "  ✅ Device file exists for " << comp_result.computation_name << std::endl;
      existing_files++;
    } else {
      std::cout << "  ❌ Device file missing for " << comp_result.computation_name << std::endl;
    }

    if (!comp_result.tiling_data.empty()) {
      std::cout << "  ✅ Tiling data generated for " << comp_result.computation_name
                << " (" << comp_result.tiling_data.size() << " bytes)" << std::endl;
      existing_files++;
    } else {
      std::cout << "  ❌ Tiling data missing for " << comp_result.computation_name << std::endl;
    }
  }

  std::cout << "\n  Files verified: " << existing_files << "/" << total_files << std::endl;
  std::cout << "\n=== Test Completed Successfully ===" << std::endl;

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::cout << "========================================" << std::endl;
  std::cout << "HLO to AscIR Backend Test" << std::endl;
  std::cout << "========================================" << std::endl;

  std::string hlo_text;
  std::string source_desc;

  // Determine HLO source
  if (!absl::GetFlag(FLAGS_hlo_file).empty()) {
    // Read from file
    std::string hlo_file = absl::GetFlag(FLAGS_hlo_file);
    std::cout << "Reading HLO from file: " << hlo_file << std::endl;

    std::ifstream file(hlo_file);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file: " << hlo_file << std::endl;
      return EXIT_FAILURE;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    hlo_text = buffer.str();
    source_desc = absl::StrCat("file: ", hlo_file);
  } else if (!absl::GetFlag(FLAGS_hlo_text).empty()) {
    // Use provided HLO text
    hlo_text = absl::GetFlag(FLAGS_hlo_text);
    source_desc = "command line text";
  } else {
    // Use default test HLO
    std::cout << "Using default test HLO" << std::endl;
    hlo_text = kComplexHLO;
    source_desc = "built-in test";
  }

  std::cout << "HLO source: " << source_desc << std::endl;
  std::cout << "HLO text length: " << hlo_text.length() << " bytes" << std::endl;
  std::cout << std::endl;

  // Run test
  std::string output_dir = absl::GetFlag(FLAGS_output_dir);

  auto status = RunTest(hlo_text, output_dir);

  if (!status.ok()) {
    std::cerr << "\n❌ Test Failed: " << status.message() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
