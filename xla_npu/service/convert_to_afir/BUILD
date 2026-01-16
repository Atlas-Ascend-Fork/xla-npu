# Copyright 2025 The OpenXLA Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# ============================================================================
# HLO to AFIR Converter
# ============================================================================

# Core conversion library
cc_library(
    name = "convert_to_afir_lib",
    srcs = ["convert_to_afir.cc"],
    hdrs = ["convert_to_afir.h"],
    deps = [
        "//xla/xla_npu/third_party:AFIRDialect",
        "//xla/xla_npu/third_party:AFIRHeaders",
        "//xla/hlo/ir:hlo",
        "//xla:util",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

# Test library
cc_test(
    name = "convert_to_afir_test",
    srcs = ["convert_to_afir_test.cc"],
    deps = [
        ":convert_to_afir_lib",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_test_base",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# Binary tool for standalone conversion
cc_binary(
    name = "hlo_to_afir_tool",
    srcs = ["hlo_to_afir_tool.cc"],
    deps = [
        ":convert_to_afir_lib",
        "//xla/xla_npu/third_party:AFIRHeaders",
        "//xla/hlo/ir:hlo",
        "//xla/tools:hlo_module_loader",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
