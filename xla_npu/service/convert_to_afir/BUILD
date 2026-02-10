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

package(default_visibility = ["//xla:internal"])

licenses(["notice"])  # Apache 2.0

# ============================================================================
# HLO to AFIR Converter
# ============================================================================

# Core conversion library
cc_library(
    name = "convert_to_afir",
    srcs = ["convert_to_afir.cc"],
    hdrs = ["convert_to_afir.h"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/xla_npu/dependency/external:AFIRDialect",
        "//xla/xla_npu/dependency/external:AFIRHeaders",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Support",
    ],
)

# AscIR backend library
cc_library(
    name = "ascir_backend",
    srcs = ["ascir_backend.cc"],
    hdrs = ["ascir_backend.h"],
    deps = [
        ":convert_to_afir",
        "//xla/xla_npu/service/kernel:npu_compile_result",

        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/xla_npu/dependency/external:AFIRHeaders",
        "//xla/xla_npu/dependency/tool:log",
        "//xla/xla_npu/service/convert_to_afir/afir_to_ascir_native:AFIRToASCIRTextNative",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
    linkstatic = True, # 不向外部传递依赖
)

# Binary test for AscIR backend
cc_binary(
    name = "ascir_backend_test",
    srcs = ["ascir_backend_test.cc"],
    deps = [
        ":ascir_backend",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

# Binary tool for standalone conversion
cc_binary(
    name = "hlo_to_afir_tool",
    srcs = ["hlo_to_afir_tool.cc"],
    deps = [
        ":convert_to_afir",
        "//xla/hlo/ir:hlo",
        "//xla/tools:hlo_module_loader",
        "//xla/xla_npu/dependency/external:AFIRHeaders",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
