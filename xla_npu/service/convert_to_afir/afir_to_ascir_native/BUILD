# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# ============================================================================
# AFIR to ASCIR Text Conversion (Native - using XLA's MLIR)
# ============================================================================

# Generate Passes.h.inc from Passes.td using mlir-tblgen
gentbl_cc_library(
    name = "passes_inc_gen",
    tbl_outs = {
        "Conversion/Passes.h.inc": [
            "-gen-pass-decls",
            "-name=AFIRToASCIR",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    visibility = ["//visibility:private"],
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

# AFIR to ASCIR Text conversion library (compiled with XLA's MLIR)
cc_library(
    name = "AFIRToASCIRTextNative",
    srcs = ["AFIRToASCIRText.cpp"],
    hdrs = [
        "AFIRToASCIRText.h",
    ],
    includes = [".."],
    deps = [
        ":passes_inc_gen",
        "//xla/xla_npu/common:npu_config",
        "//xla/xla_npu/dependency/external/afir:AFIRDialect",
        "//xla/xla_npu/dependency/external/afir:AFIRHeaders",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CastInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

# Test binary for AFIR to ASCIR conversion (native)
cc_binary(
    name = "afir_to_ascir_native_test",
    srcs = ["afir_to_ascir_native_test.cc"],
    deps = [
        ":AFIRToASCIRTextNative",
        "//xla/xla_npu/dependency/external/afir:AFIRDialect",
        "//xla/xla_npu/dependency/external/afir:AFIRHeaders",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
