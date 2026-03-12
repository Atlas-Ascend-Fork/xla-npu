# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//learning/brain/mlir:xla_friends"]),
    licenses = ["notice"],
)

td_library(
    name = "td_files",
    srcs = [
        "mair_ops.td",
        "mair_ops_base.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ParallelCombiningOpInterfaceTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "mair_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "mair_ops.h.inc": ["-gen-op-decls"],
        "mair_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mair_ops.td",
    deps = [":td_files"],
)

cc_library(
    name = "air_dialect",
    srcs = [
        "mair_ops.cc",
        "mair_ops.cc.inc",
        "mair_ops.h.inc",
    ],
    hdrs = [
        "mair_ops.h",
    ],
    deps = [
        "mair_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)
