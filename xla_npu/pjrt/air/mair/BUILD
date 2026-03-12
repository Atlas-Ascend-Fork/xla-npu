# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

package(default_visibility = ["//visibility:public"])

# 提供对外库stablehlo2mair
cc_library(
    name = "stablehlo2mair",
    srcs = ["stablehlo2mair_converter.cpp"],
    hdrs = ["stablehlo2mair_converter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/xla_npu/pjrt/air/mair/ir:air_dialect",
        "//xla/xla_npu/pjrt/air/mair/transforms:mair_passes",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

# 在现有的BUILD文件中添加可执行目标
cc_binary(
    name = "demo",
    srcs = [
        "stablehlo2ge.cpp",
    ],
    deps = [
        "//xla/xla_npu/pjrt/air/mair/ir:air_dialect",
        "//xla/xla_npu/pjrt/air/mair/transforms:mair_passes",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:RegisterAllPasses",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
)
