# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_compile_result",
    hdrs = ["npu_compile_result.h"],
    deps = [
    ],
)

cc_library(
    name = "npu_kernel_config",
    hdrs = ["npu_kernel_config.h"],
    deps = [
        ":npu_compile_result",
    ],
)

cc_library(
    name = "npu_custom_kernel",
    srcs = ["npu_custom_kernel.cc"],
    hdrs = ["npu_custom_kernel.h"],
    deps = [
        ":npu_kernel_config",

        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)