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
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_tool",
    hdrs = ["npu_tool.h"],
    deps = [
        "//xla/xla_npu/pjrt/air/npu/ge:graph_data_hdrs",
        "//xla/xla_npu/dependency/tool:log",
        "//xla:shape_util",
    ],
)

cc_library(
    name = "pjrt_common",
    hdrs = ["pjrt_common.h"],
    deps = [
        "//xla/pjrt/proto:pjrt_value_type_proto_cc",
        "//xla/tsl/lib/gtl:int_type",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "host_memory_spaces",
    srcs = ["host_memory_spaces.cc"],
    hdrs = ["host_memory_spaces.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:fingerprint",
    ],
)