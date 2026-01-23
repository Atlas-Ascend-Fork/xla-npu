load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_constant",
    srcs = ["npu_constant.cc"],
    hdrs = ["npu_constant.h"],
    deps = [
    ],
)

cc_library(
    name = "npu_log",
    hdrs = ["npu_log.h"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "npu_operation",
    srcs = ["npu_operation.cc"],
    hdrs = ["npu_operation.h"],
    deps = [
        ":npu_constant",
        ":npu_log",
        "//xla/xla_npu/third_party/register:acl_interface",

        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "npu_tool",
    hdrs = ["npu_tool.h"],
    deps = [
        ":npu_log",
        "//xla/xla_npu/third_party/register:acl_interface",
        "//xla/xla_npu/third_party/register:aclnn_interface",

        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@llvm-project//mlir:IR",
    ],
)