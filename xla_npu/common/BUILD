load("//xla/tsl:tsl.bzl", "internal_visibility")
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
        "//xla/xla_npu/third_party/register:aclnn_interface",

        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@llvm-project//mlir:IR",
    ],
)