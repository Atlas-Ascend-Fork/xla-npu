load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_auto_fuse_kernel",
    srcs = ["npu_auto_fuse_kernel.cc"],
    hdrs = ["npu_auto_fuse_kernel.h"],
    deps = [
        "//xla/xla_npu/common:npu_log",
        "//xla/xla_npu/third_party/register:functionLoader",

        "//xla/stream_executor:kernel",
        "//xla/stream_executor:stream_executor_h",
    ],
)