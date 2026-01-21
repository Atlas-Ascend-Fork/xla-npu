load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_local_client",
    hdrs = ["npu_local_client.h"],
    deps = [
        "//xla/xla_npu/stream_executor:npu_platform",
        "//xla/xla_npu/stream_executor:npu_platform_id",
        "//xla/xla_npu/service:npu_compiler",

        "//xla/client:client_library",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
    ],
)