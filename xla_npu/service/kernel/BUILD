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