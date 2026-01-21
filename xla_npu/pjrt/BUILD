load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_host_memory_spaces",
    srcs = ["npu_host_memory_spaces.cc"],
    hdrs = ["npu_host_memory_spaces.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:fingerprint",
    ],
)