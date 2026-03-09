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