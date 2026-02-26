load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "npu_device_description",
    srcs = ["npu_device_description.cc"],
    hdrs = ["npu_device_description.h"],
    deps = [
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_device_description",
        "//xla/xla_npu/pjrt_npu:pjrt_common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "npu_topology_description",
    srcs = ["npu_topology_description.cc"],
    hdrs = ["npu_topology_description.h"],
    deps = [
        ":npu_device_description",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_device_dimensions",
        "//xla/tsl/lib/strings:proto_serialization",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)