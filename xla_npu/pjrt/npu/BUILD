load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"], # 定义为xla内部完全可见，与参照的cpu的定义有所区别，cpu和gpu严格定义了这里实现能被哪些代码引用
    licenses = ["notice"],
)

cc_library(
    name = "npu_pjrt_stream_executor_client",
    srcs = ["npu_pjrt_stream_executor_client.cc"],
    hdrs = ["npu_pjrt_stream_executor_client.h"],
    deps = [
        "//xla/xla_npu/pjrt/plugin:npu_topology_description",

        "//xla/pjrt:pjrt_stream_executor_client",
    ],
)
