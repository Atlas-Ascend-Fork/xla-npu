load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//xla:internal"],
)

cc_library(
    name = "npu_super_fusion",
    srcs = ["npu_super_fusion.cc"],
    hdrs = ["npu_super_fusion.h"],
    deps = [
        "@com_google_absl//absl/log:log",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla:status_macros",
    ],
)