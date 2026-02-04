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
        "//xla/codegen/emitters:kernel_arguments",
    ],
)

cc_library(
    name = "npu_config",
    srcs = ["npu_config.cc"],
    hdrs = ["npu_config.h"],
    deps = [
        ":npu_constant",
        "//xla/xla_npu/dependency/tool:log",

        "//xla/tsl/util:env_var",
    ],
)

cc_library(
    name = "npu_operation",
    srcs = ["npu_operation.cc"],
    hdrs = ["npu_operation.h"],
    deps = [
        ":npu_constant",
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/tool:log",

        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "npu_tool",
    hdrs = ["npu_tool.h"],
    deps = [
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/interface:acl_nn_interface",
        "//xla/xla_npu/dependency/tool:log",

        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@llvm-project//mlir:IR",
    ],
)