load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//learning/brain/mlir:xla_friends"]),
    licenses = ["notice"],
)

cc_library(
    name = "graph_convertor",
    srcs = [
        "export_graphdef.cc",
        "status.cc",
        "convert_type.cc",
        "convert_tensor.cc",
    ],
    hdrs = [
        "export_graphdef.h",
        "status.h",
        "convert_type.h",
        "convert_tensor.h",
        "utils.h",
    ],
    deps = [
        "//xla/xla_npu/mair/ir:air_dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FuncDialect",
        "//xla/xla_npu/third_party/stub:ge_hdrs",
        "//xla/xla_npu/dependency/interface:acl_interface",
    ],
)