load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//learning/brain/mlir:xla_friends"]),
    licenses = ["notice"],
)

cc_binary(
    name = "test_mair_gen",
    deps = [
        "//xla/xla_npu/mair/test:test_mair_gen_lib",
    ],
)

cc_library(
    name = "test_mair_gen_lib",
    srcs = [
        "test_mair_gen.cc",
    ],
    hdrs = [
        "test_mair_gen.h",
    ],
    deps = [
        "//xla/xla_npu/mair/ir:air_dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:CAPIArith",
        "//xla/xla_npu/mair/translations:graph_convertor",
        "//xla/xla_npu/third_party/stub:ge_hdrs",
        "//xla/xla_npu/third_party/register:acl_interface",
    ],
)
