load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//learning/brain/mlir:xla_friends"]),
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "mair_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"mair_passes.h.inc": [
        "-gen-pass-decls",
        "-name=Mair",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mair_passes.td",
    deps = [
        "//xla/xla_npu/xla_npu/mair/ir:td_files",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "mair_passes",
    srcs = ["mair_passes.cc"],
    hdrs = ["mair_passes.h"],
    deps = [
        ":mair_passes_inc_gen",
        "//xla/xla_npu/xla_npu/mair/ir:air_dialect",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@stablehlo//:stablehlo_ops",
    ],
)
