load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "npuir_backend",
    hdrs = [
        "npuir_backend.h",
    ],
    srcs = [
        "npuir_backend.cc",
    ],
    deps = [
        "//xla/xla_npu/dependency/external/npuir:mlir-headers-hfusion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:CAPIArith",
        "@llvm-project//mlir:LinalgDialect",
    ],
)