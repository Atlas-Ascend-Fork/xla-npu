package(default_visibility = ["//visibility:public"])

# 提供对外库stablehlo2mair
cc_library(
    name = "stablehlo2mair",
    srcs = ["stablehlo2mair_converter.cpp"],
    hdrs = ["stablehlo2mair_converter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/xla_npu/mair/ir:air_dialect",
        "//xla/xla_npu/mair/transforms:mair_passes",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

# 在现有的BUILD文件中添加可执行目标
cc_binary(
    name = "demo",
    srcs = [
        "stablehlo2ge.cpp",
    ],
    deps = [
        "//xla/xla_npu/mair/ir:air_dialect",
        "//xla/xla_npu/mair/transforms:mair_passes",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:RegisterAllPasses",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
)
