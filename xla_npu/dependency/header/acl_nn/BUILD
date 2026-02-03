load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "acl_nn_hdrs",
    includes = [
        ".",
    ], # 将当前BUILD文件所在目录添加到头文件检索路径中
    hdrs = glob([
        "*.h",
        "*.hpp",
        "*.h.inc",
    ]), # 自动检索当前目录下所有头文件
    deps = [
        "//xla/xla_npu/dependency/header/acl:acl_hdrs",
    ],
)