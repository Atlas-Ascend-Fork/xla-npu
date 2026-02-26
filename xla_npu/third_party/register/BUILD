load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "functionLoader",
    srcs = ["functionLoader.cc"],
    hdrs = ["functionLoader.h"],
    deps = [
    ],
)

cc_library(
    name = "acl_interface",
    srcs = ["aclInterface.cc"],
    hdrs = ["aclInterface.h"],
    deps = [
        ":functionLoader",
        "//xla/xla_npu/third_party/acl:acl_hdrs",
    ],
)