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

cc_library(
    name = "aclnn_interface",
    srcs = ["aclnnInterface.cc"],
    hdrs = ["aclnnInterface.h"],
    deps = [
        ":functionLoader",
        "//xla/xla_npu/third_party/aclnn:aclnn_hdrs",
    ],
)

cc_library(
    name = "aclnnop_interface",
    srcs = ["aclnnopInterface.cc"],
    hdrs = ["aclnnopInterface.h"],
    deps = [
        ":functionLoader",
        "//xla/xla_npu/third_party/aclnn:aclnn_hdrs",
    ],
)