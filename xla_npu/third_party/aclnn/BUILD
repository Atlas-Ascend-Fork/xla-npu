load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "aclnn_hdrs",
    hdrs = [
        "acl_meta.h",
        "aclnn_base.h",
    ],
    deps = [
        "//xla/xla_npu/third_party/acl:acl_hdrs",
    ],
)