load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "aclnnop_hdrs",
    hdrs = [
        "aclnn_util.h",
        "aclnn_expand.h",
    ],
    deps = [
        "//xla/xla_npu/third_party/aclnn:aclnn_hdrs",
    ],
)