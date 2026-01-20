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
        "opdev/common_types.h",
        "opdev/bfloat16.h",
        "opdev/fp16_t.h",
        "opdev/fast_vector.h",
        "opdev/pool_allocator.h",
        "opdev/small_vector.h",
        "opdev/object.h",
        "opdev/op_def.h",
    ],
    deps = [
        "//xla/xla_npu/third_party/acl:acl_hdrs",
        "//xla/xla_npu/third_party/metadef:metadef_hdrs",
    ],
)