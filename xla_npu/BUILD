load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
)
load(
    "//xla/tsl:tsl.bzl",
    "if_macos",
    "internal_visibility",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

# NPU版本选择配置：NPU_AUTO_FUSE_BACKEND=3时编译V1.0版本，其他值或未定义时编译V2.0版本
config_setting(
    name = "npu_v1_enabled",
    define_values = {"NPU_AUTO_FUSE_BACKEND": "3"},
    visibility = ["//visibility:public"],
)

# 统一的PJRT C API库
cc_library(
    name = "pjrt_c_api_npu",
    visibility = ["//visibility:public"],
    deps = select({
        ":npu_v1_enabled": ["//xla/xla_npu/pjrt_npu/c:pjrt_c_api_npu"],
        "//conditions:default": ["//xla/xla_npu/pjrt/c:pjrt_c_api_npu"],
    }),
)

# 统一的插件：使用genrule复制到统一目录
genrule(
    name = "pjrt_c_api_npu_plugin",
    srcs = select({
        ":npu_v1_enabled": ["//xla/xla_npu/pjrt_npu/c:pjrt_c_api_npu_plugin.so"],
        "//conditions:default": ["//xla/xla_npu/pjrt/c:pjrt_c_api_npu_plugin.so"],
    }),
    outs = ["pjrt_c_api_npu_plugin.so"],
    cmd = "cp $(SRCS) $@",
    visibility = ["//visibility:public"],
)

# 别名：V1.0版本插件（保留旧路径兼容性）
alias(
    name = "pjrt_c_api_npu_plugin_v1.so",
    actual = "//xla/xla_npu/pjrt_npu/c:pjrt_c_api_npu_plugin.so",
    visibility = ["//visibility:public"],
)

# 别名：V2.0版本插件（保留旧路径兼容性）
alias(
    name = "pjrt_c_api_npu_plugin_v2.so",
    actual = "//xla/xla_npu/pjrt/c:pjrt_c_api_npu_plugin.so",
    visibility = ["//visibility:public"],
)
