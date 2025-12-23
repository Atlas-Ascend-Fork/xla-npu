load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_binary(
    name = "libge_runner_stub",
    srcs = [
        "graph_stub.cc",
        "ge_api_stub.cc",
    ],
    deps = [
        "//xla/xla_npu/third_party/metadef:metadef_hdrs",
        "//xla/xla_npu/third_party/air:air_hdrs",
    ],
    linkshared = True,
    linkopts = ["-Wl,-soname,libge_runner.so"],
)

cc_import(
    name = "ge_runner_stub",
    shared_library = ":libge_runner_stub",
)

cc_library(
    name = "ge_runner",
    deps = [
        "//xla/xla_npu/third_party/metadef:metadef_hdrs",
        "//xla/xla_npu/third_party/air:air_hdrs",
        ":ge_runner_stub",
    ],
)