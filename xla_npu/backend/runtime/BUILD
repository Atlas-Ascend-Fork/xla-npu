load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_thunk_id",
    hdrs = ["npu_thunk_id.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/tsl/lib/gtl:int_type",
    ],
)