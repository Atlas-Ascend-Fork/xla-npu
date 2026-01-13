load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/xnnpack:build_defs.bzl", "if_ynnpack")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_thunk",
    srcs = ["npu_thunk.cc"],
    hdrs = ["npu_thunk.h"],
    defines = if_ynnpack(["XLA_YNNPACK"]),
    deps = [
        "//xla/backends/cpu/runtime:buffer_allocations",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/backends/cpu/runtime:xfeed_manager",

        "//xla:executable_run_options",
        "//xla/backends/cpu/collectives:cpu_collectives",
        "//xla/backends/cpu/collectives:in_process_collectives",
        "//xla/backends/cpu/runtime/xnnpack:xnn_interop",
        "//xla/backends/cpu/runtime/xnnpack:xnn_threadpool",
        "//xla/ffi:execution_context",
        "//xla/runtime:buffer_use",
        "//xla/runtime:resource_use",
        "//xla/service:global_device_id",
        "//xla/service/cpu:cpu_executable_run_options",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
    ] + if_ynnpack([
        "//xla/backends/cpu/runtime/ynnpack:ynn_interop",
        "//xla/backends/cpu/runtime/ynnpack:ynn_threadpool",
    ]),
)

cc_library(
    name = "npu_kernel_thunk",
    srcs = ["npu_kernel_thunk.cc"],
    hdrs = ["npu_kernel_thunk.h"],
    deps = [
        ":npu_thunk",
        "//xla/xla_npu/pjrt_npu:npu_tool",
        "//xla/xla_npu/third_party/register:aclnn_interface",

        "//xla/backends/cpu/runtime:buffer_allocations",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/backends/cpu/runtime:kernel",
        "//xla/backends/cpu/runtime:kernel_c_api",

        "//xla:util",
        "//xla/codegen:kernel_spec",
        "//xla/runtime:buffer_use",
        "//xla/runtime:work_group",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)