load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//xla:internal"],
)

# NPU Fusion Pass 集合库
cc_library(
    name = "npu_fusion_passes",
    deps = [
        ":npu_fusion_pipeline",
        ":npu_priority_fusion",
        ":npu_multi_output_fusion",
        ":npu_copy_fusion",
        ":npu_dynamic_slice_fusion_rewriter",
        ":npu_fusion_dynamic_memcpy_rewriter",
    ],
)

# NpuFusionPipeline
cc_library(
    name = "npu_fusion_pipeline",
    srcs = ["npu_fusion_pipeline.cc"],
    hdrs = ["npu_fusion_pipeline.h"],
    deps = [
        "//xla/service/gpu/transforms:variadic_op_splitter",
        "//xla/service/gpu/transforms:rename_fusions",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/analysis:hlo_dfs_reachability",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:cpu_gpu_shape_verifier",
        "//xla/service:hlo_verifier",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_cse",
        "//xla/service:layout_assignment",
        "//xla/service:compilation_stats",
        ":npu_priority_fusion",
        ":npu_multi_output_fusion",
        ":npu_copy_fusion",
        ":npu_dynamic_slice_fusion_rewriter",
        ":npu_fusion_dynamic_memcpy_rewriter",
        "@com_google_absl//absl/status",
    ],
)

# NpuPriorityFusion
cc_library(
    name = "npu_priority_fusion",
    srcs = ["npu_priority_fusion.cc"],
    hdrs = ["npu_priority_fusion.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:hlo_cost_analysis",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status",
    ],
)

# NpuMultiOutputFusion
cc_library(
    name = "npu_multi_output_fusion",
    srcs = ["npu_multi_output_fusion.cc"],
    hdrs = ["npu_multi_output_fusion.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/analysis:hlo_dfs_reachability",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:hlo_cost_analysis",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status",
    ],
)

# NpuCopyFusion
cc_library(
    name = "npu_copy_fusion",
    srcs = ["npu_copy_fusion.cc"],
    hdrs = ["npu_copy_fusion.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:call_graph",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status",
    ],
)

# NpuDynamicSliceFusionRewriter
cc_library(
    name = "npu_dynamic_slice_fusion_rewriter",
    srcs = ["npu_dynamic_slice_fusion_rewriter.cc"],
    hdrs = ["npu_dynamic_slice_fusion_rewriter.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla:shape_util",
        "@com_google_absl//absl/status",
    ],
)

# NpuFusionDynamicMemcpyRewriter
cc_library(
    name = "npu_fusion_dynamic_memcpy_rewriter",
    srcs = ["npu_fusion_dynamic_memcpy_rewriter.cc"],
    hdrs = ["npu_fusion_dynamic_memcpy_rewriter.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/status",
    ],
)

# 原有的 npu_super_fusion 保留
cc_library(
    name = "npu_super_fusion",
    srcs = ["npu_super_fusion.cc"],
    hdrs = ["npu_super_fusion.h"],
    deps = [
        "@com_google_absl//absl/log:log",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla:status_macros",
    ],
)
