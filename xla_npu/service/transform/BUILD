# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//xla:internal"],
)

# NPU Transform Pass 集合库
cc_library(
    name = "npu_transform_passes",
    deps = [
        ":npu_pipeline",
        ":npu_fusion_passes",
    ],
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
        ":npu_gemm_fusion",
        ":npu_custom_kernel_fusion_rewriter",
        ":npu_nest_gemm_fusion",
        ":npu_fusion_block_level_rewriter",
        ":npu_gemm_fusion_swap_operands",
        ":npu_fusion_dispatch_pipeline",
        ":npu_hlo_printer_pass",
    ],
)

# NpuPipeline - Main entry point for all NPU optimization passes
cc_library(
    name = "npu_pipeline",
    srcs = ["npu_pipeline.cc"],
    hdrs = ["npu_pipeline.h"],
    copts = select({
        "//xla/xla_npu/common:npu_auto_fuse_backend_afir": ["-DNPU_AUTO_FUSE_BACKEND=1"],
        "//xla/xla_npu/common:npu_auto_fuse_backend_npuir": ["-DNPU_AUTO_FUSE_BACKEND=2"],
        "//xla/xla_npu/common:npu_ge_backend": ["-DNPU_AUTO_FUSE_BACKEND=3"],
        "//conditions:default": ["-DNPU_AUTO_FUSE_BACKEND=0"],
    }),
    deps = [
        ":npu_fusion_pipeline",
        ":npu_hlo_printer_pass",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/simplifiers:flatten_call_graph",
        "//xla/service:cpu_gpu_shape_verifier",
        "//xla/service:hlo_verifier",
        "//xla/service:layout_assignment",
        "@com_google_absl//absl/status",
    ],
)

# NpuFusionPipeline
cc_library(
    name = "npu_fusion_pipeline",
    srcs = ["npu_fusion_pipeline.cc"],
    hdrs = ["npu_fusion_pipeline.h"],
    deps = [
        ":npu_priority_fusion",
        ":npu_multi_output_fusion",
        ":npu_copy_fusion",
        ":npu_dynamic_slice_fusion_rewriter",
        ":npu_fusion_dynamic_memcpy_rewriter",
        ":npu_fusion_wrapper",
        ":npu_rename_fusions",
        ":npu_hlo_printer_pass",
        ":npu_gemm_fusion",
        ":npu_custom_kernel_fusion_rewriter",
        ":npu_fusion_block_level_rewriter",
        ":npu_nest_gemm_fusion",
        ":npu_gemm_fusion_swap_operands",
        ":npu_fusion_dispatch_pipeline",
        "//xla/xla_npu/dependency/tool:log",

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
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:IR",
    ],
)

# NpuHloPrinterPass
cc_library(
    name = "npu_hlo_printer_pass",
    srcs = ["npu_hlo_printer_pass.cc"],
    hdrs = ["npu_hlo_printer_pass.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/xla_npu/common:npu_tool",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status",
    ],
)

# NpuFusionWhitelist
cc_library(
    name = "npu_fusion_whitelist",
    hdrs = ["npu_fusion_whitelist.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

# NpuPriorityFusion
cc_library(
    name = "npu_priority_fusion",
    srcs = ["npu_priority_fusion.cc"],
    hdrs = ["npu_priority_fusion.h"],
    deps = [
        ":npu_fusion_whitelist",
        "//xla/xla_npu/dependency/tool:log",

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
        ":npu_fusion_whitelist",
        "//xla/xla_npu/dependency/tool:log",

        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/analysis:hlo_dfs_reachability",
        "//xla/hlo/analysis:hlo_reachability",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:fusion_queue",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:instruction_fusion",
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
        "//xla/xla_npu/dependency/tool:log",

        "//xla/codegen:ir_emission_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:call_graph",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:errors",
    ],
)

# NpuDynamicSliceFusionRewriter
cc_library(
    name = "npu_dynamic_slice_fusion_rewriter",
    srcs = ["npu_dynamic_slice_fusion_rewriter.cc"],
    hdrs = ["npu_dynamic_slice_fusion_rewriter.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",

        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:call_graph",
        "//xla:shape_util",
        "//xla:util",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuFusionDynamicMemcpyRewriter
cc_library(
    name = "npu_fusion_dynamic_memcpy_rewriter",
    srcs = ["npu_fusion_dynamic_memcpy_rewriter.cc"],
    hdrs = ["npu_fusion_dynamic_memcpy_rewriter.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",

        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/xla_npu/service:backend_config_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuFusionWrapper
cc_library(
    name = "npu_fusion_wrapper",
    srcs = ["npu_fusion_wrapper.cc"],
    hdrs = ["npu_fusion_wrapper.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",

        "//xla/codegen/emitters:fusion_wrapper_base",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/strings",
    ],
)

# NpuRenameFusions
cc_library(
    name = "npu_rename_fusions",
    srcs = ["npu_rename_fusions.cc"],
    hdrs = ["npu_rename_fusions.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",

        "//xla/codegen:ir_emission_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuGemmFusion
cc_library(
    name = "npu_gemm_fusion",
    srcs = ["npu_gemm_fusion.cc"],
    hdrs = ["npu_gemm_fusion.h"],
    deps = [
        ":npu_fusion_whitelist",

        "//xla/xla_npu/dependency/tool:log",
        "//xla/xla_npu/service:backend_config_cc",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:backend_config",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:instruction_fusion",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:errors",
        "@llvm-project//mlir:IR",
    ],
)

# NpuCustomKernelFusionRewriter
cc_library(
    name = "npu_custom_kernel_fusion_rewriter",
    srcs = ["npu_custom_kernel_fusion_rewriter.cc"],
    hdrs = ["npu_custom_kernel_fusion_rewriter.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",
        "//xla/xla_npu/service:backend_config_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:backend_config",
        "//xla/hlo/pass:hlo_pass",
        "//xla:shape_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:errors",
    ],
)

# NpuNestGemmFusion
cc_library(
    name = "npu_nest_gemm_fusion",
    srcs = ["npu_nest_gemm_fusion.cc"],
    hdrs = ["npu_nest_gemm_fusion.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",
        "//xla/xla_npu/service:backend_config_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:backend_config",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/hlo/pass:hlo_pass",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:errors",
    ],
)

# NpuFusionBlockLevelRewriter
cc_library(
    name = "npu_fusion_block_level_rewriter",
    srcs = ["npu_fusion_block_level_rewriter.cc"],
    hdrs = ["npu_fusion_block_level_rewriter.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",
        "//xla/xla_npu/service:backend_config_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:backend_config",
        "//xla/hlo/pass:hlo_pass",
        "//xla:shape_util",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:errors",
    ],
)

# NpuGemmFusionSwapOperands
cc_library(
    name = "npu_gemm_fusion_swap_operands",
    srcs = ["npu_gemm_fusion_swap_operands.cc"],
    hdrs = ["npu_gemm_fusion_swap_operands.h"],
    deps = [
        "//xla/xla_npu/service:backend_config_cc",
        "//xla/xla_npu/dependency/tool:log",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla:shape_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:errors",
    ],
)

# NpuFusionDispatchPipeline
cc_library(
    name = "npu_fusion_dispatch_pipeline",
    srcs = ["npu_fusion_dispatch_pipeline.cc"],
    hdrs = ["npu_fusion_dispatch_pipeline.h"],
    deps = [
        ":npu_custom_kernel_fusion_rewriter",
        ":npu_fusion_block_level_rewriter",
        ":npu_fusion_dynamic_memcpy_rewriter",
        ":npu_fusion_whitelist",
        ":npu_gemm_fusion",
        ":npu_gemm_fusion_swap_operands",
        ":npu_nest_gemm_fusion",
        "//xla/xla_npu/dependency/tool:log",
        "//xla/xla_npu/service:backend_config_cc",

        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_cse",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:status",
        "@llvm-project//mlir:IR",
    ],
)
