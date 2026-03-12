# Copyright 2025 The OpenXLA Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# ============================================================================
# Ascend-MLIR Headers
# ============================================================================

cc_library(
    name = "AFIRHeaders",
    hdrs = glob([
        "Ascend-MLIR/include/Dialect/AFIR/**/*.h",
        "Ascend-MLIR/include/Dialect/AFIR/**/*.hpp",
        "Ascend-MLIR/include/Utils/*.h",
        "Ascend-MLIR/include/Utils/*.hpp",
        "Ascend-MLIR/include/Interface/*.h",
        "Ascend-MLIR/include/Interface/*.hpp",
        "generated/include/Dialect/AFIR/**/*.inc",
        "generated/include/Interface/*.inc",
        "generated/include/Conversion/*.inc",
        "generated/externals/pyasc/include/ascir/**/*.inc",
    ]),
    includes = [
        "generated/externals/pyasc/include",
        "generated/include",
        "Ascend-MLIR/externals/pyasc/include",
        "Ascend-MLIR/include",
    ],
)

# ============================================================================
# Ascend-MLIR Libraries
# ============================================================================

cc_library(
    name = "MLIRAsc",
    srcs = [
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Attributes.cpp",
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Dialect.cpp",
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/OpInterfaces.cpp",
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Ops.cpp",
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/TypeInterfaces.cpp",
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Types.cpp",
    ],
    hdrs = glob([
        "Ascend-MLIR/externals/pyasc/include/ascir/**/*.h",
        "Ascend-MLIR/externals/pyasc/include/ascir/**/*.hpp",
        "generated/externals/pyasc/include/ascir/**/*.inc",
    ]),
    includes = [
        "generated/externals/pyasc/include",
        "Ascend-MLIR/externals/pyasc/include",
    ],
    deps = [
        ":AFIRHeaders",
        "@llvm-project//mlir:CastInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "AFIRUtils",
    srcs = [
        "Ascend-MLIR/lib/Utils/AFIRUtils.cpp",
    ],
    hdrs = glob([
        "Ascend-MLIR/include/Utils/*.h",
        "Ascend-MLIR/include/Utils/*.hpp",
    ]),
    includes = [
        "generated/externals/pyasc/include",
        "generated/include",
        "Ascend-MLIR/externals/pyasc/include",
        "Ascend-MLIR/include",
    ],
    deps = [
        ":AFIRHeaders",
        ":MLIRAsc",
        "@llvm-project//mlir:CastInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "AFIRDialect",
    srcs = [
        "Ascend-MLIR/lib/Dialect/AFIR/Dialect.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv/concat.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv/reduce.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/broadcast.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/cast.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/condition.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/data.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/math.cpp",
    ],
    hdrs = glob([
        "generated/include/Dialect/AFIR/*.inc",
        "generated/include/Interface/*.inc",
    ]),
    includes = [
        "generated/externals/pyasc/include",
        "generated/include",
        "Ascend-MLIR/externals/pyasc/include",
        "Ascend-MLIR/include",
    ],
    deps = [
        ":AFIRHeaders",
        ":AFIRUtils",
        ":MLIRAsc",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "AFIRTransforms",
    srcs = [
        "Ascend-MLIR/lib/Dialect/AFIR/Transforms/AFIRAddAxis.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Transforms/AFIRInferShapes.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Transforms/Canonicalize.cpp",
    ],
    hdrs = glob([
        "generated/include/Dialect/AFIR/Transforms/*.inc",
    ]),
    includes = [
        "generated/include",
        "Ascend-MLIR/include",
    ],
    deps = [
        ":AFIRDialect",
        ":AFIRHeaders",
        ":AFIRUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
