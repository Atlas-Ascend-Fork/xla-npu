# 使用本地 LLVM 构建
set(LLVM_BUILD_DIR "$ENV{LLVM_BUILD_DIR}" CACHE PATH "Path to LLVM build directory")
set(LLVM_SOURCE_DIR "$ENV{LLVM_SOURCE_DIR}" CACHE PATH "Path to LLVM source directory")
set(MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(EXISTS "${LLVM_BUILD_DIR}/lib/cmake/mlir")
    list(APPEND CMAKE_PREFIX_PATH "${LLVM_BUILD_DIR}")
    message(STATUS "Using MLIR from build dir: ${LLVM_BUILD_DIR}")
else()
    message(FATAL_ERROR "MLIR not found. Please set LLVM_BUILD_DIR or enable USE_SYSTEM_MLIR")
endif()

# 包含头文件和源文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ir)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/transforms)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/stablehlo)
include_directories(${LLVM_SOURCE_DIR}/llvm/include)
include_directories(${LLVM_SOURCE_DIR}/mlir/include)
include_directories(${LLVM_BUILD_DIR}/include)
include_directories(${LLVM_BUILD_DIR}/tools/mlir/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/ir)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/transforms)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/stablehlo)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/translations)

# 查找 MLIR（会自动关联查找 LLVM，无需单独 find_package(LLVM)）
find_package(MLIR REQUIRED CONFIG)
message(STATUS "found MLIR：${MLIR_DIR}")
message(STATUS "found LLVM：${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")

# 引入 MLIR/LLVM 的构建辅助脚本（find_package 后自动设置路径，无需手动指定）
include(AddMLIR)
include(AddLLVM)

add_subdirectory(ir)
add_subdirectory(transforms)
add_subdirectory(test)
add_subdirectory(translations)

# 添加 StableHLO目录
set(STABLEHLO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stablehlo")
add_subdirectory(${STABLEHLO_SOURCE_DIR} EXCLUDE_FROM_ALL)

macro(add_compiler_executeable name)
    add_llvm_example(${name} ${ARGN})
endmacro(add_compiler_executeable name)

add_compiler_executeable(demo
        stablehlo2ge.cpp
        ir/mair_ops.cc
        transforms/mair_passes.cc

        DEPENDS
        MLIRmair_opsIncGen
        MLIRMairPassIncGen
)

# 链接MLIR/LLVM/StableHLO的库
target_link_libraries(demo
        PRIVATE
        MLIRAnalysis
        MLIRBuiltinToLLVMIRTranslation
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRFunctionInterfaces
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRRegisterAllDialects
        MLIRRegisterAllExtensions
        MLIRRegisterAllPasses
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        LLVMCore
        LLVMSupport
        StablehloOps
)
