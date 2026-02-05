# XLA-NPU

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.8+-orange.svg)](https://github.com/google/jax)

## 简介

**XLA-NPU** 是一个面向华为昇腾NPU（Neural Processing Unit）硬件的 **XLA（Accelerated Linear Algebra）后端实现**。本项目将 OpenXLA/JAX 与华为 CANN（Compute Architecture for Neural Networks）工具链集成，使机器学习工作负载能够在华为昇腾 NPU 硬件上高效运行。

### 核心功能

- **JAX 后端支持**：通过 PJRT C API 插件，为 JAX 提供对华为昇腾 NPU 的完整支持
- **XLA 编译器集成**：实现完整的 XLA 编译器接口，支持 HLO 到 NPU 代码的编译
- **MAIR 方言**：基于 MLIR 的自定义方言，用于表示昇腾操作
- **双路径编译**：支持 CPU 风格和 NPU 原生两种编译路径
- **ACLNN 集成**：封装 100+ 个昇腾神经网络操作

### 架构概览

```
JAX 程序
    ↓
PJRT C API Plugin (pjrt_c_api_npu_plugin.so)
    ↓
XLA Compiler (HLO → MAIR → ASCIR)
    ↓
NPU Runtime (Thunk Execution)
    ↓
华为 CANN / ACLNN 操作
```

### 项目结构

```
xla-npu/
├── xla_npu/                    # 主实现目录
│   ├── backend/               # 后端编译和运行时
│   │   ├── codegen/          # NPU 操作代码生成
│   │   └── runtime/          # 运行时 thunk 执行
│   ├── client/               # 客户端实现
│   ├── common/               # 通用工具和配置
│   ├── dependency/           # 外部依赖
│   │   ├── external/         # Ascend-MLIR 子模块
│   │   ├── header/           # ACL 头文件
│   │   ├── interface/        # ACL 封装接口
│   │   └── tool/             # 工具（日志、函数加载等）
│   ├── mair/                 # MAIR 方言及转换
│   ├── pjrt/                # PJRT C API 插件
│   ├── service/             # XLA 编译器服务
│   └── stream_executor/     # NPU 流执行器
├── build/                   # 构建脚本和基础设施
├── demo/                    # 示例程序
├── README.md               # 基础文档
└── INSTALL_GUIDE.md        # 详细安装指南
```

---

## 安装与卸载

详细的安装和卸载指南请参考：**[INSTALL_GUIDE.md](INSTALL_GUIDE.md)**

### 快速安装

```bash
# 克隆代码仓库
git clone https://gitcode.com/cann/xla-npu.git
cd xla-npu

# 运行自动安装脚本
./build/build.sh

# 设置环境变量
source build/xla_npu_env
```

### 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: >= 3.11
- **磁盘空间**: 至少 10GB
- **网络**: 需要访问 github.com 和 storage.googleapis.com

### 卸载

```bash
# 使用脚本卸载
./build/build.sh uninstall

# 或手动删除
rm -rf build/
```

---

## 快速上手

### 1. 在 JAX 中使用 NPU 后端

```python
import os
import jax
import jax.numpy as jnp
import numpy as np

# 设置 PJRT 插件路径
PLATFORM_NAME = 'npu'
LIBRARY_PATH = '/path/to/pjrt_c_api_npu_plugin.so'
os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = f"{PLATFORM_NAME}:{LIBRARY_PATH}"
os.environ['JAX_PLATFORMS'] = PLATFORM_NAME

# 触发后端发现
import jax.lib.xla_client

# 检查设备
print(f"JAX Default Backend: {jax.default_backend()}")
print(f"Available Devices: {jax.devices()}")

# 运行计算
x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
func = jax.jit(lambda x: jax.nn.gelu(x))
y = func(x)
result = y.block_until_ready()
print(f"Result: {result}")
```

### 2. MAIR 方言转换

**使用 CMake 构建**：

```bash
cd xla_npu
cmake -G Ninja .. \
    -DLLVM_BUILD_DIR="/path/to/llvm/build" \
    -DLLVM_SOURCE_DIR="/path/to/llvm/source"
cmake --build . --target demo
```

**使用 Bazel 构建**：

```bash
bazel build --incompatible_disallow_empty_glob=false \
    --remote_download_minimal \
    --spawn_strategy=local \
    --copt=-w //xla/xla_npu/mair:demo
```

**运行 StableHLO 转 MAIR demo**：

```bash
echo 'module {
  func.func @simple_add(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %res = stablehlo.add %a, %b : tensor<2x3xf32>
    return %res : tensor<2x3xf32>
  }
}' | ./demo -emit=mair
```

**输出结果**：

```
module {
  func.func @simple_add(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "mair.Add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
```

### 3. HLO 到 AFIR 转换工具

```bash
# 构建 HLO 到 AFIR 转换工具
bazel build //xla/xla_npu/service/convert_to_afir:hlo_to_afir_tool

# 运行转换（需要设置环境变量）
export ASCEND_MLIR_PYTHON_PATH=/path/to/Ascend-MLIR/python
./bazel-bin/xla/xla_npu/service/convert_to_afir/hlo_to_afir_tool \
    --hlo_file=input.hlo \
    --output_dir=./output
```

---

## 特性介绍

### 1. 完整的 XLA 后端实现

XLA-NPU 实现了完整的 XLA 编译器接口，包括：

- **HLO 编译**：将 XLA HLO IR 转换为 NPU 可执行代码
- **内存管理**：支持 NPU 设备内存分配和管理
- **流执行**：异步执行和事件同步
- **Thunk 运行时**：基于 thunk 的执行模型

**关键文件**：
- `xla_npu/backend/` - 后端编译和运行时实现
- `xla_npu/service/` - XLA 编译器服务
- `xla_npu/stream_executor/` - 流执行器平台

### 2. PJRT C API 插件

通过 PJRT C API 插件机制，XLA-NPU 能够作为 JAX 的后端运行：

- **标准 PJRT API**：实现完整的 PJRT C API 接口
- **插件式架构**：通过环境变量动态加载
- **设备管理**：支持多 NPU 设备管理

**关键文件**：
- `xla_npu/pjrt/c/pjrt_c_api_npu.cc` - PJRT C API 入口点
- `xla_npu/pjrt/npu/` - NPU 特定 PJRT 客户端实现

**生成的插件**：`pjrt_c_api_npu_plugin.so`

### 3. MAIR 方言（MLIR-based AIR）

MAIR 是基于 MLIR 的自定义方言，用于表示昇腾 NPU 操作：

- **方言定义**：完整的 MLIR 方言定义
- **StableHLO 转换**：支持从 StableHLO 到 MAIR 的转换
- **优化 Pass**：MLIR 优化和转换 Pass

**关键文件**：
- `xla_npu/mair/ir/mair_ops.h` - MAIR 方言定义
- `xla_npu/mair/translations/` - 转换工具
- `xla_npu/mair/transforms/` - MLIR 转换 Pass

### 4. 双路径编译策略

XLA-NPU 支持两种编译路径，满足不同场景需求：

#### CPU Compiler Type 路径（当前使用）
- 生成 CPU 风格的 LLVM IR
- 使用基于 thunk 的执行模型
- 实现简单，兼容性好

#### NPU Compiler Type 路径（高级路径）
- 真正的 NPU 感知编译
- LLVM IR 为 NPU 生成
- 支持 fusion 优化和自定义内核

**关键文件**：
- `cpu_compiler_type.npu_*` - CPU 路径实现
- `npu_compiler.cc` - NPU 路径实现
- `npu_ir_emitter_unnested.cc` - NPU IR 生成

### 5. AFIR 转换系统

完整的 HLO → AFIR → ASCIR 转换管道：

- **HLO 到 AFIR**：XLA HLO 转换为 Ascend Frontend IR
- **AFIR 到 ASCIR**：AFIR 转换为 Ascend IR
- **原生转换**：支持原生和基于 Python 的转换

**关键文件**：
- `xla_npu/service/convert_to_afir/convert_to_afir.cc` - 核心转换逻辑
- `xla_npu/service/convert_to_afir/ascir_backend.cc` - ASCIR 后端集成
- `xla_npu/service/convert_to_afir/afir_to_ascir_native/` - 原生转换实现

**构建目标**：
- `afir_to_ascir_native` - AFIR 到 ASCIR 转换库
- `ascir_backend_test` - 后端测试工具
- `hlo_to_afir_tool` - HLO 到 AFIR 转换工具

### 6. ACLNN 操作集成

封装了 100+ 个华为昇腾神经网络操作（ACLNN）：

- **操作封装**：ACLNN 操作的 C++ 封装
- **Thunk 执行**：基于 thunk 的操作执行
- **类型支持**：支持各种张量类型和形状

**关键文件**：
- `xla_npu/dependency/interface/` - ACL 接口封装
- `xla_npu/backend/runtime/npu_aclnn_thunk.cc` - ACLNN thunk 实现

### 7. 运行时 Thunk 系统

基于 thunk 的运行时执行系统：

- **顺序执行**：`npu_sequential_thunk` - 顺序执行多个操作
- **自定义内核**：`npu_custom_kernel_thunk` - 自定义内核执行
- **ACLNN 操作**：`npu_aclnn_thunk` - ACLNN 操作执行
- **基类**：`npu_thunk` - 所有 thunk 的基类

**关键文件**：
- `xla_npu/backend/runtime/thunk.h` - Thunk 基类定义
- `xla_npu/backend/runtime/npu_sequential_thunk.cc` - 顺序执行
- `xla_npu/backend/runtime/npu_aclnn_thunk.cc` - ACLNN 操作

### 8. 动态配置系统

运行时可配置的系统：

- **单例模式**：全局配置管理
- **字符串和布尔配置**：支持多种配置类型
- **自动融合开关**：支持操作自动融合配置

**配置选项**：
- `NPU_AUTO_FUSE_SWITCH` - 自动融合开关
- `NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR` - 编译产物根目录

**关键文件**：
- `xla_npu/common/npu_config.h` - 配置系统实现

### 9. 自动化构建系统

完整的自动化构建脚本，支持：

- **依赖管理**：自动安装 JDK、Bazel、Python
- **断点续传**：支持从失败处继续安装
- **并行构建**：支持多核并行编译
- **环境配置**：自动生成环境变量配置

**构建脚本**：
- `build/build.sh` - 主安装脚本
- 支持 CMake 和 Bazel 两种构建系统

### 10. 完整的日志系统

- **NPU 专用日志**：集成到工具依赖中
- **VLOG 支持**：详细的调试日志
- **日志级别控制**：通过环境变量控制日志级别

**关键文件**：
- `xla_npu/dependency/tool/` - 日志实现

---

## 许可证

本项目采用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 贡献

欢迎贡献！请随时提交 Issue 或 Pull Request。

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://gitcode.com/cann/xla-npu/issues)
- 查看 [项目文档](https://gitcode.com/cann/xla-npu)

---

## 致谢

本项目基于以下开源项目：

- [OpenXLA](https://github.com/openxla/xla) - XLA 编译器框架
- [JAX](https://github.com/google/jax) - 高性能数值计算库
- [Ascend-MLIR](https://github.com/Ascend/Ascend-MLIR) - 昇腾 MLIR 实现
- [StableHLO](https://github.com/openxla/stablehlo) - StableHLO 方言
