# XLA-NPU

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.8+-orange.svg)](https://github.com/google/jax)

## 简介

**XLA-NPU** 是一个面向华为昇腾NPU（Neural Processing Unit）硬件的 **XLA（Accelerated Linear Algebra）后端实现**。本项目将 OpenXLA/JAX 与华为 CANN（Compute Architecture for Neural Networks）工具链集成，使机器学习工作负载能够在华为昇腾 NPU 硬件上高效运行。

## 安装与卸载

详细的安装和卸载指南请参考：**[INSTALL_GUIDE.md](INSTALL_GUIDE.md)**

---

### 快速安装

```bash
# 克隆代码仓库
git clone https://gitcode.com/cann/xla-npu.git
cd xla-npu

# 添加可执行权限
chmod +x build/build.sh

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

## 环境变量说明

### 编译环境变量

编译前需要设置以下环境变量:

- `JAVA_HOME`: Java开发工具包路径
- `PATH`: 需包含Java bin目录和Bazel可执行文件目录
- `XDG_CACHE_HOME`: (可选) 缓存目录,用于存储Bazel编译过程中的缓存文件

### 条件编译选项

通过设置 `NPU_AUTO_FUSE_BACKEND` 环境变量控制算子编译所使用的后端:

- `0`: 不支持融合, 仅使用aclnn执行
- `1`: 使用afir后端
- `2`: 使用npuir后端  
- `3`: 使用ge后端

### 运行环境变量

运行测试前需要在xla代码仓目录下的xla_configure.bazelrc文件(由执行configure.py生成)中设置以下环境变量:

- `ASCEND_MLIR_PYTHON_PATH`: 环境上Python可执行文件路径
- `NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR`: 算子编译产生输出的目录, 每次编译时会在其下创建以时间戳命名的子目录

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

---

## 特性介绍

### 1. 完整的 XLA 后端实现


### 2. PJRT C API 插件


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
