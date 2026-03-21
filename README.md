# XLA-NPU

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.8.1-orange.svg)](https://github.com/google/jax)

## 简介

**XLA-NPU** 是一个面向华为昇腾NPU（Neural Processing Unit）硬件的 **XLA（Accelerated Linear Algebra）后端实现**。本项目通过接入OpenXLA/XLA开源项目，将XLA开源生态与华为 CANN（Compute Architecture for Neural Networks）软件栈集成，对接JAX框架。JAX框架运行时可以直接加载XLA-NPU，使得基于JAX框架开发的模型可以运行在昇腾NPU上，提供推理场景图编译加速能力。

昇腾为基于华为昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问昇腾社区，了解关于昇腾的更多信息。

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

详细的编译和运行环境变量配置请参考：[INSTALL_GUIDE.md](INSTALL_GUIDE.md)

### 快速设置

使用 `build/build.sh` 脚本安装后，可通过以下命令自动设置环境变量：
```bash
source build/xla_npu_env
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
LIBRARY_PATH = '/path/to/xla-npu/build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so'
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
xla\_npu插件通过实现OpenXLA PJRT接口，将NPU设备接入XLA生态，对接JAX框架。详细特性介绍请访问[docs](docs/zh/README.md)。

---

## 支持的型号
- Atlas A2/A3 训练系列产品
- Atlas A3 推理系列产品

## 许可证

本项目采用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 贡献

欢迎贡献！请随时提交 Issue 或 Pull Request。

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://gitcode.com/cann/xla-npu/issues)

---

## 致谢

本项目基于以下开源项目：

- [OpenXLA](https://github.com/openxla/xla) - XLA 编译器框架
- [JAX](https://github.com/google/jax) - 高性能数值计算库
- [StableHLO](https://github.com/openxla/stablehlo) - StableHLO 方言
