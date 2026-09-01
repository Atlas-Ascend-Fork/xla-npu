# XLA-NPU

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-green.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.8.1-orange.svg)](https://github.com/google/jax)

## 简介

**XLA-NPU** 是一个面向华为昇腾NPU（Neural Processing Unit）硬件的 **XLA（Accelerated Linear Algebra）后端实现**。本项目通过接入OpenXLA/XLA开源项目，将XLA开源生态与华为 CANN（Compute Architecture for Neural Networks）软件栈集成，对接JAX框架。JAX框架运行时可以直接加载XLA-NPU，使得基于JAX框架开发的模型可以运行在昇腾NPU上，提供推理场景图编译加速能力。

昇腾为基于华为昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问[昇腾社区](https://www.hiascend.com/zh/)，了解关于昇腾的更多信息。

## 安装与卸载

详细的安装和卸载指南请参考：**[INSTALL_GUIDE.md](INSTALL_GUIDE.md)**
- **XLA-NPU在运行时依赖CANN 8.5.0及以上版本，请参考[安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=Ubuntu)完成软件栈的安装。**
---


## 快速上手
### 1. 配置CANN环境变量
```bash
# 默认路径安装
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
source ${ASCEND_INSTALL_PATH}/cann/set_env.sh
```
### 2. 配置运行期环境变量

> 建议在运行任意xla-npu测试用例前，执行`source ./build/xla_npu_env` （此文件在执行`./build/build.sh`后生成，且重复执行`./build/build.sh`后，不会再覆盖./build/xla_npu_env中的内容），确保所有必要运行环境变量已被配置。下面列举的环境变量中，标注为`已默认配置在xla_npu_env中`的，执行`source ./build/xla_npu_env`后，可不用再次手动配置。

#### 必须配置
- `export ASCEND_MLIR_PYTHON_PATH=`: xla-npu代码仓中dependency下载的Ascend-MLIR中Python可执行文件路径（比如`/path/to/xla-npu/xla_npu/dependency/external/afir/Ascend-MLIR/python`） - `已默认配置在xla_npu_env中`
- `export NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR=`：算子编译产生输出的目录， 每次编译时会在其下创建以时间戳命名的子目录 - `已默认配置在xla_npu_env中`
- `export SOC_VERSION=Ascend910B1`：设置此环境变量用于标识实际环境使用的NPU芯片版本，查看方式参考[acl API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/appdevgapi/aclpythondevg_01_0009.html)。
#### 可选配置
- `export USE_OLD_CANN_STYLE=true`：在使用NPU_AUTO_FUSE_BACKEND=1，即afir融合后端时，如果是在CANN 9.0.0及其之前的版本运行xla-npu用例，则需要设置此环境变量，否则会报错

### 3. 在 JAX 中使用 NPU 后端

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

## 运行样例验证
完成安装与环境配置后，可参考 [examples/README.md](examples/README.md) 运行基础算子测试与模型推理样例（ops / mmoe / qwen），验证 XLA-NPU 功能，并支持 CPU 与 NPU 执行结果对比。

---

## 特性介绍
xla\_npu插件通过实现OpenXLA PJRT接口，将NPU设备接入XLA生态，对接JAX框架。详细特性介绍请访问[docs](docs/zh/README.md)。

---

## 支持的型号
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

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
