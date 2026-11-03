# 构建与验证

本文介绍 XLA-NPU 的依赖准备、一键构建、运行期环境变量配置与样例验证流程。安装前请确保已安装匹配版本的昇腾 CANN 软件栈。

## 1. 环境要求

### 必须满足

- **Python = 3.11**：必须预先安装（系统自带、apt 安装或 conda 环境均可）。
- **基本工具**：`wget`、`curl`、`git`、`tar`、`xz`、`cmake`、`g++`。
- **磁盘空间**：至少 10 GB 可用空间。
- **CANN 软件栈**：提供 ACL / ACLNN / GE 等运行时与编译接口。

### 推荐配置

- 内存：至少 8 GB。
- CPU：多核处理器（脚本默认使用所有可用核心）。

## 2. 获取代码

```bash
git clone https://gitcode.com/cann/xla-npu.git /path/to/xla-npu
```

## 3. 一键构建

### 3.1 执行构建脚本

```bash
cd /path/to/xla-npu/build
chmod +x build.sh
./build.sh install
```

`build.sh` 会依次完成 JDK、Bazel 与 XLA 的下载，将 `xla_npu/` 软链进 XLA 工作区，使用 Bazel 编译 PJRT 插件，并生成运行环境文件 `build/xla_npu_env`。

### 3.2 构建命令

```bash
./build.sh [命令] [选项]
```

| 命令 | 说明 |
| --- | --- |
| `install` | 执行完整安装（默认） |
| `uninstall` | 卸载 XLA-NPU |
| `help` | 显示帮助信息 |

### 3.3 后端选项

`--npu-auto-fuse-backend=<n>` 用于选择编译期后端，仅 `install` 时有效：

| 取值 | 说明 |
| --- | --- |
| `0` | 不使用融合，仅使用 ACLNN 逐算子执行 |
| `1` | 使用 AFIR/AscIR 自动融合后端（默认） |
| `3` | 使用 GE/AIR 后端 |

示例：

```bash
./build.sh install --npu-auto-fuse-backend=3
```

> 不同后端对 `mmoe`、`qwen` 模型样例的支持存在差异，详见[安装与卸载指南](../../INSTALL_GUIDE.md)。

### 3.4 常用环境变量

| 环境变量 | 说明 |
| --- | --- |
| `PARALLEL_JOBS` | 并行编译任务数，默认取 CPU 核数，如 `export PARALLEL_JOBS=4` |
| `FORCE_REINSTALL` | 设为 `1` 时强制重新执行所有安装步骤，跳过断点续传 |
| `FORCE_UNINSTALL` | 设为 `1` 时卸载无需二次确认 |
| `INSTALL_DIR` | 自定义安装根目录，默认取代码仓根目录 |

## 4. 构建产物

构建完成后可检查以下产物：

```bash
cd /path/to/xla-npu
ls -lh build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so
ls -lh build/xla_npu_env
```

- `pjrt_c_api_npu_plugin.so`：PJRT C API 插件，JAX 通过该动态库接入 NPU。
- `xla_npu_env`：运行期环境变量文件，重复构建不会覆盖其中已有内容。

## 5. 运行期环境变量

### 5.1 加载环境

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/cann/set_env.sh

# 使用自定义安装路径时
source ${ASCEND_INSTALL_PATH}/cann/set_env.sh

# 加载 XLA-NPU 运行期环境变量
source /path/to/xla-npu/build/xla_npu_env
```

`xla_npu_env` 已默认配置以下变量，无需手动设置：

- `XLA_NPU_ROOT_DIR`：XLA-NPU 代码仓根目录。
- `ASCEND_MLIR_PYTHON_PATH`：仓内 Ascend-MLIR 的 Python 目录。
- `NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR`：融合算子编译产物根目录，每次编译会在其下创建以时间戳命名的子目录。

### 5.2 必须配置

- `SOC_VERSION`：标识实际使用的 NPU 芯片版本，例如 `Ascend910B1`。查看方式参考 [acl API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/appdevgapi/aclpythondevg_01_0009.html)。

### 5.3 可选配置

- `ASCEND_OPP_PATH`：部分 CANN 安装的 OPP 不在 `$ASCEND_HOME_PATH/opp`，需要指向实际目录（如 `.../ascend-toolkit/<version>/opp`），否则 ACLNN 算子可能返回 `561103`。
- `USE_OLD_CANN_STYLE=true`：仅在使用 AFIR 融合后端且 CANN ≤ 8.5.0 时需要设置；CANN 9.0.0 及以上请勿设置。
- `XLA_NPU_LOG_LEVEL`：日志级别，取值为 `DEBUG`/`0`、`INFO`/`1`、`WARN`/`2`、`ERROR`/`3`，默认 `INFO`。

## 6. 验证

### 6.1 JAX 后端发现

```python
import os
import jax
import jax.numpy as jnp
import numpy as np

PLATFORM_NAME = 'npu'
LIBRARY_PATH = '/path/to/xla-npu/build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so'
os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = f"{PLATFORM_NAME}:{LIBRARY_PATH}"
os.environ['JAX_PLATFORMS'] = PLATFORM_NAME

import jax.lib.xla_client

print(f"JAX Default Backend: {jax.default_backend()}")
print(f"Available Devices: {jax.devices()}")

x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
y = jax.jit(lambda x: jnp.tanh(x))(x)
print(f"Result: {y.block_until_ready()}")
```

若 `jax.default_backend()` 输出 `npu`，说明插件加载成功。

### 6.2 运行样例

```bash
# 算子测试
cd /path/to/xla-npu/examples/ops
python main.py

# MMOE 模型样例
cd /path/to/xla-npu/examples/mmoe
python main.py --init
python main.py

# Qwen2 模型样例
cd /path/to/xla-npu/examples/qwen
python main.py --init
python main.py
```

样例会自动对比 CPU 与 NPU 的执行结果，详细参数见[样例说明](../../examples/README.md)。

## 7. 卸载

```bash
cd /path/to/xla-npu/build
./build.sh uninstall
```

## 8. 常见问题

### 8.1 Python 版本不满足要求

错误信息：

```text
Python 版本不满足要求（需要 >= 3.11）
```

解决方法：

```bash
# 使用 conda 安装
conda install python=3.11

# 或使用 apt 安装（Ubuntu 22.04+）
sudo apt-get install python3.11
```

### 8.2 编译时内存不足

错误信息：

```text
java.lang.OutOfMemoryError: Java heap space
```

减少并行编译任务数：

```bash
PARALLEL_JOBS=4 ./build.sh install
```

### 8.3 权限问题

错误信息：

```text
无法创建目录: Permission denied
```

解决方法：

```bash
sudo chown -R $USER:$USER /path/to/xla-npu
```

### 8.4 GE 后端场景运行时缺少依赖

错误信息：

```text
INTERNAL: Failed to initialize GE.
```

安装依赖包：

```bash
pip install decorator==5.2.1
pip install sympy==1.14.0
pip install attrs==25.4.0
```

### 8.5 AFIR 后端报错 `No module named 'runtime'`

问题原因：`ASCEND_MLIR_PYTHON_PATH` 环境变量指向了错误或无效路径。

解决方法：重新加载 `build/xla_npu_env`，或手动导出正确路径：

```bash
export ASCEND_MLIR_PYTHON_PATH=/path/to/xla-npu/xla_npu/dependency/external/afir/Ascend-MLIR/python
```
