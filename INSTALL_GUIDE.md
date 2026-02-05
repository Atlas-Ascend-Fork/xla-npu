# XLA-NPU 安装指南 (简化版)

## 概述

本文档介绍如何使用 `build_new.sh` 脚本安装 XLA-NPU。该脚本是一个简化版本，相比原版 `build.sh` 做了以下优化：

1. **不安装 Python**：Python 作为系统要求，需要用户预先安装
2. **使用 pip 安装 JAX**：不再编译安装 JAX，直接使用 `pip install jax==0.8.1`
3. **使用华为镜像源**：所有下载使用华为镜像加速

## 系统要求

### 必须满足
- **Python >= 3.11**：必须预先安装（系统自带、apt 安装或 conda 环境）
- **基本工具**：`wget`, `curl`, `git`, `tar`, `xz`
- **磁盘空间**：至少 10GB 可用空间

### 自动安装
以下工具由脚本自动安装：
- OpenJDK 21
- Bazel 7.7.0

### 推荐配置
- 内存：至少 8GB
- CPU：多核处理器（脚本会自动使用所有可用核心）

## 安装前准备

### 1. 安装 Python

#### 方法一：使用系统包管理器
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.11 python3-pip

# 验证安装
python3.11 --version
```

#### 方法二：使用 conda（推荐）
```bash
# 创建新环境
conda create -n xla-npu python=3.11
conda activate xla-npu

# 验证安装
python --version
```

#### 方法三：从源码编译
如果系统没有 Python 3.11+，可以参考旧版 `build.sh` 中的 Python 编译部分。

### 2. 安装基本工具
```bash
sudo apt-get update
sudo apt-get install wget curl git tar xz build-essential
```

### 3. 克隆 xla-npu 仓库
```bash
git clone https://gitcode.com/cann/xla-npu.git /path/to/xla-npu
cd /path/to/xla-npu
```

## 安装步骤

### 1. 进入安装脚本目录
```bash
cd /path/to/xla-npu/install
```

### 2. 给脚本添加执行权限
```bash
chmod +x build_new.sh
```

### 3. 运行安装脚本
```bash
./build_new.sh install
```

或者，如果你使用 conda 环境：
```bash
conda activate xla-npu
./build_new.sh install
```

### 4. 设置环境变量
安装完成后，运行以下命令使环境变量生效：
```bash
source ../build/xla_npu_env_new
```

或者添加到 `~/.bashrc` 自动加载：
```bash
echo "source $(pwd)/../build/xla_npu_env_new" >> ~/.bashrc
source ~/.bashrc
```

## 安装选项

### 环境变量配置

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `WORKSPACE` | 工作目录 | xla-npu 根目录 |
| `PARALLEL_JOBS` | 并行编译任务数 | `nproc` (CPU 核心数) |
| `INSTALL_JAX` | 是否安装 JAX | `1` (安装) |
| `FORCE_REINSTALL` | 强制重新安装 | `0` (启用断点续传) |
| `FORCE_UNINSTALL` | 强制卸载无需确认 | `0` (需要确认) |

### 使用示例

#### 仅安装 XLA-NPU，不安装 JAX
```bash
INSTALL_JAX=0 ./build_new.sh install
```

#### 强制重新安装所有步骤
```bash
FORCE_REINSTALL=1 ./build_new.sh install
```

#### 自定义工作目录
```bash
WORKSPACE=/opt/xla-npu ./build_new.sh install
```

#### 指定并行编译任务数
```bash
PARALLEL_JOBS=8 ./build_new.sh install
```

## 验证安装

### 1. 验证 Java
```bash
java --version
# 应输出: openjdk 21.x.x
```

### 2. 验证 Bazel
```bash
bazel --version
# 应输出: bazel 7.7.0
```

### 3. 验证 Python
```bash
python --version
# 应输出: Python 3.11.x 或更高版本
```

### 4. 验证 XLA-NPU 编译产物
```bash
ls -lh build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so
```

### 5. 验证 JAX 安装
```bash
python -c "import jax; print(jax.__version__)"
# 应输出: 0.8.1
```

## 目录结构

安装完成后的目录结构：
```
xla-npu/                          # 项目根目录
├── build/                        # 构建目录
│   ├── build_new.sh             # 安装脚本
│   ├── xla_npu_env_new          # 环境变量文件
│   ├── .install_state_new       # 安装状态文件
│   ├── env/                     # 已安装的工具
│   │   ├── java/                # OpenJDK 21
│   │   └── bazel/               # Bazel 7.7.0
│   ├── code/                    # 下载的代码
│   │   ├── xla/                 # XLA 代码仓库
│   │   └── logs/                # 安装日志
│   └── downloads/               # 下载的安装包（缓存）
└── xla_npu/                      # NPU 实现代码（软链接目标）
```

## 与原版 build.sh 的区别

| 特性 | build.sh (原版) | build_new.sh (新版) |
|------|-----------------|---------------------|
| Python 安装 | 自动编译安装 Python 3.11.9 | 要求系统自带 Python >= 3.11 |
| JAX 安装 | 从源码编译安装 | 使用 `pip install jax==0.8.1` |
| 安装时间 | 较长（需编译 Python 和 JAX） | 较短（无需编译 Python 和 JAX） |
| pip 源 | 默认 PyPI | 华为镜像源 |
| 适用场景 | 需要完整隔离环境 | 系统已有合适的 Python |

## 常见问题

### 1. Python 版本不满足要求

**错误信息**：
```
Python 版本不满足要求 (需要 >= 3.11, 当前 3.9.x)
```

**解决方案**：
```bash
# 使用 conda 安装
conda install python=3.11

# 或使用 apt 安装 (Ubuntu 22.04+)
sudo apt-get install python3.11
```

### 2. 下载速度慢或失败

**解决方案**：
- 脚本默认使用华为镜像源
- 如仍有问题，可配置代理：
  ```bash
  export http_proxy=http://proxy.example.com:8080
  export https_proxy=http://proxy.example.com:8080
  ```

### 3. 编译时内存不足

**错误信息**：
```
java.lang.OutOfMemoryError: Java heap space
```

**解决方案**：
减少并行编译任务数：
```bash
PARALLEL_JOBS=4 ./build_new.sh install
```

### 4. JAX 安装失败

**错误信息**：
```
安装 jaxlib 失败
```

**解决方案**：
```bash
# 手动安装
python -m pip install --upgrade pip
python -m pip install -i https://mirrors.huaweicloud.com/repository/pypi/simple --trusted-host mirrors.huaweicloud.com jaxlib==0.8.1 jax==0.8.1
```

### 5. 权限问题

**错误信息**：
```
无法创建目录: Permission denied
```

**解决方案**：
```bash
# 确保对工作目录有写权限
sudo chown -R $USER:$USER /path/to/xla-npu

# 或使用 sudo 运行脚本（不推荐）
sudo ./build_new.sh install
```

## 卸载

### 自动卸载
```bash
cd /path/to/xla-npu/install
./build_new.sh uninstall
```

### 强制卸载（无需确认）
```bash
FORCE_UNINSTALL=1 ./build_new.sh uninstall
```

### 手动卸载

#### 1. 卸载 JAX
```bash
python -m pip uninstall -y jax jaxlib
```

#### 2. 删除环境变量配置
```bash
rm build/xla_npu_env_new

# 如果已添加到 ~/.bashrc
sed -i '/xla_npu_env_new/d' ~/.bashrc
```

#### 3. 删除编译产物和代码
```bash
rm -rf build/code/xla
rm -rf build/env
```

#### 4. 清理缓存
```bash
# 清理 Bazel 缓存
rm -rf ~/.cache/bazel

# 清理 pip 缓存
python -m pip cache purge
```

## 高级用法

### 1. 使用特定的 Python 环境
```bash
# 激活 conda 环境
conda activate my-env

# 或使用 virtualenv
source /path/to/venv/bin/activate

# 运行安装
./build_new.sh install
```

### 2. 断点续传
脚本支持断点续传，如果安装过程中断，重新运行脚本会从上次中断的地方继续：
```bash
./build_new.sh install
```

如需从头开始：
```bash
FORCE_REINSTALL=1 ./build_new.sh install
```

### 3. 仅安装特定组件
```bash
# 仅安装 XLA-NPU，不安装 JAX
INSTALL_JAX=0 ./build_new.sh install
```

## 日志和调试

### 查看安装日志
```bash
# 日志目录
ls -lh build/logs/

# 查看最新日志
cat build/logs/build_*.log | tail -100
```

### 启用详细输出
```bash
# 使用 bash -x 运行脚本查看详细执行过程
bash -x ./build_new.sh install
```

## 技术支持

如遇问题，请检查：
1. 安装日志：`build/logs/build_*.log`
2. Python 版本：`python --version`
3. 网络连接：`ping mirrors.huaweicloud.com`

## 附录：命令参考

### 脚本命令

| 命令 | 说明 |
|------|------|
| `./build_new.sh install` | 执行完整安装 |
| `./build_new.sh uninstall` | 卸载 XLA-NPU |
| `./build_new.sh help` | 显示帮助信息 |

### 关键文件路径

| 文件 | 路径 |
|------|------|
| 安装脚本 | `build/build_new.sh` |
| 环境变量文件 | `build/xla_npu_env_new` |
| 状态文件 | `build/.install_state_new` |
| 安装日志 | `build/logs/build_*.log` |
| XLA 代码 | `build/code/xla/` |
| XLA 产物 | `build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so` |

## 版本信息

- **脚本版本**: build_new.sh (简化版)
- **JAX 版本**: 0.8.1
- **XLA Commit**: a7e58876b73dfa9bb6bc785f6981a332c133bb55
- **Bazel 版本**: 7.7.0
- **OpenJDK 版本**: 21

---

**注意**: 本安装脚本已针对国内网络环境优化，使用华为镜像源加速下载。如需使用其他镜像源，可修改脚本中的镜像配置。
