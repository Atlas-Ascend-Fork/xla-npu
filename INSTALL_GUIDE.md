# XLA-NPU 安装指南

本指南提供了 XLA-NPU 的自动安装脚本使用说明。

## 📋 目录

- [快速开始](#快速开始)
- [安装脚本说明](#安装脚本说明)
- [环境变量配置](#环境变量配置)
- [常见问题](#常见问题)
- [卸载说明](#卸载说明)

---

## 🚀 快速开始

### 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: >= 3.11 (脚本会自动安装 Python 3.11.9，或使用系统/conda Python 3.11+)
- **磁盘空间**: 至少 10GB 可用空间
- **网络**: 需要访问 github.com 和 storage.googleapis.com（或使用代理）

### 安装步骤

1. **克隆 xla-npu 代码**（如果还没有）：
```bash
git clone https://gitcode.com/cann/xla-npu.git
cd xla-npu
```

2. **运行安装脚本**：
```bash
./build/build.sh
```

### 其他安装选项

```bash
# 不安装 JAX
INSTALL_JAX=0 ./build/build.sh

# 使用系统/conda Python（跳过 Python 编译）
SKIP_PYTHON_INSTALL=1 ./build/build.sh

# 自定义工作目录（默认为 xla-npu 根目录）
WORKSPACE=/path/to/workspace ./build/build.sh
```

---

## 📝 安装脚本说明

### 安装脚本：`build/build.sh`

**目录结构**：
```
xla-npu/                          # xla-npu 根目录（用户已克隆）
├── build/                         # 构建目录
│   ├── build.sh        # 安装脚本
│   ├── .install_state            # 安装状态（运行后创建）
│   └── env/                      # 工具安装目录（运行后创建）
│       ├── java/
│       ├── bazel/
│       └── python/
└── xla_npu/                      # NPU 实现代码
```

**功能**：
- 检测并使用系统 Python（如果 >= 3.11）
- 自动安装 OpenJDK 21
- 自动安装 Bazel 7.7.0
- 自动安装 Python 3.11.9（系统 Python 不满足要求时）
- 下载并配置 XLA 代码
- 创建软链接：XLA/xla/xla_npu → xla-npu/xla_npu
- 编译 XLA-NPU 插件
- 编译安装 JAX（默认）

**特点**：
- ✅ 自动检测系统 Python 版本
- ✅ 自动错误处理和恢复建议
- ✅ 并行下载安装包，提高效率
- ✅ 断点续传功能（支持重新运行）
- ✅ 详细的安装日志
- ✅ 进度显示
- ✅ 支持卸载

**使用方法**：
```bash
# 在 xla-npu 根目录执行
./build/build.sh

# 或显式指定安装命令
./build/build.sh install

# 不安装 JAX
INSTALL_JAX=0 ./build/build.sh

# 使用系统/conda Python
SKIP_PYTHON_INSTALL=1 ./build/build.sh

# 自定义工作目录
WORKSPACE=/opt/xla ./build/build.sh

# 自定义并行任务数
PARALLEL_JOBS=16 ./build.sh

# 卸载 XLA-NPU
./build.sh uninstall

# 显示帮助信息
./build.sh help
```

---

## 🔧 环境变量配置

### 脚本环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `WORKSPACE` | 工作目录 | `脚本所在目录/workspace` |
| `PARALLEL_JOBS` | 并行任务数 | `nproc` |
| `MAKE_JOBS` | make 并行数 | `$PARALLEL_JOBS` |
| `INSTALL_JAX` | 是否安装 JAX | 1 (安装) |
| `SKIP_PYTHON_INSTALL` | 跳过 Python 安装 | 0 (不跳过) |
| `FORCE_REINSTALL` | 强制重新安装所有步骤 | 0 (启用断点续传) |
| `SKIP_DOWNLOAD` | 跳过下载步骤 | 0 (不跳过) |
| `FORCE_UNINSTALL` | 强制卸载 | 0 (需要确认) |

**使用示例**：
```bash
# 使用 16 个并行任务编译
PARALLEL_JOBS=16 ./build.sh

# 使用自定义工作目录
WORKSPACE=/opt/xla ./build.sh

# 安装包含 JAX 的完整环境，使用 conda Python
SKIP_PYTHON_INSTALL=1 INSTALL_JAX=1 ./build.sh

# 强制重新安装所有步骤 (跳过断点续传)
FORCE_REINSTALL=1 ./build.sh
```

### 断点续传功能

脚本支持断点续传功能，当安装失败后重新运行脚本时：

- ✅ **自动跳过已完成步骤**：已成功完成的步骤会被跳过
- ✅ **从失败处继续**：自动从失败的步骤继续执行
- ✅ **智能下载检测**：已下载的依赖工具（JDK、Bazel、Python）不会重复下载
- ✅ **配置状态缓存**：XLA 配置完成后会缓存状态，避免重复执行 `./configure.py`
- ✅ **状态持久化**：安装状态保存在 `build/.install_state`

**状态文件位置**：
- 安装状态：`build/.install_state`
- XLA 配置标记：`$XLA_CODE_DIR/.xla_configured_cpu`

**下载和安装分离**：
- 下载目录：`build/downloads/` - 保存所有下载的安装包
- 安装目录：`build/env/` - 工具的实际安装位置
- 安装时从 downloads 目录复制文件到 env 目录，保持 downloads 目录内容不变

**重新执行所有步骤**：
```bash
# 强制重新安装，不使用断点续传
FORCE_REINSTALL=1 ./build.sh
```

### 安装后的环境变量

安装脚本会自动生成环境变量配置文件：`build/xla_npu_env`

**使环境变量生效**：
```bash
# 在 xla-npu 根目录执行
source build/xla_npu_env
```

**或添加到 ~/.bashrc 以自动加载**（可选）：
```bash
echo 'source $(pwd)/build/xla_npu_env' >> ~/.bashrc
source ~/.bashrc
```

**环境变量文件内容**：
```bash
# XLA-NPU 环境变量配置
# 生成时间: 安装时自动生成
# 使用方法: source build/xla_npu_env

export JAVA_HOME=build/env/java/jdk-21
export PATH=$JAVA_HOME/bin:$PATH
export PATH=build/env/bazel:$PATH
export PATH=build/env/python/Python-3.11.9:$PATH
export LD_LIBRARY_PATH=build/env/python/Python-3.11.9:$LD_LIBRARY_PATH
```

---

## ❓ 常见问题

### 1. 网络问题

**问题**: 下载失败或速度慢

**解决方案**：
```bash
# 方式一：使用代理
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080
./build.sh

# 方式二：配置 Git 代理
git config --global http.proxy http://proxy.example.com:8080
git config --global https.proxy http://proxy.example.com:8080

# 方式三：手动下载后跳过下载步骤
# 1. 手动下载以下文件到对应目录：
#    - openjdk-21_linux-x64_bin.tar.gz → $WORKSPACE/env/java/
#    - bazel_nojdk-7.7.0-linux-x86_64 → $WORKSPACE/env/bazel/
#    - Python-3.11.9.tar.xz → $WORKSPACE/env/python/
# 2. 运行安装脚本
SKIP_DOWNLOAD=1 ./build.sh
```

### 2. 编译错误

**问题**: 编译过程中出现错误

**解决方案**：
```bash
# 减少并行任务数
PARALLEL_JOBS=8 ./build.sh

# 检查磁盘空间
df -h

# 查看详细日志
cat $WORKSPACE/logs/build_*.log
```

### 3. Python 版本问题

**问题**: Python 版本不满足要求

**解决方案**：
```bash
# 脚本会自动安装 Python 3.11.9
# 如果系统已有 Python 3.11+，可以跳过自动安装

# 检查 Python 版本
python --version

# 使用 conda Python
SKIP_PYTHON_INSTALL=1 ./build.sh

# 如需使用系统 Python，请确保版本 >= 3.11
```

### 4. 权限问题

**问题**: 安装时权限不足

**解决方案**：
```bash
# 使用 sudo 运行（不推荐）
sudo ./build.sh

# 或确保安装目录有写权限（推荐）
mkdir -p $WORKSPACE
chmod u+w $WORKSPACE
```

### 5. Bazel 首次编译慢

**问题**: 第一次 Bazel 编译非常慢

**说明**：
- 这是正常现象，Bazel 首次编译需要下载大量依赖
- 依赖主要来自 github.com 和 storage.googleapis.com
- 可以使用代理加速

**解决方案**：
```bash
# 设置代理加速
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080

# 或配置 Bazel 使用代理
export BAZEL_HTTP_PROXY=http://proxy.example.com:8080
```

### 6. Git 私仓权限问题

**问题**: 无法克隆 xla-npu 私仓

**解决方案**：
```bash
# 配置 Git 凭据
git config --global credential.helper store

# 或使用 SSH 密钥
# 1. 生成 SSH 密钥
ssh-keygen -t rsa -b 4096

# 2. 添加到 GitCode/GitHub

# 3. 使用 SSH URL 克隆
# 修改脚本中的 git clone 命令使用 git@gitcode.com:cann/xla-npu.git
```

### 7. 网络受限环境（无法访问 github.com）

**问题**: Git clone 失败，无法连接到 github.com

**解决方案**：
```bash
# 方式一：使用代理
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080
git config --global http.proxy http://proxy.example.com:8080
git config --global https.proxy http://proxy.example.com:8080

# 方式二：使用 GitHub 镜像站
# 修改脚本中的 git clone URL：
# GitHub: https://github.com/openxla/xla.git
# 镜像1: https://gitee.com/mirrors/xla.git
# 镜像2: https://hub.fastgit.xyz/openxla/xla.git

# 方式三：手动下载代码
# 1. 在有网络的机器上下载代码仓库
# 2. 传输到目标机器
# 3. 放置到 $WORKSPACE/code 目录
```

### 8. 重新运行脚本时如何处理

**问题**: 重新运行脚本是否会重复执行所有步骤？

**解决方案**：
```bash
# 默认情况下，脚本会自动跳过已完成的步骤
./build/build.sh

# 查看当前安装状态
cat build/.install_state

# 强制重新执行所有步骤
FORCE_REINSTALL=1 ./build/build.sh

# 手动清除某个步骤的状态 (例如重新下载代码)
sed -i '/download_code=/d' build/.install_state
```

---

## 🧹 卸载说明

### 删除环境变量配置

```bash
# 删除环境变量配置文件
rm build/xla_npu_env

# 如果已添加到 ~/.bashrc，删除相关行
sed -i '/xla_npu_env/d' ~/.bashrc
```

### 删除代码和编译产物

```bash
# 使用脚本卸载（交互式）
./build/build.sh uninstall

# 或手动删除整个 build 目录
rm -rf build/
```

### 清理缓存

```bash
# 清理 Bazel 缓存
rm -rf ~/.cache/bazel

# 清理 pip 缓存
pip cache purge
```

---

## 📊 安装目录结构

安装完成后的目录结构：

```
xla-npu/                          # xla-npu 根目录（默认 WORKSPACE）
├── build/                        # 构建目录
│   ├── build.sh      # 安装脚本
│   ├── xla_npu_env            # 环境变量配置文件
│   ├── .install_state          # 安装状态文件
│   ├── env/                    # 工具安装目录
│   │   ├── java/
│   │   │   └── jdk-21/         # OpenJDK 21
│   │   ├── bazel/              # Bazel 7.7.0
│   │   └── python/
│   │       └── Python-3.11.9/  # Python 3.11.9
│   ├── code/                   # 代码下载目录
│   │   ├── xla/               # OpenXLA 代码
│   │   │   ├── xla/
│   │   │   │   └── xla_npu/    # xla-npu 软链接 → ../../xla_npu/
│   │   │   └── bazel-bin/     # Bazel 编译产物
│   │   └── jax/               # JAX 代码 (可选)
│   ├── downloads/              # 临时下载目录
│   └── logs/                   # 日志目录
│       ├── build_*.log # 安装日志
│       └── BUILD_DOCUMENTATION.md # 安装文档
└── xla_npu/                    # NPU 实现代码（用户已克隆）
```

**说明**：
- `XLA/xla/xla_npu` 是软链接，指向 `xla-npu/xla_npu`
- xla-npu 代码不会被删除（用户仓库）
- XLA、JAX 代码在 `build/code/` 目录下
- 编译工具在 `build/env/` 目录下

---

## ✅ 验证安装

**注意**：脚本运行时环境变量已自动设置，可直接验证。如在新终端中验证，请先设置环境变量（见上文"安装后的环境变量"）。

### 验证 Java

```bash
$WORKSPACE/env/java/jdk-21/bin/java --version
# 预期输出: openjdk 21.x.x
```

### 验证 Bazel

```bash
$WORKSPACE/env/bazel/bazel --version
# 预期输出: bazel 7.7.0
```

### 验证 Python

```bash
$WORKSPACE/env/python/Python-3.11.9/bin/python --version
# 预期输出: Python 3.11.9
```

### 验证 XLA-NPU 产物

```bash
ls -lh $WORKSPACE/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so
```

### 验证 JAX（如果安装）

```bash
$WORKSPACE/env/python/Python-3.11.9/bin/python -c "import jax; print(jax.__version__)"
# 预期输出: 0.8.x
```

