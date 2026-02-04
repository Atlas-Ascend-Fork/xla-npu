#!/bin/bash
################################################################################
# XLA-NPU 自动安装脚本
# 功能：自动安装和编译 XLA-NPU 及相关依赖，支持卸载
# 作者：基于 /home/code/xla/install/install-guide.txt
# 日期：$(date +%Y-%m-%d)
#
# 使用方法:
#   ./build.sh [install|uninstall|help]
#   install  - 执行完整安装 (默认)
#   uninstall- 卸载 XLA-NPU
#   help     - 显示帮助信息
################################################################################

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检测 xla-npu 根目录（脚本目录的父目录）
XLA_NPU_ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

#################### 配置区域 ####################
# 安装目录配置
# 优先使用环境变量 WORKSPACE，否则使用 xla-npu 根目录
WORKSPACE=${WORKSPACE:-"$XLA_NPU_ROOT_DIR"}
ENV_DIR="$WORKSPACE/build/env"
CODE_DIR="$WORKSPACE/build/code"
DOWNLOADS_DIR="$WORKSPACE/build/downloads"
JAVA_INSTALL_DIR="$ENV_DIR/java"
PYTHON_INSTALL_DIR="$ENV_DIR/python"
BAZEL_INSTALL_DIR="$ENV_DIR/bazel"
XLA_CODE_DIR="$CODE_DIR/xla"
# xla-npu 代码目录为脚本所在目录的父目录（用户已克隆）
XLA_NPU_CODE_DIR="$XLA_NPU_ROOT_DIR"
JAX_CODE_DIR="$CODE_DIR/jax"

# 版本配置
JAVA_VERSION="21"
PYTHON_VERSION="3.11.9"
BAZEL_VERSION="7.7.0"
JAX_VERSION="jax-v0.8.1"
XLA_COMMIT_ID="a7e58876b73dfa9bb6bc785f6981a332c133bb55"

# 下载镜像配置
JAVA_MIRROR="https://mirrors.huaweicloud.com/openjdk"
BAZEL_MIRROR="https://mirrors.huaweicloud.com/bazel"
PYTHON_MIRROR="https://mirrors.huaweicloud.com/python"

# 并行编译任务数
PARALLEL_JOBS=${PARALLEL_JOBS:-$(nproc)}
MAKE_JOBS=${MAKE_JOBS:-$PARALLEL_JOBS}

# 默认安装 JAX (1=安装, 0=不安装)
INSTALL_JAX=${INSTALL_JAX:-1}

# 跳过 Python 安装，使用系统/conda Python (1=跳过, 0=安装)
SKIP_PYTHON_INSTALL=${SKIP_PYTHON_INSTALL:-0}

# 断点续传：强制重新安装 (1=强制重新执行所有步骤, 0=启用断点续传)
FORCE_REINSTALL=${FORCE_REINSTALL:-0}

# 状态文件路径（放在 build 目录）
STATE_FILE="$SCRIPT_DIR/.install_state"

# 日志文件
LOG_DIR="$WORKSPACE/build/logs"
LOG_FILE="$LOG_DIR/build_$(date +%Y%m%d_%H%M%S).log"
INSTALL_DOC="$LOG_DIR/BUILD_DOCUMENTATION.md"

# 环境变量文件（放在 build 目录）
ENV_FILE="$SCRIPT_DIR/xla_npu_env"

#################### 工具函数 ####################
log_info() {
    echo -e "\033[32m[INFO]\033[0m $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "\033[33m[WARN]\033[0m $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "\033[31m[ERROR]\033[0m $*" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "\033[36m[STEP]\033[0m $*" | tee -a "$LOG_FILE"
}

# 错误处理函数
error_exit() {
    local error_msg="$1"
    local suggestion="$2"
    log_error "$error_msg"
    if [ -n "$suggestion" ]; then
        echo -e "\033[33m[SUGGESTION]\033[0m $suggestion" | tee -a "$LOG_FILE"
    fi
    echo -e "\033[31m操作失败，请查看日志文件: $LOG_FILE\033[0m"
    exit 1
}

# 检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

# 检查 Python 命令 (兼容 python 和 python3)
check_python() {
    if command -v python &> /dev/null; then
        echo "python"
        return 0
    elif command -v python3 &> /dev/null; then
        echo "python3"
        return 0
    fi
    return 1
}

# 创建目录
ensure_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1" || error_exit "无法创建目录: $1" "请检查目录权限或磁盘空间"
    fi
}

#################### 状态管理函数 ####################
# 标记步骤为已完成
mark_step_done() {
    local step="$1"
    echo "$step=done" >> "$STATE_FILE"
    log_info "步骤完成: $step"
}

# 检查步骤是否已完成
is_step_done() {
    local step="$1"

    # 如果强制重新安装，返回未完成
    if [ "$FORCE_REINSTALL" = "1" ]; then
        return 1
    fi

    # 检查状态文件
    if [ -f "$STATE_FILE" ]; then
        if grep -q "^${step}=done" "$STATE_FILE" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 清除步骤状态
clear_step_state() {
    local step="$1"
    if [ -f "$STATE_FILE" ]; then
        sed -i "/^${step}=/d" "$STATE_FILE" 2>/dev/null || true
    fi
}

# 检查并跳过已完成的步骤
skip_if_done() {
    local step="$1"
    local step_name="$2"

    if is_step_done "$step"; then
        log_info "跳过已完成步骤: $step_name"
        return 0  # 返回0表示应该跳过
    fi
    return 1  # 返回1表示不应该跳过
}

# 确认操作
confirm_action() {
    local prompt="$1"
    local default="${2:-n}"

    if [ "$FORCE_UNINSTALL" = "1" ]; then
        return 0
    fi

    echo -en "\033[33m$prompt\033[0m"
    read -p " [y/N]: " -r response
    response=${response:-$default}

    if [[ $response =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

#################### 初始化环境 ####################
init_environment() {
    # 首先创建日志目录（在调用任何 log 函数之前）
    mkdir -p "$LOG_DIR" 2>/dev/null || true
    mkdir -p "$WORKSPACE" 2>/dev/null || true

    log_step "=== 初始化安装环境 ==="

    # 创建必要的目录
    ensure_dir "$WORKSPACE"
    ensure_dir "$ENV_DIR"
    ensure_dir "$CODE_DIR"
    ensure_dir "$LOG_DIR"
    ensure_dir "$JAVA_INSTALL_DIR"
    ensure_dir "$PYTHON_INSTALL_DIR"
    ensure_dir "$BAZEL_INSTALL_DIR"

    log_info "工作目录: $WORKSPACE"
    log_info "环境目录: $ENV_DIR"
    log_info "代码目录: $CODE_DIR"
    log_info "日志目录: $LOG_DIR"
    log_info "日志文件: $LOG_FILE"
    log_info "并行任务数: $PARALLEL_JOBS"
    log_info "安装 JAX: $([ "$INSTALL_JAX" = "1" ] && echo "是" || echo "否")"

    # 检查必要的工具
    local missing_tools=()

    for tool in wget curl git tar xz; do
        if ! check_command "$tool"; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -gt 0 ]; then
        error_exit "缺少必要工具: ${missing_tools[*]}" \
            "请先安装: sudo apt-get install ${missing_tools[*]}"
    fi

    # 检查 Python
    PYTHON_CMD=$(check_python)
    if [ -z "$PYTHON_CMD" ]; then
        error_exit "未找到 Python" \
            "请安装 Python 3.11+: conda install python=3.11 或 apt-get install python3.11"
    fi

    # 获取当前 Python 版本
    local current_py_version=$($PYTHON_CMD --version 2>&1 | head -1)
    log_info "检测到 Python: $current_py_version (命令: $PYTHON_CMD)"

    log_info "环境初始化完成"
}

#################### 安装 Java ####################
install_java() {
    # 检查是否已完成（必须检查实际目录是否存在）
    local java_dir="$JAVA_INSTALL_DIR/jdk-$JAVA_VERSION"
    if [ -d "$java_dir/bin" ] && skip_if_done "install_java" "OpenJDK $JAVA_VERSION"; then
        # 设置环境变量
        export JAVA_HOME="$java_dir"
        export PATH="$JAVA_HOME/bin:$PATH"
        return 0
    fi

    log_step "=== 安装 OpenJDK $JAVA_VERSION ==="

    local java_file="openjdk-${JAVA_VERSION}_linux-x64_bin.tar.gz"
    local downloads_file="$DOWNLOADS_DIR/$java_file"

    # 检查下载的文件是否存在
    if [ ! -f "$downloads_file" ]; then
        error_exit "JDK 安装文件不存在: $downloads_file" \
            "请先运行下载步骤或手动下载到该位置"
    fi

    cd "$JAVA_INSTALL_DIR" || error_exit "无法进入目录: $JAVA_INSTALL_DIR"

    # 从 downloads 目录复制文件
    log_info "准备 JDK 安装文件..."
    if [ ! -f "$java_file" ] || [ "$downloads_file" -nt "$java_file" ]; then
        cp "$downloads_file" "$java_file" || error_exit "复制 JDK 文件失败"
    fi

    # 解压 JDK
    log_info "解压 OpenJDK..."
    if [ ! -d "$java_dir" ]; then
        tar -xzf "$java_file" || error_exit "解压 OpenJDK 失败" "检查文件是否完整"
        # 清理 tar.gz 文件以节省空间
        rm -f "$java_file"
    else
        log_info "JDK 已解压，跳过"
    fi

    # 设置环境变量
    export JAVA_HOME="$java_dir"
    export PATH="$JAVA_HOME/bin:$PATH"

    # 验证安装
    log_info "验证 Java 安装..."
    if ! "$java_dir/bin/java" --version; then
        error_exit "Java 安装验证失败" "检查 JAVA_HOME 和 PATH 配置"
    fi

    log_info "Java 安装成功"
    mark_step_done "install_java"
}

#################### 安装 Bazel ####################
install_bazel() {
    # 检查是否已完成（必须检查实际文件是否存在）
    local bazel_link="$BAZEL_INSTALL_DIR/bazel"
    if [ -f "$bazel_link" ] && skip_if_done "install_bazel" "Bazel $BAZEL_VERSION"; then
        export PATH="$BAZEL_INSTALL_DIR:$PATH"
        return 0
    fi

    log_step "=== 安装 Bazel $BAZEL_VERSION ==="

    local bazel_file="bazel_nojdk-${BAZEL_VERSION}-linux-x86_64"
    local downloads_file="$DOWNLOADS_DIR/$bazel_file"

    # 检查下载的文件是否存在
    if [ ! -f "$downloads_file" ]; then
        error_exit "Bazel 安装文件不存在: $downloads_file" \
            "请先运行下载步骤或手动下载到该位置"
    fi

    cd "$BAZEL_INSTALL_DIR" || error_exit "无法进入目录: $BAZEL_INSTALL_DIR"

    # 从 downloads 目录复制文件
    log_info "准备 Bazel 安装文件..."
    if [ ! -f "$bazel_file" ] || [ "$downloads_file" -nt "$bazel_file" ]; then
        cp "$downloads_file" "$bazel_file" || error_exit "复制 Bazel 文件失败"
    fi

    # 设置可执行权限并创建符号链接
    chmod +x "$bazel_file" || error_exit "无法设置 Bazel 可执行权限"
    ln -sf "$BAZEL_INSTALL_DIR/$bazel_file" "$bazel_link" || \
        error_exit "无法创建 Bazel 符号链接"

    # 设置环境变量
    export PATH="$BAZEL_INSTALL_DIR:$PATH"

    # 验证安装
    log_info "验证 Bazel 安装..."
    if ! "$BAZEL_INSTALL_DIR/bazel" --version; then
        error_exit "Bazel 安装验证失败" "检查 PATH 配置"
    fi

    log_info "Bazel 安装成功"
    mark_step_done "install_bazel"
}

#################### 安装 Python ####################
install_python() {
    # 检查是否已完成
    local python_dir="$PYTHON_INSTALL_DIR/Python-${PYTHON_VERSION}"

    # 首先检查是否已完成（无论是否设置 SKIP_PYTHON_INSTALL）
    if skip_if_done "install_python" "Python $PYTHON_VERSION"; then
        # 已完成过安装，检查已安装的 Python
        if [ "$SKIP_PYTHON_INSTALL" != "1" ]; then
            # 需要编译安装 Python，检查是否已完成
            # 检查两种情况：1. make install 后的 bin/python 2. 编译完成但未 install 的源码目录 python
            local python_exe=""
            if [ -f "$python_dir/bin/python" ]; then
                python_exe="$python_dir/bin/python"
            elif [ -f "$python_dir/python" ] && [ -f "$python_dir/libpython3.11.so" ]; then
                # 检查编译完成但未 install 的情况
                python_exe="$python_dir/python"
            fi

            # 如果找到 python 可执行文件，验证是否真正可用
            if [ -n "$python_exe" ]; then
                # 验证 Python 是否真的可用
                local temp_ld_path="$LD_LIBRARY_PATH"
                export LD_LIBRARY_PATH="$python_dir:$LD_LIBRARY_PATH"
                if $python_exe --version >/dev/null 2>&1; then
                    export PATH="$python_dir:$PATH"
                    PYTHON_CMD="$python_exe"
                    return 0
                else
                    export LD_LIBRARY_PATH="$temp_ld_path"
                    log_warn "Python 可执行文件存在但不可用，将重新安装"
                fi
            fi
        else
            # 使用系统/conda Python
            return 0
        fi
    fi

    log_step "=== 检查 Python 环境 ==="

    # 先检查系统 Python 版本（无论是否设置 SKIP_PYTHON_INSTALL）
    if check_command python3 || check_command python; then
        local system_python=""
        if check_command python3; then
            system_python="python3"
        else
            system_python="python"
        fi

        log_info "检测到系统 Python: $($system_python --version 2>&1 | head -1)"

        # 检查 Python 版本是否满足要求 (>= 3.11)
        local py_major=$($system_python -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
        local py_minor=$($system_python -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")

        if [ "$py_major" -gt 3 ] || ([ "$py_major" -eq 3 ] && [ "$py_minor" -ge 11 ]); then
            log_info "系统 Python 版本满足要求 ($py_major.$py_minor >= 3.11)，跳过编译安装"
            PYTHON_CMD="$system_python"
            mark_step_done "install_python"
            return 0
        else
            log_info "系统 Python 版本不满足要求 ($py_major.$py_minor < 3.11)，将编译安装 Python $PYTHON_VERSION"
        fi
    else
        log_info "未检测到系统 Python，将编译安装 Python $PYTHON_VERSION"
    fi

    # 如果设置了跳过 Python 安装且系统 Python 不满足要求，则报错
    if [ "$SKIP_PYTHON_INSTALL" = "1" ]; then
        error_exit "SKIP_PYTHON_INSTALL=1 但系统 Python 版本不满足要求" \
            "需要 Python >= 3.11\n解决方案: conda install python=3.11 或取消 SKIP_PYTHON_INSTALL"
    fi

    log_step "=== 安装 Python $PYTHON_VERSION ==="

    # 安装编译依赖
    log_info "安装 Python 编译依赖..."
    if [ "$EUID" -eq 0 ] || check_command sudo; then
        if [ "$EUID" -eq 0 ]; then
            apt-get update -qq || log_warn "apt-get update 失败，继续"
        fi
        local sudo_cmd=""
        [ "$EUID" -ne 0 ] && sudo_cmd="sudo"
        $sudo_cmd apt-get install -y build-essential zlib1g-dev libncurses5-dev \
            libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev \
            libsqlite3-dev wget 2>/dev/null || \
            log_warn "安装编译依赖失败，请确保已安装必要依赖"
    else
        log_warn "无 sudo 权限，跳过依赖安装，请确保已安装必要依赖"
    fi

    cd "$PYTHON_INSTALL_DIR" || error_exit "无法进入目录: $PYTHON_INSTALL_DIR"

    local python_file="Python-${PYTHON_VERSION}.tar.xz"
    local downloads_file="$DOWNLOADS_DIR/$python_file"

    # 检查下载的文件是否存在
    if [ ! -f "$downloads_file" ]; then
        error_exit "Python 安装文件不存在: $downloads_file" \
            "请先运行下载步骤或手动下载到该位置"
    fi

    # 从 downloads 目录复制文件
    log_info "准备 Python 安装文件..."
    if [ ! -f "$python_file" ] || [ "$downloads_file" -nt "$python_file" ]; then
        cp "$downloads_file" "$python_file" || error_exit "复制 Python 文件失败"
    fi

    # 解压 Python 源码
    log_info "解压 Python 源码..."
    if [ ! -d "$python_dir" ]; then
        xz -d "$python_file" || error_exit "解压 Python 源码失败 (xz)"
        tar -xf "Python-${PYTHON_VERSION}.tar" || error_exit "解压 Python 源码失败 (tar)"
        rm -f "Python-${PYTHON_VERSION}.tar"
    else
        log_info "Python 已解压，跳过"
    fi

    # 编译安装 Python
    cd "$python_dir" || error_exit "无法进入目录: $python_dir"

    # 检查是否已编译（通过检查编译产物）
    local compiled_marker="$python_dir/libpython3.11.so"
    if [ -f "$compiled_marker" ] && [ -f "$python_dir/python" ]; then
        log_info "Python 已编译完成，跳过 configure 和 make"
    else
        # 需要编译
        log_info "配置 Python 编译选项..."
        local configure_cmd="./configure --enable-optimizations --with-lto --enable-shared"

        # 使用 sudo 执行 configure（如果需要）
        if [ "$EUID" -ne 0 ] && check_command sudo; then
            log_info "使用 sudo 执行 configure..."
            sudo $configure_cmd || \
                error_exit "Python 配置失败" \
                    "1. 检查编译依赖是否完整\n2. 查看配置日志: $python_dir/config.log"
        else
            $configure_cmd || \
                error_exit "Python 配置失败" \
                    "1. 检查编译依赖是否完整\n2. 查看配置日志: $python_dir/config.log"
        fi

        log_info "编译 Python (使用 $MAKE_JOBS 个并行任务)..."
        make -j "$MAKE_JOBS" || error_exit "Python 编译失败" \
            "1. 减少并行任务数: export MAKE_JOBS=8\n2. 检查磁盘空间\n3. 查看编译错误"
    fi

    # 设置环境变量
    export PATH="$python_dir:$PATH"
    export LD_LIBRARY_PATH="$python_dir:$LD_LIBRARY_PATH"

    # 使用编译目录中的 Python
    PYTHON_CMD="$python_dir/python"

    # 验证安装
    log_info "验证 Python 安装..."
    if ! $PYTHON_CMD --version; then
        error_exit "Python 安装验证失败" "检查 PATH 和 LD_LIBRARY_PATH 配置"
    fi

    log_info "Python 安装成功"
    mark_step_done "install_python"
}

#################### 配置环境变量 ####################
setup_env() {
    log_step "=== 配置环境变量 ==="

    # 直接执行 export
    export JAVA_HOME="$JAVA_INSTALL_DIR/jdk-$JAVA_VERSION"
    export PATH="$JAVA_HOME/bin:$PATH"

    export PATH="$BAZEL_INSTALL_DIR:$PATH"

    if [ "$SKIP_PYTHON_INSTALL" != "1" ]; then
        export PATH="$PYTHON_INSTALL_DIR/Python-$PYTHON_VERSION:$PATH"
        export LD_LIBRARY_PATH="$PYTHON_INSTALL_DIR/Python-$PYTHON_VERSION:$LD_LIBRARY_PATH"
    fi

    # 生成环境变量配置文件
    cat > "$ENV_FILE" << EOF
# XLA-NPU 环境变量配置
# 生成时间: $(date)
# 使用方法: source build/xla_npu_env

export JAVA_HOME=$JAVA_INSTALL_DIR/jdk-$JAVA_VERSION
export PATH=\$JAVA_HOME/bin:\$PATH

export PATH=$BAZEL_INSTALL_DIR:\$PATH
EOF

    if [ "$SKIP_PYTHON_INSTALL" != "1" ]; then
        cat >> "$ENV_FILE" << EOF

export PATH=$PYTHON_INSTALL_DIR/Python-$PYTHON_VERSION:\$PATH
export LD_LIBRARY_PATH=$PYTHON_INSTALL_DIR/Python-$PYTHON_VERSION:\$LD_LIBRARY_PATH
EOF
    fi

    log_info "环境变量已设置"
    if [ "$SKIP_PYTHON_INSTALL" = "1" ]; then
        log_info "使用系统/conda Python"
    fi

    log_info "环境变量配置文件: $ENV_FILE"
}

#################### 下载代码 ####################
download_code() {
    # 检查是否已完成
    if skip_if_done "download_code" "下载代码"; then
        return 0
    fi

    log_step "=== 下载代码 ==="

    ensure_dir "$CODE_DIR"
    cd "$CODE_DIR" || error_exit "无法进入目录: $CODE_DIR"

    # 检查 xla-npu 目录（用户已克隆）
    if [ ! -d "$XLA_NPU_CODE_DIR/xla_npu" ]; then
        error_exit "xla-npu 代码目录不存在" \
            "请确保 xla-npu 已克隆到: $XLA_NPU_CODE_DIR\n运行: git clone https://gitcode.com/cann/xla-npu.git $XLA_NPU_CODE_DIR"
    fi
    log_info "检测到 xla-npu 代码: $XLA_NPU_CODE_DIR"

    # 下载 XLA 代码
    if [ -d "$XLA_CODE_DIR" ]; then
        log_info "XLA 代码目录已存在，更新代码..."
        cd "$XLA_CODE_DIR" || error_exit "无法进入目录: $XLA_CODE_DIR"
        git fetch || log_warn "git fetch 失败，继续使用本地代码"
    else
        log_info "克隆 XLA 代码仓库..."
        git clone https://github.com/openxla/xla.git "$XLA_CODE_DIR" || \
            error_exit "克隆 XLA 失败" \
                "1. 检查网络连接\n2. 配置代理: git config --global http.proxy <proxy>\n3. 使用镜像源"
        cd "$XLA_CODE_DIR" || error_exit "无法进入目录: $XLA_CODE_DIR"
    fi

    # 切换到指定 commit
    log_info "切换 XLA 到 commit: $XLA_COMMIT_ID"
    git checkout "$XLA_COMMIT_ID" || \
        error_exit "切换 XLA commit 失败" \
            "检查 commit ID 是否正确: $XLA_COMMIT_ID"

    # 创建软链接：XLA/xla/xla_npu -> xla-npu/xla_npu
    log_info "创建 xla_npu 软链接..."
    local link_target="$XLA_CODE_DIR/xla/xla_npu"

    if [ -L "$link_target" ]; then
        log_info "软链接已存在，删除旧链接..."
        rm -f "$link_target"
    fi

    ln -s "$XLA_NPU_CODE_DIR/xla_npu" "$link_target" || \
        error_exit "创建软链接失败" \
            "检查源路径: $XLA_NPU_CODE_DIR/xla_npu\n检查目标路径: $link_target"

    log_info "代码准备完成"
    mark_step_done "download_code"
}

#################### 编译 XLA ####################
build_xla() {
    # 检查是否已完成
    if skip_if_done "build_xla" "编译 XLA-NPU"; then
        return 0
    fi

    log_step "=== 编译 XLA-NPU ==="

    cd "$XLA_CODE_DIR" || error_exit "无法进入目录: $XLA_CODE_DIR"

    # 配置 XLA (检查是否已配置)
    local xla_config_marker="$XLA_CODE_DIR/.xla_configured_cpu"
    if [ -f "$xla_config_marker" ] && [ "$FORCE_REINSTALL" != "1" ]; then
        log_info "XLA 已配置 (backend=CPU)，跳过配置步骤"
    else
        log_info "配置 XLA (backend=CPU)..."
        $PYTHON_CMD ./configure.py --backend=CPU || \
            error_exit "XLA 配置失败" \
                "1. 检查 Python 版本 >= 3.11\n2. 检查 JDK 版本\n3. 查看配置日志"
        # 标记配置完成
        touch "$xla_config_marker"
        log_info "XLA 配置完成"
    fi

    # 编译
    log_info "开始编译 XLA-NPU (这可能需要很长时间)..."
    log_info "编译任务数: $PARALLEL_JOBS"

    bazel build \
        --incompatible_disallow_empty_glob=false \
        --remote_download_minimal \
        --spawn_strategy=local \
        --verbose_failures \
        --copt=-w \
        --jobs=$PARALLEL_JOBS \
        //xla/xla_npu/pjrt/c:pjrt_c_api_npu_plugin.so || \
        error_exit "XLA-NPU 编译失败" \
            "1. 检查网络连接 (需要下载依赖)\n2. 查看编译错误日志\n3. 尝试减少并行任务数\n4. 使用代理加速下载"

    log_info "XLA-NPU 编译成功"
    log_info "产物位置: $XLA_CODE_DIR/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so"
    mark_step_done "build_xla"
}

#################### 编译安装 JAX ####################
install_jax() {
    # 检查是否已完成
    if skip_if_done "install_jax" "编译安装 JAX"; then
        return 0
    fi

    log_step "=== 编译安装 JAX ==="

    ensure_dir "$CODE_DIR"
    cd "$CODE_DIR" || error_exit "无法进入目录: $CODE_DIR"

    # 下载 JAX 代码
    if [ -d "$JAX_CODE_DIR" ]; then
        log_info "JAX 代码目录已存在，更新代码..."
        cd "$JAX_CODE_DIR" || error_exit "无法进入目录: $JAX_CODE_DIR"
        git fetch || log_warn "git fetch 失败，继续使用本地代码"
    else
        log_info "克隆 JAX 代码仓库..."
        git clone https://github.com/jax-ml/jax.git "$JAX_CODE_DIR" || \
            error_exit "克隆 JAX 失败" \
                "1. 检查网络连接\n2. 配置代理\n3. 使用镜像源"
        cd "$JAX_CODE_DIR" || error_exit "无法进入目录: $JAX_CODE_DIR"
    fi

    # 切换到指定版本
    log_info "切换 JAX 到版本: $JAX_VERSION"
    git checkout "$JAX_VERSION" || \
        error_exit "切换 JAX 版本失败" \
            "检查版本标签是否存在: git tag -l | grep $JAX_VERSION"

    # 编译 JAX
    log_info "编译 JAX 和 jaxlib (使用本地 XLA)..."
    $PYTHON_CMD build/build.py build --wheels=jaxlib,jax --local_xla_path="$XLA_CODE_DIR" || \
        error_exit "JAX 编译失败" \
            "1. 检查 XLA 编译产物\n2. 检查 Python 环境\n3. 查看编译错误日志"

    # 卸载旧版本 (如果存在)
    log_info "卸载旧版本 JAX..."
    $PYTHON_CMD -m pip uninstall -y jax jaxlib 2>/dev/null || true

    # 安装 jaxlib
    log_info "安装 jaxlib..."
    local jaxlib_wheel=$(find "$JAX_CODE_DIR/dist" -name "jaxlib-*.whl" | head -1)
    if [ -z "$jaxlib_wheel" ]; then
        error_exit "找不到 jaxlib wheel 文件" "检查编译产物目录: $JAX_CODE_DIR/dist"
    fi

    $PYTHON_CMD -m pip install "$jaxlib_wheel" || \
        error_exit "安装 jaxlib 失败" \
            "1. 检查 pip 版本\n2. 使用 sudo 安装\n3. 检查 Python 环境"

    # 安装 JAX
    log_info "安装 JAX..."
    cd "$JAX_CODE_DIR" || error_exit "无法进入目录: $JAX_CODE_DIR"
    $PYTHON_CMD -m pip install -e . --no-deps || \
        error_exit "安装 JAX 失败" "检查 pip 和 Python 环境"

    # 安装可能的依赖
    log_info "安装额外依赖..."
    $PYTHON_CMD -m pip install ml_dtypes opt_einsum || \
        log_warn "安装额外依赖失败，可能需要手动安装"

    log_info "JAX 安装成功"
    mark_step_done "install_jax"
}

#################### 生成安装文档 ####################
generate_documentation() {
    log_step "=== 生成安装文档 ==="

    cat > "$INSTALL_DOC" << 'EOF'
# XLA-NPU 安装文档

## 环境配置

### 安装目录
- 工作空间: `WORKSPACE`
- 环境目录: `ENV_DIR`
- 代码目录: `CODE_DIR`

### 环境变量

安装脚本已自动生成环境变量配置文件: `build/xla_npu_env`

要使环境变量生效，请执行:

```bash
source build/xla_npu_env
```

或者添加到 ~/.bashrc 以自动加载:

```bash
echo 'source $(pwd)/build/xla_npu_env' >> ~/.bashrc
source ~/.bashrc
```

### 验证安装

验证 Java:
```bash
java --version
```

验证 Bazel:
```bash
bazel --version
```

验证 Python:
```bash
python --version
```

验证 JAX:
```bash
python -c "import jax; print(jax.__version__)"
```

## 编译产物

### XLA-NPU 产物
- 位置: `bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so`
- 可用于 JAX 插件加载

### JAX 产物
- jaxlib wheel 包
- JAX 可编辑安装

## 常见问题

### 1. 网络问题
**问题**: 下载失败或速度慢

**解决方案**:
- 使用代理
- 配置 Git 代理: `git config --global http.proxy <proxy_url>`
- 使用国内镜像源

### 2. 编译错误
**问题**: 编译过程中出现错误

**解决方案**:
- 减少并行任务数: `export PARALLEL_JOBS=8`
- 检查磁盘空间
- 查看详细日志

### 3. Python 版本问题
**问题**: Python 版本不满足要求

**解决方案**:
- XLA-JAX 需要 Python >= 3.11
- 使用 conda: `conda install python=3.11`
- 或使用脚本自动安装 Python 3.11.9 (设置 SKIP_PYTHON_INSTALL=0)

### 4. 权限问题
**问题**: 安装时权限不足

**解决方案**:
- 使用 sudo 运行脚本
- 或确保安装目录有写权限

## 卸载

### 自动卸载
```bash
cd /home/code/xla/install
./build.sh uninstall
```

### 手动卸载

#### 删除环境变量
```bash
# 删除环境变量配置文件
rm build/xla_npu_env

# 如果已添加到 ~/.bashrc，编辑删除相关行
# 或执行：
sed -i '/xla_npu_env/d' ~/.bashrc
```

#### 删除代码和编译产物
```bash
rm -rf $CODE_DIR/xla
rm -rf $CODE_DIR/xla-npu
rm -rf $CODE_DIR/jax
```

#### 卸载 JAX
```bash
python -m pip uninstall -y jax jaxlib
```

#### 清理缓存
```bash
# 清理 Bazel 缓存
rm -rf ~/.cache/bazel

# 清理 pip 缓存
pip cache purge
```

## 日志文件

安装日志: `LOG_FILE`

## 支持

如有问题，请查看:
- 安装指南: `/home/code/xla/install/install-guide.txt`
- 日志文件: `LOG_FILE`
EOF

    # 替换变量
    sed -i "s|\$WORKSPACE|$WORKSPACE|g" "$INSTALL_DOC"
    sed -i "s|\$ENV_DIR|$ENV_DIR|g" "$INSTALL_DOC"
    sed -i "s|\$CODE_DIR|$CODE_DIR|g" "$INSTALL_DOC"
    sed -i "s|\$LOG_FILE|$LOG_FILE|g" "$INSTALL_DOC"

    log_info "安装文档已生成: $INSTALL_DOC"
}

#################### 卸载功能 ####################
uninstall_jax() {
    log_step "=== 卸载 JAX ==="

    PYTHON_CMD=$(check_python)
    if [ -n "$PYTHON_CMD" ]; then
        # 检查是否安装了 JAX
        if $PYTHON_CMD -c "import jax" 2>/dev/null; then
            log_info "检测到 JAX 已安装，开始卸载..."

            # 卸载 JAX 和 jaxlib
            log_info "卸载 jax 和 jaxlib..."
            $PYTHON_CMD -m pip uninstall -y jax jaxlib 2>/dev/null || \
                log_warn "卸载 JAX 失败，可能需要手动卸载"

            log_info "JAX 卸载完成"
        else
            log_info "JAX 未安装，跳过卸载"
        fi
    else
        log_warn "Python 未找到，跳过 JAX 卸载"
    fi

    # 删除 JAX 代码目录
    if [ -d "$JAX_CODE_DIR" ]; then
        if confirm_action "是否删除 JAX 代码目录 ($JAX_CODE_DIR)?"; then
            log_info "删除 JAX 代码目录..."
            rm -rf "$JAX_CODE_DIR" || log_warn "删除 JAX 代码目录失败"
        else
            log_info "保留 JAX 代码目录"
        fi
    fi
}

uninstall_xla_npu() {
    log_step "=== 卸载 XLA-NPU ==="

    # 删除 XLA 编译产物
    if [ -d "$XLA_CODE_DIR/bazel-bin" ]; then
        if confirm_action "是否删除 XLA 编译产物?"; then
            log_info "删除 XLA 编译产物..."
            cd "$XLA_CODE_DIR" || log_warn "无法进入 XLA 目录"
            bazel clean --expunge 2>/dev/null || rm -rf bazel-* || \
                log_warn "清理 Bazel 缓存失败"
        else
            log_info "保留 XLA 编译产物"
        fi
    fi

    # 删除 XLA 代码目录
    if [ -d "$XLA_CODE_DIR" ]; then
        if confirm_action "是否删除 XLA 代码目录 ($XLA_CODE_DIR)?"; then
            log_info "删除 XLA 代码目录..."
            rm -rf "$XLA_CODE_DIR" || log_warn "删除 XLA 代码目录失败"
            # 配置标记文件会随代码目录一起删除，无需单独处理
        else
            log_info "保留 XLA 代码目录"
            # 如果保留代码目录，清除配置标记以便下次重新配置
            rm -f "$XLA_CODE_DIR/.xla_configured_cpu" 2>/dev/null || true
        fi
    fi

    # 注意：不删除 xla-npu 代码目录，这是用户的代码仓库
    log_info "保留 xla-npu 代码目录 ($XLA_NPU_CODE_DIR)"
}

uninstall_env() {
    log_step "=== 卸载环境配置 ==="

    # 删除环境变量配置文件
    if [ -f "$ENV_FILE" ]; then
        if confirm_action "是否删除环境变量配置文件 ($ENV_FILE)?"; then
            log_info "删除环境变量配置文件..."
            rm -f "$ENV_FILE" || log_warn "删除环境变量文件失败"
        else
            log_info "保留环境变量配置文件"
        fi
    fi

    # 提示用户手动清除 ~/.bashrc 中的配置（如果添加了）
    if grep -q "xla_npu_env" "$HOME/.bashrc" 2>/dev/null; then
        log_info "检测到 ~/.bashrc 中包含 xla_npu_env 配置"
        log_info "请手动编辑 ~/.bashrc 删除相关行，或执行："
        log_info "  sed -i '/xla_npu_env/d' ~/.bashrc"
    fi
}

uninstall_tools() {
    log_step "=== 卸载工具 (可选) ==="

    # 询问是否删除编译工具
    if confirm_action "是否删除编译工具 (Java, Bazel)?"; then
        # 删除 Java
        if [ -d "$JAVA_INSTALL_DIR" ]; then
            log_info "删除 Java..."
            rm -rf "$JAVA_INSTALL_DIR" || log_warn "删除 Java 失败"
        fi

        # 删除 Bazel
        if [ -d "$BAZEL_INSTALL_DIR" ]; then
            log_info "删除 Bazel..."
            rm -rf "$BAZEL_INSTALL_DIR" || log_warn "删除 Bazel 失败"
        fi

        log_info "注意: conda Python 未被删除"
    else
        log_info "保留编译工具"
    fi
}

uninstall_cache() {
    log_step "=== 清理缓存 ==="

    if confirm_action "是否清理缓存 (Bazel, pip)?"; then
        # 清理 Bazel 缓存
        local bazel_cache="$HOME/.cache/bazel"
        if [ -d "$bazel_cache" ]; then
            log_info "清理 Bazel 缓存..."
            rm -rf "$bazel_cache" || log_warn "清理 Bazel 缓存失败"
        fi

        # 清理 pip 缓存
        PYTHON_CMD=$(check_python)
        if [ -n "$PYTHON_CMD" ]; then
            log_info "清理 pip 缓存..."
            $PYTHON_CMD -m pip cache purge 2>/dev/null || \
                log_warn "清理 pip 缓存失败"
        fi

        # 清理下载临时文件
        local temp_download_dir="$DOWNLOADS_DIR"
        if [ -d "$temp_download_dir" ]; then
            log_info "清理下载临时文件..."
            rm -rf "$temp_download_dir" || log_warn "清理下载临时文件失败"
        fi
    else
        log_info "保留缓存"
    fi
}

uninstall_all() {
    echo -e "\033[35m========================================\033[0m"
    echo -e "\033[35m  XLA-NPU 卸载程序\033[0m"
    echo -e "\033[35m========================================\033[0m"
    echo ""
    log_warn "此操作将卸载 XLA-NPU 及相关组件"
    echo ""

    # 创建卸载日志
    local uninstall_log="$LOG_DIR/uninstall_$(date +%Y%m%d_%H%M%S).log"
    LOG_FILE="$uninstall_log"

    log_info "卸载日志: $uninstall_log"
    echo ""

    # 确认卸载
    if ! confirm_action "确认要卸载 XLA-NPU 吗？"; then
        log_info "取消卸载"
        exit 0
    fi

    echo ""

    # 按顺序卸载
    uninstall_jax
    uninstall_xla_npu
    uninstall_env

    # 可选卸载
    echo ""
    uninstall_tools
    uninstall_cache

    # 清除安装状态文件
    log_info "清除安装状态..."
    rm -f "$STATE_FILE" 2>/dev/null || true

    echo ""
    echo -e "\033[32m========================================\033[0m"
    echo -e "\033[32m  卸载完成!\033[0m"
    echo -e "\033[32m========================================\033[0m"
    echo ""
    echo "请执行以下命令重新加载环境变量:"
    echo -e "\033[36msource ~/.bashrc\033[0m"
    echo ""
    echo "卸载日志: $uninstall_log"
    echo ""
}

#################### 显示帮助 ####################
show_help() {
    cat << EOF
XLA-NPU 安装脚本

使用方法:
    $0 [命令] [选项]

命令:
    install    - 执行完整安装 (默认)
    uninstall  - 卸载 XLA-NPU
    help       - 显示帮助信息

环境变量:
    WORKSPACE          - 工作目录 (默认: 脚本所在目录/workspace)
    PARALLEL_JOBS      - 并行任务数 (默认: nproc)
    MAKE_JOBS          - make 并行数 (默认: PARALLEL_JOBS)
    INSTALL_JAX        - 是否安装 JAX (默认: 1, 设为 0 跳过)
    SKIP_PYTHON_INSTALL- 跳过 Python 安装，使用系统/conda Python (默认: 0, 设为 1 启用)
    FORCE_REINSTALL    - 强制重新安装所有步骤 (默认: 0, 设为 1 启用)
    FORCE_UNINSTALL    - 强制卸载，无需确认 (默认: 0, 设为 1 启用)
    SKIP_DOWNLOAD      - 跳过下载步骤 (默认: 0)

示例:
    # 完整安装 (包含 JAX)，使用 conda Python
    SKIP_PYTHON_INSTALL=1 $0 install

    # 完整安装 (包含 JAX)，编译 Python
    $0 install

    # 完整安装 (不包含 JAX)
    INSTALL_JAX=0 $0 install

    # 强制重新安装所有步骤 (跳过断点续传)
    FORCE_REINSTALL=1 $0 install

    # 自定义工作目录
    WORKSPACE=/opt/xla $0 install

    # 自定义并行数
    PARALLEL_JOBS=16 $0 install

    # 卸载 XLA-NPU
    $0 uninstall

    # 强制卸载 (无需确认)
    FORCE_UNINSTALL=1 $0 uninstall
EOF
}

#################### 主流程 ####################
main_install() {
    echo -e "\033[35m========================================\033[0m"
    echo -e "\033[35m  XLA-NPU 自动安装脚本\033[0m"
    echo -e "\033[35m========================================\033[0m"
    echo ""

    # 初始化环境
    init_environment

    # 并行下载可以并行的部分
    log_step "=== 并行下载安装包 ==="

    # 创建临时目录用于下载
    local temp_download_dir="$DOWNLOADS_DIR"
    ensure_dir "$temp_download_dir"
    cd "$temp_download_dir" || error_exit "无法进入下载目录"

    # 检查哪些文件需要下载（检查 downloads 目录）
    local need_jdk=0
    local need_bazel=0
    local need_python=0

    local jdk_file="openjdk-${JAVA_VERSION}_linux-x64_bin.tar.gz"
    local bazel_file="bazel_nojdk-${BAZEL_VERSION}-linux-x86_64"
    local python_file="Python-${PYTHON_VERSION}.tar.xz"

    # 检查 JDK（检查 downloads 目录）
    if [ ! -f "$temp_download_dir/$jdk_file" ] && [ "$FORCE_REINSTALL" != "1" ]; then
        need_jdk=1
    elif [ "$FORCE_REINSTALL" = "1" ]; then
        need_jdk=1
    else
        log_info "JDK 文件已存在，跳过下载: $jdk_file"
    fi

    # 检查 Bazel（检查 downloads 目录）
    if [ ! -f "$temp_download_dir/$bazel_file" ] && [ "$FORCE_REINSTALL" != "1" ]; then
        need_bazel=1
    elif [ "$FORCE_REINSTALL" = "1" ]; then
        need_bazel=1
    else
        log_info "Bazel 文件已存在，跳过下载: $bazel_file"
    fi

    # 检查 Python（检查 downloads 目录）
    if [ "$SKIP_PYTHON_INSTALL" != "1" ]; then
        if [ ! -f "$temp_download_dir/$python_file" ] && [ "$FORCE_REINSTALL" != "1" ]; then
            need_python=1
        elif [ "$FORCE_REINSTALL" = "1" ]; then
            need_python=1
        else
            log_info "Python 文件已存在，跳过下载: $python_file"
        fi
    fi

    # 准备并行下载任务
    local download_count=0
    if [ $need_jdk -eq 1 ]; then
        log_info "准备下载 JDK..."
        download_count=$((download_count + 1))
    fi
    if [ $need_bazel -eq 1 ]; then
        log_info "准备下载 Bazel..."
        download_count=$((download_count + 1))
    fi
    if [ $need_python -eq 1 ]; then
        log_info "准备下载 Python 源码..."
        download_count=$((download_count + 1))
    fi

    if [ $download_count -eq 0 ]; then
        log_info "所有安装包已存在，跳过下载步骤"
    else
        log_info "开始下载 $download_count 个文件..."

        # 启动并行下载 (直接后台运行，只下载需要的文件)
        local jdk_pid=""
        local bazel_pid=""
        local python_pid=""

        if [ $need_jdk -eq 1 ]; then
            wget --no-check-certificate -c --show-progress \
                "$JAVA_MIRROR/${JAVA_VERSION}/$jdk_file" \
                -O "$jdk_file" &
            jdk_pid=$!
        fi

        if [ $need_bazel -eq 1 ]; then
            wget --no-check-certificate -c --show-progress \
                "$BAZEL_MIRROR/${BAZEL_VERSION}/$bazel_file" \
                -O "$bazel_file" &
            bazel_pid=$!
        fi

        if [ $need_python -eq 1 ]; then
            wget --no-check-certificate -c --show-progress \
                "$PYTHON_MIRROR/${PYTHON_VERSION}/$python_file" \
                -O "$python_file" &
            python_pid=$!
        fi

        # 等待所有下载完成
        log_info "等待下载完成..."
        [ -n "$jdk_pid" ] && wait $jdk_pid 2>/dev/null || true
        [ -n "$bazel_pid" ] && wait $bazel_pid 2>/dev/null || true
        [ -n "$python_pid" ] && wait $python_pid 2>/dev/null || true

        # 额外等待确保文件写入完成
        sleep 1
    fi

    # 检查下载结果
    local download_failed=0
    if [ $need_jdk -eq 1 ]; then
        if [ ! -f "$jdk_file" ] || [ ! -s "$jdk_file" ]; then
            log_error "下载失败: $jdk_file"
            download_failed=1
        else
            log_info "下载成功: $jdk_file"
        fi
    fi

    if [ $need_bazel -eq 1 ]; then
        if [ ! -f "$bazel_file" ] || [ ! -s "$bazel_file" ]; then
            log_error "下载失败: $bazel_file"
            download_failed=1
        else
            log_info "下载成功: $bazel_file"
        fi
    fi

    if [ $need_python -eq 1 ]; then
        if [ ! -f "$python_file" ] || [ ! -s "$python_file" ]; then
            log_error "下载失败: $python_file"
            download_failed=1
        else
            log_info "下载成功: $python_file"
        fi
    fi

    if [ $download_failed -eq 1 ]; then
        error_exit "部分文件下载失败" \
            "检查网络连接或手动下载文件到对应目录"
    fi

    # 安装组件
    install_java
    install_bazel
    install_python
    setup_env

    # 下载代码
    download_code

    # 编译 XLA
    build_xla

    # 编译安装 JAX (默认安装)
    if [ "$INSTALL_JAX" = "1" ]; then
        install_jax
    else
        log_info "跳过 JAX 安装 (INSTALL_JAX=0)"
    fi

    # 生成文档
    generate_documentation

    echo ""
    echo -e "\033[32m========================================\033[0m"
    echo -e "\033[32m  安装完成!\033[0m"
    echo -e "\033[32m========================================\033[0m"
    echo ""
    echo "请执行以下命令使环境变量生效:"
    echo -e "\033[36msource build/xla_npu_env\033[0m"
    echo ""
    echo "或者添加到 ~/.bashrc 以自动加载:"
    echo -e "\033[36mecho 'source $(pwd)/build/xla_npu_env' >> ~/.bashrc\033[0m"
    echo ""
    echo "安装文档: $INSTALL_DOC"
    echo "安装日志: $LOG_FILE"
    echo ""
}

main() {
    local action="${1:-install}"

    case "$action" in
        install|"")
            main_install
            ;;
        uninstall)
            uninstall_all
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "未知命令: $action"
            echo "运行 '$0 help' 查看帮助"
            exit 1
            ;;
    esac
}

# 捕获错误
trap 'log_error "脚本执行失败 (行号: $LINENO)"; exit 1' ERR

# 执行主流程
main "$@"
