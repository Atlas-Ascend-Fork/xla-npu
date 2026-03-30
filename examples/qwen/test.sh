#!/bin/bash
# Qwen2 测试脚本 - 参考 xla-verify/test.sh 实现

set -e

XLA_NPU_ENV="${WORKSPACE_DIR}/code/xla-npu/build/xla_npu_env"

if [ ! -f "$XLA_NPU_ENV" ]; then
    echo "错误: 找不到 xla-npu 环境文件: $XLA_NPU_ENV"
    echo "请先运行 build.sh 进行构建"
    exit 1
fi

echo "=== Qwen2 Test Script ==="
echo "Loading environment from: $XLA_NPU_ENV"
echo ""

source "$XLA_NPU_ENV"

# 使用 /opt/buildtools/Python-3.11.4/bin/python3.11 (参考 xla-verify 使用系统 Python)
PYTHON_CMD="/opt/buildtools/Python-3.11.4/bin/python3.11"

if [ ! -x "$PYTHON_CMD" ]; then
    echo "错误: 找不到 Python: $PYTHON_CMD"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo "Python version: $($PYTHON_CMD --version)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/main.py"
LOG_FILE="${WORKSPACE_DIR}/user-workspace/log/qwen-test.log"

echo "Test script: $TEST_SCRIPT"
echo "Log file: $LOG_FILE"
echo ""

# 创建日志目录
mkdir -p "$(dirname "$LOG_FILE")"

# 第一步：初始化参数
echo "Step 1: Initializing model parameters..."
cd "$SCRIPT_DIR"
$PYTHON_CMD "$TEST_SCRIPT" --init > "$LOG_FILE" 2>&1
echo "✓ Parameters initialized"
echo ""

# 第二步：运行 CPU 测试
echo "Step 2: Running CPU inference..."
cd "$SCRIPT_DIR"
$PYTHON_CMD "$TEST_SCRIPT" --skip-npu >> "$LOG_FILE" 2>&1
echo "✓ CPU inference completed"
echo ""

echo "=== Test Complete ==="
echo "Log saved to: $LOG_FILE"
echo ""

if [ -f "$LOG_FILE" ]; then
    echo "✓ Log file created"
    echo ""
    echo "=== Last 30 lines of log ==="
    tail -30 "$LOG_FILE"
else
    echo "✗ Log file NOT found"
    exit 1
fi
