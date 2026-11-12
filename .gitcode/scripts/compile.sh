#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set +e
function LOG_DO() {
   local cmd="$*"
   date_time=$(date +%Y%m%d-%H%M%S)
   echo -e "[Command] ${date_time} ${cmd}"
   ${cmd}
}
function LOG_HEAD() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "[INFO] ${date_time} ${assert_msg}"
}
function DP_ASSERT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    local log_flag=${4:-"true"}
    local log_path=${5}
    if [ "${actual_value}" != "${expect_value}" ]; then
        if [ -n "${log_path}" ] && [ -f "${log_path}" ]; then
            cat ${log_path}
        fi
        echo "${assert_msg} is failed."
        exit 1
    else
        if [ "${log_flag}" = "true" ]; then
            echo "${assert_msg} is success."
        fi
    fi
}
echo "Build ${repo_name}."
LOG_HEAD "Build ${repo_name}."
cd "${WORKSPACE:?WORKSPACE is not set}" || exit 1
export INSTALL_DIR=/tmp/xla/workspace
apt-get update 
apt-get install -y ccache clang
echo "+++++++"
ls

if [ ! -s pr_filelist.txt ]; then
  echo "pr_filelist.txt is missing or empty, abort build."
  exit 1
fi

# 如果输出为空，说明全是 .md文件
grep -v "\.md$" pr_filelist.txt

# 直接判断并输出结果
if grep -vq "\.md$" pr_filelist.txt; then
  echo "存在非 .md 文件"
else
  echo "全部都是 .md 文件"
  mkdir -p build_out
  touch ./build_out/tmp.so
  exit 0
fi

if [[ "${task_name}" =~ API_Check ]]; then
  sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
fi

mkdir -p /tmp/ccache_wrappers /tmp/cc_probe
printf 'int main() {}\n' > /tmp/cc_probe/empty.cc
cat > /tmp/ccache_wrappers/clang << 'EOF'
#!/bin/bash
for arg in "$@"; do
  case "$arg" in
    *empty.cc) exec env -u LD_PRELOAD clang "$@" ;;
  esac
done
exec ccache clang "$@"
EOF
cat > /tmp/ccache_wrappers/clang++ << 'EOF'
#!/bin/bash
for arg in "$@"; do
  case "$arg" in
    *empty.cc) exec env -u LD_PRELOAD clang++ "$@" ;;
  esac
done
exec ccache clang++ "$@"
EOF
chmod +x /tmp/ccache_wrappers/clang /tmp/ccache_wrappers/clang++
export CC=/tmp/ccache_wrappers/clang
export CXX=/tmp/ccache_wrappers/clang++
echo "[cc-probe] lld : $($CC /tmp/cc_probe/empty.cc -o /dev/null -Wl,--start-lib -Wl,--end-lib -fuse-ld=lld -v 2>&1 >/dev/null | tail -1)"
echo "[cc-probe] gold: $($CC /tmp/cc_probe/empty.cc -o /dev/null -Wl,--start-lib -Wl,--end-lib -fuse-ld=gold -v 2>&1 >/dev/null | tail -1)"

CCACHE_REDIS_HOST=${CCACHE_REDIS_HOST:-10.0.0.136}
CCACHE_REDIS_PORT=${CCACHE_REDIS_PORT:-6380}
echo "[ccache] $(ccache --version | head -1)"
echo "[ccache] $(ccache --version | sed -n '/^Features:/p')"
echo "[ccache] inherited: CCACHE_DIR=${CCACHE_DIR:-<unset>} CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-<unset>} CCACHE_COMPILERCHECK=${CCACHE_COMPILERCHECK:-<unset>} CCACHE_NOHASHDIR=${CCACHE_NOHASHDIR:-<unset>} CCACHE_SECONDARY_STORAGE=${CCACHE_SECONDARY_STORAGE:-<unset>}"
if timeout 5 bash -c 'echo > /dev/tcp/$0/$1' "$CCACHE_REDIS_HOST" "$CCACHE_REDIS_PORT" 2>/dev/null; then
  echo "[ccache] redis ${CCACHE_REDIS_HOST}:${CCACHE_REDIS_PORT} reachable -> secondary storage ON, reshare ON"
  export CCACHE_SECONDARY_STORAGE="redis://${CCACHE_REDIS_HOST}:${CCACHE_REDIS_PORT}"
  export CCACHE_RESHARE=true
else
  unset CCACHE_SECONDARY_STORAGE CCACHE_RESHARE
  echo "[ccache] redis ${CCACHE_REDIS_HOST}:${CCACHE_REDIS_PORT} UNREACHABLE -> secondary storage OFF"
fi
export CCACHE_DIR=${CCACHE_DIR:-/tmp/ccache_xla}
export CCACHE_COMPILERCHECK=content
export CCACHE_NOHASHDIR=true
export CCACHE_SLOPPINESS=include_file_mtime,time_macros,include_file_ctime
export CCACHE_UMASK=002
export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-50G}
mkdir -p "${CCACHE_DIR}"
echo "[ccache] effective: CCACHE_DIR=${CCACHE_DIR} CCACHE_SECONDARY_STORAGE=${CCACHE_SECONDARY_STORAGE:-<none>} CCACHE_RESHARE=${CCACHE_RESHARE:-<unset>} CCACHE_COMPILERCHECK=${CCACHE_COMPILERCHECK}"
ccache -z

cat > ~/.bazelrc << 'EOF'
build --action_env=CC=/tmp/ccache_wrappers/clang
build --action_env=CXX=/tmp/ccache_wrappers/clang++
build --action_env=CCACHE_DIR
build --action_env=CCACHE_SECONDARY_STORAGE
build --action_env=CCACHE_RESHARE
build --action_env=CCACHE_COMPILERCHECK
build --action_env=CCACHE_SLOPPINESS
build --action_env=CCACHE_NOHASHDIR
build --action_env=CCACHE_MAXSIZE
build --action_env=CCACHE_UMASK
build --host_action_env=CCACHE_DIR
build --host_action_env=CCACHE_SECONDARY_STORAGE
build --host_action_env=CCACHE_RESHARE
build --host_action_env=CCACHE_COMPILERCHECK
build --host_action_env=CCACHE_SLOPPINESS
build --host_action_env=CCACHE_NOHASHDIR
build --host_action_env=CCACHE_MAXSIZE
build --host_action_env=CCACHE_UMASK
EOF

BAZELRC_FILE="${INSTALL_DIR}/build/code/xla/.bazelrc"
if [ -f "${BAZELRC_FILE}" ]; then
    sed -i 's/USE_HERMETIC_CC_TOOLCHAIN=1/USE_HERMETIC_CC_TOOLCHAIN=0/g' "${BAZELRC_FILE}"
    sed -i 's/--incompatible_enable_cc_toolchain_resolution/--noincompatible_enable_cc_toolchain_resolution/g' "${BAZELRC_FILE}"
fi

LOG_DO bash build/build.sh install --npu-auto-fuse-backend=${ge_st_rt2}
build_rc=$?
echo "[ccache] ================= stats ================="
ccache -s
echo "[ccache] ========================================="
DP_ASSERT_EQUAL "$build_rc" "0" "Build"
exit 0