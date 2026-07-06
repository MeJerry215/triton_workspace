#!/bin/bash
#
# TileLang 环境变量设置脚本
# 使用方式: source ops/tilelang/set_env.sh
#
# 说明: 只包含 TileLang 自身需要的环境变量。
#       TileLang 通过 pip install . -v --no-build-isolation 安装，
#       不需要额外设置 PYTHONPATH。
#

source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate triton

# 脚本所在目录（兼容 source 和直接执行两种方式）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"

# ===== CUDA 工具链 =====
# 优先使用系统 CUDA nvcc，避免 conda nvidia/cu13 的 nvcc header 不兼容问题
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"

# ===== TileLang 基础配置 =====
export TILELANG_PRINT_ON_COMPILATION=1        # kernel 编译时打印名称
export TILELANG_CACHE_DIR="$SCRIPT_DIR/.cache/tilelang"
export TILELANG_TMP_DIR="$TILELANG_CACHE_DIR/tmp"
export TILELANG_KERNEL_CACHE_USE_LIB_STAMP=0  # 缓存键是否包含 lib 哈希
export TILELANG_DISABLE_CACHE=0               # 禁用缓存（调试用）
export TILELANG_CLEANUP_TEMP_FILES=1          # 编译后清理临时文件
export TILELANG_JIT_DIAGNOSTICS=0             # JIT 阶段诊断

# ===== TileLang 目标设备 =====
# "auto" 自动检测，也可指定 "cuda -arch=sm_90"、{"kind":"cuda","arch":"sm_90"} 等
export TILELANG_DEFAULT_TARGET="auto"
# export TILELANG_EXECUTION_BACKEND="auto"

# ===== TileLang 编译超时 =====
# 空字符串表示不限时；单位秒
export TILELANG_COMPILE_TIMEOUT_SECONDS=""

# ===== TileLang Pass 调试 =====
# 0=off, terminal=终端彩色diff, html=HTML报告, both=两者
export TILELANG_PASS_DIFF=0
export TILELANG_PASS_DIFF_OUTPUT="$SCRIPT_DIR/.pass_diff"

# ===== 第三方库路径（按需设置）=====
# TileLang 通常会自动检测 3rdparty 路径；若自动检测失败可手工指定
# export TL_CUTLASS_PATH="/path/to/cutlass/include"
# export TL_COMPOSABLE_KERNEL_PATH="/path/to/composable_kernel/include"

# ===== 打印当前配置 =====
echo "[TileLang Env]"
echo "  CUDA_HOME                      = $CUDA_HOME"
echo "  TILELANG_CACHE_DIR             = $TILELANG_CACHE_DIR"
echo "  TILELANG_DEFAULT_TARGET        = $TILELANG_DEFAULT_TARGET"
echo "  TILELANG_PRINT_ON_COMPILATION  = $TILELANG_PRINT_ON_COMPILATION"
echo "  TILELANG_DISABLE_CACHE         = $TILELANG_DISABLE_CACHE"
