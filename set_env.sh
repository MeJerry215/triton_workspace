#! /bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate triton

# 脚本所在目录（兼容 source 和直接执行两种方式）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"

export PYTHONPATH="$(cd "$SCRIPT_DIR/triton/python" && pwd -P)"

# ----- 编译控制 -----
export TRITON_ALWAYS_COMPILE=1         # 强制重编，避免 cache 跳过 pass pipeline

# ----- Pass 级 MLIR Dump (MLIR_ENABLE_DUMP) -----
export MLIR_ENABLE_DUMP=1              # dump 所有 kernel 的所有 pass
export MLIR_DUMP_PATH="$SCRIPT_DIR/dump.log"

# 可选：只 dump 指定 kernel（输出量小很多）
# export MLIR_ENABLE_DUMP=memcpy_kernel

# ----- Stage 级 Dump (TRITON_KERNEL_DUMP) -----
export TRITON_KERNEL_DUMP=0
export TRITON_DUMP_DIR="$SCRIPT_DIR/dump"

# ----- AST Dump -----
export TRITON_DUMP_AST_PATH="$SCRIPT_DIR/dump.ast"

# ----- LLVM IR Dump -----
export LLVM_IR_ENABLE_DUMP=0

# ----- Triton Cache -----
export TRITON_CACHE_DIR="$SCRIPT_DIR/cache"

# ----- LLVM 工具链 -----
export PATH="$(cd "$SCRIPT_DIR/llvm-project/build/bin" && pwd -P):$PATH"
