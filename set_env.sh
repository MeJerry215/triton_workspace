#! /bin/bash

# 脚本所在目录（兼容 git clone 到任意位置）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"

export TRITON_ALWAYS_COMPILE=0
export TRITON_CACHE_DIR="$SCRIPT_DIR/cache"
export MLIR_ENABLE_DUMP=1
export MLIR_DUMP_PATH="$SCRIPT_DIR/dump.log"
export TRITON_DUMP_AST_PATH="$SCRIPT_DIR/dump.ast"
export LLVM_IR_ENABLE_DUMP=0
export TRITON_KERNEL_DUMP=0
export TRITON_DUMP_DIR="$SCRIPT_DIR/dump"
export PATH="$SCRIPT_DIR/llvm-project/build/bin/:$PATH"
