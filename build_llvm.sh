#! /bin/bash
# 参考 markdowns/comple.md：在本地构建 LLVM/MLIR（Ninja + Release + Assertions）。
# 可通过环境变量覆盖路径，例如: LLVM_SRC=/path/to/llvm-project ./build_llvm.sh

set -euo pipefail
export MAX_JOBS=64

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LLVM_SRC="${LLVM_SRC:-$ROOT/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$LLVM_SRC/build}"

if [[ ! -d "$LLVM_SRC/llvm" ]]; then
  echo "error: LLVM 源码目录不存在或无效: $LLVM_SRC (期望存在 $LLVM_SRC/llvm)" >&2
  exit 1
fi

mkdir -p "$LLVM_BUILD_DIR"
cd "$LLVM_BUILD_DIR"

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  "$LLVM_SRC/llvm" \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"

if [[ -n "${MAX_JOBS:-}" ]]; then
  ninja -j"$MAX_JOBS"
else
  ninja
fi

echo "LLVM 构建完成。请设置并导出:"
echo "  export LLVM_BUILD_DIR=$LLVM_BUILD_DIR"
