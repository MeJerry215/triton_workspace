#! /bin/bash
# 参考 markdowns/comple.md：在已构建的 LLVM 上 pip 可编辑安装 Triton。
# 先 source 或执行 set_env.sh 中的 TRITON_HOME 等；本脚本会 source 同目录下的 set_env.sh。
# 示例: LLVM_BUILD_DIR=/path/to/build ./build_triton.sh

set -euo pipefail
export MAX_JOBS=64
export TRITON_BUILD_WITH_CCACHE=true


ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
[[ -f "$ROOT/set_env.sh" ]] && source "$ROOT/set_env.sh"

LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$ROOT/llvm-project/build}"
# export LLVM_BUILD_DIR=/root/triton_workspace/llvm-project/build
TRITON_SRC="${TRITON_SRC:-$ROOT/triton}"

if [[ ! -d "$TRITON_SRC" ]]; then
  echo "error: Triton 源码目录不存在: $TRITON_SRC" >&2
  exit 1
fi

if [[ ! -d "$LLVM_BUILD_DIR/include" ]] || [[ ! -d "$LLVM_BUILD_DIR/lib" ]]; then
  echo "error: LLVM 构建目录不完整（需要 include/ 与 lib/）: $LLVM_BUILD_DIR" >&2
  echo "请先运行 build_llvm.sh 或设置正确的 LLVM_BUILD_DIR。" >&2
  exit 1
fi

export LLVM_INCLUDE_DIRS="$LLVM_BUILD_DIR/include"
export LLVM_LIBRARY_DIR="$LLVM_BUILD_DIR/lib"
export LLVM_SYSPATH="$LLVM_BUILD_DIR"
export TRITON_BUILD_WITH_CLANG_LLD=1

export http_proxy=http://squid.iluvatar.ai:3128
export https_proxy=http://squid.iluvatar.ai:3128

cd "$TRITON_SRC"
pip install -r python/requirements.txt
pip install -e . --no-build-isolation
