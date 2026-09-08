#! /bin/bash
# 构建并安装 TileLang，支持 debug 模式 + 可编辑安装。
# 当前目录下需要存在 tilelang/ 源码目录。
#
# 用法:
#   ./build_tilelang.sh                  # release 模式 + 可编辑安装
#   ./build_tilelang.sh -d               # debug 模式 + 可编辑安装
#   ./build_tilelang.sh --debug          # debug 模式
#   ./build_tilelang.sh --release        # release 模式（默认）
#   ./build_tilelang.sh --no-edit        # 非可编辑模式（完整安装）
#   USE_CUDA=OFF ./build_tilelang.sh     # 禁用 CUDA 后端
#   USE_ROCM=ON  ./build_tilelang.sh     # 启用 ROCm 后端
#
# 环境变量:
#   MAX_JOBS          — 并行编译线程数（默认 64）
#   CMAKE_BUILD_TYPE  — Debug / Release（默认由 --debug/--release 控制）
#   USE_CUDA          — ON / OFF（默认自动检测）
#   USE_ROCM          — ON / OFF（默认 OFF）

set -euo pipefail

# ─── 默认值 ──────────────────────────────────────────────
MAX_JOBS="${MAX_JOBS:-64}"
BUILD_TYPE="Release"      # 默认 release
EDITABLE=true             # 默认可编辑安装

# ─── 解析参数 ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    --release)
      BUILD_TYPE="Release"
      shift
      ;;
    --no-edit)
      EDITABLE=false
      shift
      ;;
    *)
      echo "未知参数: $1" >&2
      echo "用法: $0 [--debug|--release] [--no-edit]" >&2
      exit 1
      ;;
  esac
done

# ─── 路径 ────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
TILELANG_SRC="${TILELANG_SRC:-$ROOT/tilelang}"

if [[ ! -d "$TILELANG_SRC" ]]; then
  echo "错误: TileLang 源码目录不存在: $TILELANG_SRC" >&2
  echo "请将本脚本放在包含 tilelang/ 子目录的根目录，或设置 TILELANG_SRC 环境变量。" >&2
  exit 1
fi

# ─── set_env.sh（可选） ─────────────────────────────────
[[ -f "$ROOT/set_env.sh" ]] && source "$ROOT/set_env.sh"

# ─── 导出 CMake 选项 ────────────────────────────────────
export CMAKE_BUILD_TYPE="$BUILD_TYPE"
export MAX_JOBS

# 透传 USE_CUDA / USE_ROCM（如果没设就让 CMake 自动检测）
[[ -n "${USE_CUDA:-}" ]] && export USE_CUDA
[[ -n "${USE_ROCM:-}" ]]  && export USE_ROCM

# ─── 打印配置 ───────────────────────────────────────────
echo "========================================"
echo " TileLang Build Script"
echo "========================================"
echo "  源码目录:      $TILELANG_SRC"
echo "  构建类型:      $CMAKE_BUILD_TYPE"
echo "  可编辑安装:    $EDITABLE"
echo "  MAX_JOBS:       $MAX_JOBS"
echo "  USE_CUDA:       ${USE_CUDA:-<auto>}"
echo "  USE_ROCM:       ${USE_ROCM:-<auto>}"
echo "========================================"

# ─── 安装 Python 依赖 ──────────────────────────────────
# pyproject.toml 中声明了 build-system 依赖（cython, scikit-build-core 等）
# 但在 --no-build-isolation 模式下需要手动装好
cd "$TILELANG_SRC"

# pip install --quiet \
#   cython>=3.1.0 \
#   scikit-build-core \
#   z3-solver>=4.13.0 \
#   2>/dev/null || true

# 运行时依赖（可选 — pip install -e 时会自动处理，但提前装好避免提示）
# pip install --quiet \
#   "apache-tvm-ffi>=0.1.10,<=0.1.11" \
#   cloudpickle ml-dtypes numpy psutil torch tqdm \
#   "typing-extensions>=4.10.0" \
#   "z3-solver>=4.13.0,<4.15.5" \
#   2>/dev/null || true

# ─── 构建并安装 ─────────────────────────────────────────
if [[ "$EDITABLE" == true ]]; then
  echo "→ 正在以可编辑模式安装（editable）..."
  pip install -e . --no-build-isolation -v
else
  echo "→ 正在以完整模式安装..."
  pip install . --no-build-isolation -v
fi

echo "✓ 安装完成！"
echo "  Python 包: tilelang"
echo "  构建类型:  $CMAKE_BUILD_TYPE"
echo ""
echo "  验证:  python3 -c \"import tilelang; print(tilelang.__version__)\""
