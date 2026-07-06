"""
211_allocate_shared_memory.py
=============================

演示 `allocate-shared-memory` pass 的作用，以及 NVIDIA 版本与 Generic 版本的差异。

核心概念：
  GPU kernel 中有多个操作需要 shared memory（如 convert_layout、reduce、scan 等）。
  本 pass 使用图着色（graph coloring）算法，在满足所有生命周期约束的前提下，
  最小化 shared memory 的总占用。

  两种实现：
    1. Generic: `allocate-shared-memory`
    2. NVIDIA:  `allocate-shared-memory-nv`（带 compute-capability 参数）

  两种分配方式：
    a. 隐式 scratch buffer（convert_layout 等）— 编译器自动分配
    b. 显式 local_alloc（Gluon 风格）— 用户通过 ttg.local_alloc 手动分配

本脚本通过以下方式展示：
  1. 【文本对比】直接展示 pass 变换前后的 IR
  2. 【triton-opt】如果 triton-opt 可用，用 .mlir 测试文件运行 pass，
     同时测试隐式（convert_layout）和显式（local_alloc）两种分配方式
  3. 【核心函数分析】对比 defaultAllocationAnalysisScratchSizeFn 与
     getNvidiaAllocationAnalysisScratchSizeFn 的实现差异
"""

import subprocess
import sys
import os


TRITON_OPT = None
for p in [
    "/root/triton_workspace/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt",
    "/root/triton_workspace/triton/build/cmake.linux-x86_64-cpython-3.13/bin/triton-opt",
]:
    if os.path.isfile(p):
        TRITON_OPT = p
        break


# ===========================================================================
# 示例 1: 文本形式的 IR 变换对比
# ===========================================================================

def demo_text():
    """用文本直接展示 pass 变换前后的 IR 对比。"""
    print("=" * 72)
    print("示例 1: pass 变换前后 IR 对比（文本展示）")
    print("=" * 72)

    print("""
  场景：一个 convert_layout 操作需要在 blocked layout 和 dot_op layout 之间转换，
        shared memory 作为临时 buffer。

  ┌─ 变换前 ─────────────────────────────────────────────────────────┐
  │                                                                    │
  │  module attributes {"ttg.num-warps" = 4 : i32} {                   │
  │    tt.func @kernel(%arg0: tensor<128x32xf16, #blocked>) {           │
  │      %0 = ttg.convert_layout %arg0                                 │
  │           : tensor<128x32xf16, #blocked>                            │
  │           -> tensor<128x32xf16, #dot_op>                           │
  │      tt.return                                                      │
  │    }                                                                │
  │  }                                                                  │
  │                                                                    │
  │  问题: 哪些操作需要 shared memory scratch buffer？                   │
  │        需要的总大小是多少？                                          │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─ 变换后 ─────────────────────────────────────────────────────────┐
  │                                                                    │
  │  module attributes {                                               │
  │    "ttg.num-warps" = 4 : i32,                                      │
  │    "ttg.shared" = 8192 : i32       ← 模块总 shared memory        │
  │  } {                                                               │
  │    tt.func @kernel(%arg0: tensor<128x32xf16, #blocked>) {           │
  │      %0 = ttg.convert_layout %arg0 {                                │
  │        allocation.offset = 0 : i32  ← 此操作的偏移量              │
  │      } : tensor<128x32xf16, #blocked>                               │
  │        -> tensor<128x32xf16, #dot_op>                              │
  │      tt.return                                                      │
  │    }                                                                │
  │  }                                                                  │
  │                                                                    │
  │  pass 的输出：                                                     │
  │    - allocation.offset = 0 : i32  每个操作的偏移量                │
  │    - ttg.shared = 8192 : i32      shared memory 总大小            │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘

  多操作的 shared memory 复用示例（图着色）：
  ┌─ 变换前 ─────────────────────────────────────────────────────────┐
  │                                                                    │
  │  %a_cvt = ttg.convert_layout %a    ← 需要 8192 字节 scratch       │
  │  %c = tt.dot %a_cvt, ...           ← a_cvt 被消费，不再使用        │
  │  %b_cvt = ttg.convert_layout %b    ← 需要 8192 字节 scratch       │
  │                                    与 a_cvt 生命周期不重叠         │
  │                                     → 可复用同一块内存！            │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─ 变换后 ─────────────────────────────────────────────────────────┐
  │                                                                    │
  │  %a_cvt = ttg.convert_layout %a {allocation.offset = 0}           │
  │  %c = tt.dot %a_cvt, ...                                          │
  │  %b_cvt = ttg.convert_layout %b {allocation.offset = 0}  ← 复用! │
  │  ttg.shared = 8192 : i32    ← 而非 16384                          │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
  """)


# ===========================================================================
# 示例 2: 用 triton-opt 运行 .mlir 测试
# ===========================================================================

def demo_tritonopt():
    """如果 triton-opt 可用，用 .mlir 测试文件运行 pass。"""
    print("=" * 72)
    print("示例 2: 用 triton-opt 运行 allocate-shared-memory pass")
    print("=" * 72)

    if TRITON_OPT is None:
        print("  [SKIP] triton-opt not found, skip this demo.")
        return

    # 构造 MLIR 测试模块（用 split-input-file 分隔多个测试场景）
    mlir_input = """\
// 场景 1: convert_layout 的隐式 scratch buffer（默认 Triton 风格）
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [1, 0]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @kernel(%arg0: tensor<128x32xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #dot>
    tt.return
  }
}

// -----

// 场景 2: local_alloc 的显式 shared memory 分配（Gluon 风格）
// 使用 ttg.local_alloc 显式申请 shared memory buffer，
// 然后通过 ttg.local_load / ttg.local_store 读写。
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @gluon_shared(%arg0: tensor<32x16xf16, #blocked2>) {
    // 显式分配 shared memory buffer
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #shared, #smem, mutable>
    // 写入 shared memory
    ttg.local_store %arg0, %buf : tensor<32x16xf16, #blocked2> -> !ttg.memdesc<32x16xf16, #shared, #smem, mutable>
    // 从 shared memory 读取
    %loaded = ttg.local_load %buf : !ttg.memdesc<32x16xf16, #shared, #smem, mutable> -> tensor<32x16xf16, #blocked2>
    tt.return
  }
}
"""

    for pass_name, pass_args in [
        ("allocate-shared-memory", ""),
        ("allocate-shared-memory-nv", "compute-capability=90 ptx-version=81"),
    ]:
        cmd = [TRITON_OPT, "--split-input-file"]
        if pass_args:
            cmd += [f"--{pass_name}={pass_args}"]
        else:
            cmd += [f"--{pass_name}"]
        print(f"\n  triton-opt --{pass_name} ...")
        try:
            proc = subprocess.run(
                cmd,
                input=mlir_input,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                module_idx = 0
                for line in proc.stdout.strip().split("\\n"):
                    if "module " in line and "attributes" in line:
                        print(f"\n  ── 场景 {module_idx + 1} ──")
                        module_idx += 1
                    if "ttg.shared" in line or "allocation.offset" in line or "tt.func" in line or "module " in line:
                        print(f"    {line}")
            else:
                print(f"    [ERROR] {proc.stderr.strip()}")
        except FileNotFoundError:
            print(f"    [SKIP] triton-opt not found")


# ===========================================================================
# 示例 3: 核心函数分析
# ===========================================================================

def demo_function_analysis():
    """对比 defaultAllocationAnalysisScratchSizeFn 与 getNvidiaAllocationAnalysisScratchSizeFn。"""
    print("=" * 72)
    print("示例 3: 两个 scratch size 函数的深入对比")
    print("=" * 72)

    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │ 一、defaultAllocationAnalysisScratchSizeFn (在 Allocation.cpp)   │
  │        triton/lib/Analysis/Allocation.cpp                         │
  └────────────────────────────────────────────────────────────────────┘

  定义：
    unsigned defaultAllocationAnalysisScratchSizeFn(Operation *op)

  处理的 Operation 类型：
  ┌─────────────────┬────────────────────────────────────────────┐
  │ Operation       │ 计算方式                                    │
  ├─────────────────┼────────────────────────────────────────────┤
  │ ReduceOp        │ ReduceOpHelper::getScratchSizeInBytes()    │
  │ ScanOp          │ ScanLoweringHelper::getScratchSizeInBytes()│
  │ GatherOp        │ GatherLoweringHelper::getScratchSizeInBytes│
  │ HistogramOp     │ max(numElements, threadsPerWarp) * bw / 8 │
  │ ConvertLayoutOp │ getNumScratchElemsSwizzledCvt() * bw / 8  │
  │ AtomicRMW/CAS   │ getRepShapeForAtomic() * elemBw / 8       │
  │ TensormapCreate  │ 128 (TMA descriptor size)                │
  │ Other           │ 0                                          │
  └─────────────────┴────────────────────────────────────────────┘

  对于 ConvertLayoutOp，它调用的是：
    unsigned getNumScratchElemsSwizzledCvt(RankedTensorType srcTy,
                                           RankedTensorType dstTy)

  其中使用 optimalSwizzlingLdSt() 来计算最优的 swizzling 参数。
  这个版本没有 TargetInfo 感知，由所有 backend 共享。

  关键流程：
    src layout ──→ removeBroadcastedRegs ──┐
                                           ├──→ optimalSwizzlingLdSt ──→ scratch size
    dst layout ──→ removeBroadcastedRegs ──┘


  ┌────────────────────────────────────────────────────────────────────┐
  │ 二、getNvidiaAllocationAnalysisScratchSizeFn (在 Allocation.cpp)  │
  │    triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/          │
  └────────────────────────────────────────────────────────────────────┘

  这是一个工厂函数，接受 targetInfo 参数，返回一个 lambda：
    std::function<unsigned(Operation *)>
    getNvidiaAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo)

  对于 ConvertLayoutOp：
    调用 NVIDIA 自己的 getNumScratchElemsSwizzledCvt()：
      ┌─ src layout ──→ removeBroadcastedRegs ──┐
      │                                          ├──→ optimalSwizzling
      │                                          │      (general version)
      │ dst layout ──→ removeBroadcastedRegs ────┘          ↑
      │                                              targetInfo
      │                                         getSrcDstTiles()
      │                                         ┌─ ld/st.shared
      │                                         ├─ ldmatrix (cc≥75)
      │                                         ├─ stmatrix (cc≥90)
      │                                         └─ ldmatrix.trans (bw=16)
      │
      └──→ 返回 elems * getBitwidth(srcTy) / 8

  对于非 ConvertLayoutOp，回退到 defaultAllocationAnalysisScratchSizeFn。

  关键区别 - getSrcDstTiles 如何利用 TargetInfo：
    if (targetInfo.supportLdMatrix() || targetInfo.supportStMatrix()) {
      if (bitwidth <= 32)
        add ldmatrix/stmatrix tile
      if (bitwidth == 16)
        add ldmatrix.trans/stmatrix.trans tile
    }

    ┌─────────────┬──────────────┬──────────────┬──────────────────────┐
    │ Architecture │ ComputeCap  │ supportLdMat │ supportStMat         │
    ├─────────────┼──────────────┼──────────────┼──────────────────────┤
    │ Volta/Turing│ sm_70/sm_75  │  cc ≥ 75 ✅  │  ❌ (cc < 90)        │
    │ Ampere      │ sm_80        │  ✅          │  ❌                  │
    │ Hopper      │ sm_90        │  ✅          │  ✅ (cc ≥ 90)       │
    │ Blackwell   │ sm_100       │  ✅          │  ✅                  │
    └─────────────┴──────────────┴──────────────┴──────────────────────┘


  ┌────────────────────────────────────────────────────────────────────┐
  │ 三、两者的核心差异                                               │
  └────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┬────────────────────────┬──────────────────────────┐
  │ 维度             │ Generic                │ NVIDIA                   │
  ├──────────────────┼────────────────────────┼──────────────────────────┤
  │ Swizzling 函数   │ optimalSwizzlingLdSt   │ optimalSwizzling         │
  │                  │ (load/store 专用)       │ (通用，支持更多 tile)    │
  ├──────────────────┼────────────────────────┼──────────────────────────┤
  │ TargetInfo 感知  │ ❌ 无                   │ ✅ 通过 getSrcDstTiles    │
  │                  │                         │   传入 targetInfo        │
  ├──────────────────┼────────────────────────┼──────────────────────────┤
  │ ldmatrix 利用    │ ❌ 无                   │ ✅ cc≥75 时包含          │
  ├──────────────────┼────────────────────────┼──────────────────────────┤
  │ stmatrix 利用    │ ❌ 无                   │ ✅ cc≥90 时包含          │
  ├──────────────────┼────────────────────────┼──────────────────────────┤
  │ 架构特定 tile    │ ❌ 固定                 │ ✅ 按 compute capability │
  │                  │                         │    动态调整 tile         │
  ├──────────────────┼────────────────────────┼──────────────────────────┤
  │ 回退行为         │ 直接返回结果            │ 非 ConvertLayout 回退   │
  │                  │                         │ 到 default 函数          │
  └──────────────────┴────────────────────────┴──────────────────────────┘


  ┌────────────────────────────────────────────────────────────────────┐
  │ 四、为什么需要 NVIDIA 独立实现                                    │
  └────────────────────────────────────────────────────────────────────┘

  1. ldmatrix/stmatrix 指令利用
     ldmatrix 是 NVIDIA 从 sm_75（Turing）开始引入的 shared memory -> 寄存器
     加载指令。stmatrix 是 sm_90（Hopper）引入的寄存器 -> shared memory 存储
     指令。这些指令可以显著提高矩阵转置和特定布局转换的效率。

     如果没有 TargetInfo 感知，optimalSwizzling 就不知道应该预留多大空间
     给这些矩阵操作指令进行 tile 对齐。

  2. optimalSwizzling  vs  optimalSwizzlingLdSt
     Generic 版本使用的 optimalSwizzlingLdSt 是为 load/store 指令优化的
     简化版本。NVIDIA 使用更通用的 optimalSwizzling，可以容纳
     ldmatrix/stmatrix 等多种指令的不同 tile 约束。

  3. 不同架构的 Tile 需求不同
     Volta (sm_70)：只有 ld/st.shared，不需要额外的 tile
     Turing (sm_75)：支持 ldmatrix，需要 32-bit 的 tile
     Ampere (sm_80)：支持 ldmatrix，同上
     Hopper (sm_90)：支持 ldmatrix + stmatrix，需要 32-bit 和 16-bit 的 tile
     Blackwell (sm_100)：同上，额外支持 B8 格式

  ┌────────────────────────────────────────────────────────────────────┐
  │ 五、代码对应                                                     │
  └────────────────────────────────────────────────────────────────────┘

  Generic 版本（Allocation.h 默认参数）：
    ModuleAllocation(ModuleOp moduleOp,
                     AllocationAnalysisScratchSizeFn scratchSizeGetter = 
                         defaultAllocationAnalysisScratchSizeFn)
    // 创建时使用默认的 scratchSizeGetter

  NVIDIA 版本（传入自定义的 scratchSizeGetter）：
    ModuleAllocation allocation(
        mod, getNvidiaAllocationAnalysisScratchSizeFn(targetInfo));
    // 创建时使用 NV 特定的 scratchSizeGetter
    """)

    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │ 附录：getSrcDstTiles 完整实现                                    │
  └────────────────────────────────────────────────────────────────────┘

  SmallVector<LocalMemOpTile> getSrcDstTiles(TargetInfoBase &targetInfo,
                                             int bitwidth) {
    // 1) 始终添加基础的 ld/st.shared tile (所有架构都支持)
    src.push_back({{}, {0, 1, 2}});
    dst.push_back({{}, {0, 1, 2}});

    // 2) 如果架构支持 ldmatrix/stmatrix (cc ≥ 75 / cc ≥ 90)
    if (targetInfo.supportLdMatrix() || targetInfo.supportStMatrix()) {
      if (bitwidth <= 32) {
        // 添加 ldmatrix/stmatrix tile: {{0, 1}, {2, 3, 4}}
        // 这表示一个 2D tile，在 dim 0/1 上加载，在 dim 2/3/4 上存储
        if (targetInfo.supportStMatrix()) src.push_back(ldmatrix);
        if (targetInfo.supportLdMatrix()) dst.push_back(ldmatrix);
      }
      if (bitwidth == 16) {
        // 添加 ldmatrix.trans/stmatrix.trans tile: {{2, 3, 4}, {0, 1}}
        // 这表示转置加载
        if (targetInfo.supportStMatrix()) src.push_back(trans);
        if (targetInfo.supportLdMatrix()) dst.push_back(trans);
      }
    }
    return {src, dst};
  }
  """)


# ===========================================================================
# 入口
# ===========================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--text-only":
        demo_text()
    elif len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        demo_function_analysis()
    else:
        demo_text()
        print()
        demo_tritonopt()
        print()
        demo_function_analysis()
