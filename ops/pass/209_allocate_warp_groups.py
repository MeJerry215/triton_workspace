"""
209_allocate_warp_groups.py
============================

演示 `tritongpu-allocate-warp-groups` pass 的作用。

核心概念：
  当一个 GPU 程序包含 warp specialization 时（由 `ttg.warp_specialize` 表达），
  额外的 warp 被启动以执行不同的分区代码。本 pass 负责：
    1. 填充每个 warp_specialize 使其具有相同数量的 warp group
    2. 为每个分区分配 warp 起始 ID
    3. 计算并设置每个 warp group 的寄存器上限

本脚本通过以下方式展示：
  1. 【文本对比】直接展示 pass 变换前后的 IR 结构
  2. 【triton-opt】如果 triton-opt 可用，用现有 .mlir 测试文件运行 pass
  3. 【概念说明】解释 warp group、warp specialization 和寄存器分配的关系
"""

import subprocess
import sys
import os


# ===========================================================================
# 示例 1: 文本形式的 IR 变换对比
# ===========================================================================

def demo_text():
    """用文本直接展示 pass 变换前后的 IR 对比。"""
    print("=" * 72)
    print("示例 1: pass 变换前后 IR 对比（文本展示）")
    print("=" * 72)

    print("""
  场景：一个包含 3 个 partition 的 warp_specialize，
        partition warp 数分别为 1 + 8 + 4 = 13 个 warp。

  ┌─ 变换前 ─────────────────────────────────────────────────────────┐
  │                                                                  │
  │  module attributes {"ttg.num-warps" = 4 : i32} {                 │
  │                                                                  │
  │  tt.func @kernel() {                                             │
  │    ttg.warp_specialize()                                         │
  │    default { ttg.warp_yield }        ← 默认 4 warp               │
  │    partition0() num_warps(1) { ... }                              │
  │    partition1() num_warps(8) { ... }                              │
  │    partition2() num_warps(4) { ... }                              │
  │    : () -> ()                                                     │
  │    tt.return                                                      │
  │  }                                                                │
  │  }                                                                │
  │                                                                  │
  │  问题: partition 总 warp 数（13）不是 4 的倍数，不能对齐到       │
  │        完整的 warp group。各 partition 没有分配 start ID，       │
  │        也没有寄存器预算。                                        │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─ 变换后 ─────────────────────────────────────────────────────────┐
  │                                                                  │
  │  module attributes {"ttg.num-warps" = 4 : i32,                   │
  │                     "ttg.total-num-warps" = 20 : i32} {          │
  │                                                                  │
  │  tt.func @kernel() {                                             │
  │    ttg.warp_specialize() attributes {warpGroupStartIds =         │
  │        array<i32: 18, 4, 12, 16, 19>}                            │
  │    default { ttg.warp_yield }                                    │
  │    partition0() num_warps(1) { ... }   ← startId=18              │
  │    partition1() num_warps(8) { ... }   ← startId=4（最大→最低） │
  │    partition2() num_warps(4) { ... }   ← startId=12              │
  │    partition3() num_warps(2) { ... }   ← padding（新填充）      │
  │    partition4() num_warps(1) { ... }   ← padding（新填充）      │
  │    : () -> ()                                                    │
  │    tt.return                                                      │
  │  }                                                                │
  │  }                                                                │
  │                                                                  │
  │  效果: ✅ 填充到 16 个 extra warp = 4 个 warp group              │
  │        ✅ start ID 已分配（大分区优先）                           │
  │        ✅ total-num-warps = 4(默认) + 16(extra) = 20             │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ 寄存器分配场景 ──────────────────────────────────────────────────┐
  │  当 requestedRegisters 已设置时，还会计算寄存器 budget:           │
  │                                                                  │
  │  变换前:                                                         │
  │    ttg.warp_specialize() attributes {requestedRegisters =        │
  │        array<i32: 48, 80, 48>}                                   │
  │    default { ... }                                               │
  │    partition0() num_warps(1) { ... }   (预估 48 寄存器)          │
  │    partition1() num_warps(2) { ... }   (预估 80 寄存器)          │
  │    partition2() num_warps(1) { ... }   (预估 48 寄存器)          │
  │                                                                  │
  │  变换后:                                                         │
  │    module attributes {ttg.maxnreg = 168 : i32}                   │
  │    ttg.warp_specialize() attributes {actualRegisters =           │
  │        array<i32: 208, 80, 80, 80>}   ← 每个 warp group          │
  │                                         的寄存器上限             │
  └──────────────────────────────────────────────────────────────────┘
""")


# ===========================================================================
# 示例 2: 用 triton-opt 直接运行 pass
# ===========================================================================

def _find_triton_opt():
    """查找 triton-opt 可执行文件路径。"""
    candidates = [
        "triton-opt",
        os.path.expanduser("~/triton/build/bin/triton-opt"),
        "/root/triton_workspace/triton/build/bin/triton-opt",
    ]
    for cand in candidates:
        try:
            subprocess.run([cand, "--version"], capture_output=True)
            return cand
        except FileNotFoundError:
            continue
    return None


# 从 triton/test/Conversion/allocate_warp_groups.mlir 中提取的测试用例

TEST1_MLIR = """\
// Test: 空模块
module attributes {"ttg.num-warps" = 4 : i32} {
}
"""

TEST2_MLIR = """\
// Test: 基础 warp specialization 填充 + ID 分配
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @kernel() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(8) {
    ttg.warp_return
  }
  partition2() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}
"""

TEST3_MLIR = """\
// Test: 两个 warp_specialize
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @two_warp_specialize() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    ttg.warp_return
  }
  partition1() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

tt.func @another() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(8) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}
"""

TEST4_MLIR = """\
// Test: 寄存器分配
module attributes {"ttg.num-warps" = 8 : i32} {

tt.func @setmaxnreg() {
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 48, 80, 48>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(2) {
    ttg.warp_return
  }
  partition2() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}
"""

TEST5_MLIR = """\
// Test: 从默认 warp group 借用寄存器
module attributes {"ttg.num-warps" = 8 : i32} {

tt.func @steal_from_default() {
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 192>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(8) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}
"""


def _analyze_output(output, label):
    """分析 triton-opt 输出，提取关键属性。"""
    lines = output.strip().split("\n")
    print(f"  [{label}] IR 输出：")
    for line in lines:
        stripped = line.strip()
        if stripped:
            print(f"    {stripped}")

    # 提取关键属性
    attrs_to_check = [
        "ttg.total-num-warps",
        "ttg.maxnreg",
        "warpGroupStartIds",
        "actualRegisters",
        "requestedRegisters",
    ]
    for attr in attrs_to_check:
        for line in lines:
            if attr in line:
                print(f"    → 发现属性: {attr}")
                break


def demo_triton_opt():
    """尝试用 triton-opt 运行 pass，展示实际 IR 变换。"""
    print("=" * 72)
    print("示例 2: 用 triton-opt 运行 pass 验证")
    print("=" * 72)

    triton_opt = _find_triton_opt()
    if triton_opt is None:
        print("\n⚠️  未找到 triton-opt，跳过此示例。")
        print("   若要运行，请确保 Triton 已完整编译（build 目录中有 triton-opt）。")
        return

    import tempfile

    test_cases = [
        ("空模块", TEST1_MLIR, "✅ 无 warp_specialize → total-num-warps = num-warps"),
        ("基础填充+ID分配", TEST2_MLIR, "✅ 填充到 16 warp + 分配 start ID"),
        ("两个 warp_specialize", TEST3_MLIR, "✅ 多个 ws 各自填充 + 分配"),
        ("寄存器分配", TEST4_MLIR, "✅ actualRegisters + maxnreg 已设置"),
        ("从默认组借用寄存器", TEST5_MLIR, "✅ 回收默认组寄存器给 warp group"),
    ]

    for name, mlir_src, description in test_cases:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as f:
            f.write(mlir_src)
            f.flush()
            mlir_path = f.name

        try:
            print(f"\n--- 子测试: {name} ---")
            print(f"  预期: {description}")

            result = subprocess.run(
                [triton_opt, mlir_path, "-tritongpu-allocate-warp-groups"],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                _analyze_output(result.stdout, name)
            else:
                print(f"    ❌ triton-opt 运行失败: {result.stderr}")

        except Exception as e:
            print(f"    ❌ 错误: {e}")
        finally:
            os.unlink(mlir_path)

    # 额外：运行正式的 lit 测试文件（如果存在）
    lit_test_path = "/root/triton_workspace/triton/test/Conversion/allocate_warp_groups.mlir"
    if os.path.exists(lit_test_path):
        print(f"\n--- 额外: 运行官方 lit 测试文件 ---")
        result = subprocess.run(
            [triton_opt, lit_test_path, "-split-input-file",
             "-tritongpu-allocate-warp-groups"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            num_checks = result.stdout.count("warpGroupStartIds")
            num_modules = result.stdout.count("module attributes")
            print(f"  ✅ 官方测试文件运行成功")
            print(f"  → warpGroupStartIds 出现次数: {num_checks}")
            print(f"  → module attributes 数量: {num_modules}")
        else:
            print(f"  ❌ 官方测试文件运行失败: {result.stderr}")


# ===========================================================================
# 示例 3: 概念说明
# ===========================================================================

def demo_concept():
    """解释 warp group 是什么、warp specialization 的硬件背景。"""
    print("=" * 72)
    print("示例 3: Warp Group 与 Warp Specialization 概念解析")
    print("=" * 72)

    print("""
  ┌─ 硬件背景：从 SM90 (Hopper) 到 SM100 (Blackwell) ──────────┐
  │                                                              │
  │  NVIDIA GPU 架构演进中的 warp 相关概念：                      │
  │                                                              │
  │  SM70 (Volta)    ─── Warp (32 threads)                      │
  │                      引入独立线程调度                         │
  │                                                              │
  │  SM80 (Ampere)   ─── Warp (32 threads)                      │
  │                      引入异步拷贝、TF32                       │
  │                                                              │
  │  SM90 (Hopper)   ─── Warp Group (4 warps = 128 threads)      │
  │                      引入 Warp Specialization、TMEM           │
  │                                                              │
  │  SM100 (Blackwell) ─ Warp Group (4 warps)                    │
  │                      增强并发执行、更大的寄存器文件           │
  └──────────────────────────────────────────────────────────────┘

  ┌─ 为什么要用 Warp Specialization？ ─────────────────────────┐
  │                                                             │
  │  传统 GPU kernel 中，所有 warp 执行同一份代码。但对于       │
  │  HPC/AI 工作负载，不同的 warp 可以做不同的事：              │
  │                                                             │
  │  ┌─────────────────────────────────────────────────────────┐│
  │  │  Default WG:  主控制流、地址计算、标量操作              ││
  │  │  Partition 0: 数据加载（TMA load）                      ││
  │  │  Partition 1: 矩阵计算（MMA / Tensor Core）             ││
  │  │  Partition 2: 结果存储（TMA store）                     ││
  │  └─────────────────────────────────────────────────────────┘│
  │                                                             │
  │  优势：                                                     │
  │  • 隐藏延迟：加载单元在等待数据时，计算单元继续工作         │
  │  • 提高吞吐：流水线各阶段由不同 warp 组承载                │
  │  • 减少同步：warp group 之间通过 shared memory 通信         │
  └─────────────────────────────────────────────────────────────┘

  ┌─ 寄存器文件的挑战 ──────────────────────────────────────────┐
  │                                                              │
  │  GPU 的寄存器文件是所有 warp 共享的物理资源。当使用           │
  │  warp specialization 时，不同 warp group 对寄存器的需求      │
  │  不同：                                                      │
  │                                                              │
  │   ┌──────┬──────────┬──────────────┬──────────────┐          │
  │   │ WG   │ Warp 数  │ 请求寄存器   │ 实际分配     │          │
  │   ├──────┼──────────┼──────────────┼──────────────┤          │
  │   │ 默认 │ 8        │ -            │ 208 (回收后) │          │
  │   │ WG 0 │ 1+2=3→4  │ max(48,80)=80│ 80           │          │
  │   │ WG 1 │ 1        │ 48→48        │ 80           │          │
  │   └──────┴──────────┴──────────────┴──────────────┘          │
  │                                                              │
  │  关键洞察：                                                  │
  │  如果一个 warp group 只需要 80 个寄存器，而每个 warp          │
  │  理论上可以分配 168 个，那么剩余的 88×4×32 个寄存器槽        │
  │  可以被默认 warp group 回收利用。                            │
  │                                                              │
  │  这就是 allocate-warp-groups pass 的寄存器分配核心逻辑：      │
  │    registerBudget = Σ(maxnreg × numWarps × 32)               │
  │    leftover = registerBudget / (baseNumWarps × 32)           │
  │    这个 leftover 就是默认 warp group 最终可以使用的寄存器数   │
  └──────────────────────────────────────────────────────────────┘

  ┌─ 本 pass 在编译流程中的定位 ──────────────────────────────┐
  │                                                            │
  │  在 NVIDIA backend 的 make_llir pipeline 中：               │
  │                                                            │
  │  输入: TTGPUIR (包含 ttg.warp_specialize)                  │
  │    │                                                        │
  │    ▼                                                        │
  │  ┌────────────────────────────────────────────────────┐     │
  │  │ combine-tensor-select-and-if  (减少 IR 宽度)       │     │
  │  ├────────────────────────────────────────────────────┤     │
  │  │ ▸ allocate-warp-groups (本 pass)                   │     │
  │  │   · 填充 warp group                                │     │
  │  │   · 分配 start ID                                  │     │
  │  │   · 计算寄存器 budget                              │     │
  │  ├────────────────────────────────────────────────────┤     │
  │  │ scf-to-cf (结构化控制流 → CFG)                     │     │
  │  ├────────────────────────────────────────────────────┤     │
  │  │ allocate-shared-memory / allocate-tensor-memory    │     │
  │  ├────────────────────────────────────────────────────┤     │
  │  │ to-llvmir (TTGPUIR → LLVM IR)                      │     │
  │  ├────────────────────────────────────────────────────┤     │
  │  │ warp-specialize-to-llvm  (使用标注信息)            │     │
  │  └────────────────────────────────────────────────────┘     │
  │                                                            │
  │  输出: LLVM IR (warp 分配信息已标注在 Attribute 中)        │
  └────────────────────────────────────────────────────────────┘

  ┌─ 关键术语表 ──────────────────────────────────────────────┐
  │                                                            │
  │  ┌────────────────────────┬──────────────────────────┐     │
  │  │ 术语                    │ 含义                     │     │
  │  ├────────────────────────┼──────────────────────────┤     │
  │  │ Warp                   │ 32 线程，基本调度单位    │     │
  │  │ Warp Group             │ 4 warp = 128 线程        │     │
  │  │ Default Warp Group     │ 执行主函数的 warp 组     │     │
  │  │ Partition              │ WarpSpecialize 的代码分区│     │
  │  │ Extra Warp             │ 主函数之外的额外 warp    │     │
  │  │ warpGroupStartIds      │ 各 partition 的起始 ID   │     │
  │  │ requestedRegisters     │ 用户/前一 pass 预估的    │     │
  │  │                        │ 寄存器需求量             │     │
  │  │ actualRegisters        │ 本 pass 计算的最终分配量 │     │
  │  │ ttg.maxnreg            │ Module 级别的寄存器上限  │     │
  │  │ ttg.total-num-warps    │ 程序所需的总 warp 数     │     │
  │  └────────────────────────┴──────────────────────────┘     │
  └────────────────────────────────────────────────────────────┘
""")


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  tritongpu-allocate-warp-groups Pass 演示                          ║
║                                                                    ║
║  作用：分析 ttg.warp_specialize 操作，确定所需 warp 总数，         ║
║        为每个 warp group 分配 warp ID 范围并计算寄存器预算。       ║
║                                                                    ║
║  三个阶段：                                                        ║
║    ① 填充到最大 warp group（对齐到 4 的倍数）                     ║
║    ② 分配 warp 起始 ID（大分区优先）                              ║
║    ③ 计算寄存器 budget（跨 warp group 回收）                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    demo_text()
    demo_triton_opt()
    demo_concept()

    print("=" * 72)
    print("总结")
    print("=" * 72)
    print("""
  allocate-warp-groups pass 的核心价值：

  ① 实现 warp group 对齐 → 所有 warp_specialize 的 partition 总数
     被填充到相同的 warp group 数量，确保硬件调度一致性。

  ② 分配 warp 起始 ID → 通过 "大分区优先" 策略减少 warp ID
     碎片化，使 warp group 的 ID 连续且高效。

  ③ 寄存器预算计算与回收 → 利用 warp 的时间复用特性，将
     warp group 未使用的寄存器回收给默认 warp group 使用，
     最大化寄存器利用率。

  ④ 提供下游 pass 所需标注 → warpGroupStartIds、actualRegisters、
     ttg.maxnreg、ttg.total-num-warps 等属性后续会被
     warp-specialize-to-llvm 等 pass 使用。

  实现要点：
    ▸ pass 在整个 ModuleOp 上全局分析
    ▸ padding 填充使用 2 的幂分区大小
    ▸ 寄存器取整到 8 的倍数（PTX setmaxnreg 的限制）
    ▸ 回收默认组寄存器时至少有 24 个寄存器的下限
""")
