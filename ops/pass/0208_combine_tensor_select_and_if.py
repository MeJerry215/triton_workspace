"""
208_combine_tensor_select_and_if.py
====================================

演示 `tritongpu-combine-tensor-select-and-if` pass 的作用。

核心概念：
  当 `arith.select`（张量类型）和 `scf.if` 使用相同的条件时，
  将 select 的 true/false 值合并到 if 的 yield 中，消除多余的 select 操作。

本脚本通过以下方式展示：
  1. 【文本对比】直接展示 pass 变换前后的 IR
  2. 【triton-opt】如果 triton-opt 可用，直接用现有 .mlir 测试文件运行 pass
  3. 【概念说明】解释该 pattern 在 Triton 编译流程中的出现场景
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
  场景：arith.select 和 scf.if 共享同一条件 %cnd，

  ┌─ 变换前 ─────────────────────────────────────────────────────────┐
  │                                                                  │
  │  %cst0 = arith.constant dense<0.000000e+00> : tensor<64xf32>    │
  │  %cst1 = arith.constant dense<1.000000e+00> : tensor<64xf32>    │
  │  %sel  = arith.select %cnd, %cst0, %cst1 : tensor<64xf32>      │
  │  scf.if %cnd {                                                  │
  │    tt.store %ptr, %arg0 : tensor<64x!tt.ptr<f32>>               │
  │  }                                                              │
  │  tt.store %ptr, %sel : tensor<64x!tt.ptr<f32>>                  │
  │                                                                  │
  │  问题: %sel 是一个独立的中间张量 select，浪费寄存器和带宽。     │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─ 变换后 ─────────────────────────────────────────────────────────┐
  │                                                                  │
  │  %cst0 = arith.constant dense<0.000000e+00> : tensor<64xf32>    │
  │  %cst1 = arith.constant dense<1.000000e+00> : tensor<64xf32>    │
  │  %r    = scf.if %cnd -> tensor<64xf32> {                        │
  │    tt.store %ptr, %arg0 : tensor<64x!tt.ptr<f32>>               │
  │    scf.yield %cst0 : tensor<64xf32>    ← true 值（来自 select） │
  │  } else {                                                        │
  │    scf.yield %cst1 : tensor<64xf32>    ← false 值（来自 select）│
  │  }                                                               │
  │  tt.store %ptr, %r : tensor<64x!tt.ptr<f32>>                    │
  │                                                                  │
  │  效果: ✅ arith.select 被消除，%cst0/%cst1 被直接 yield         │
  │        ✅ 张量无需物化，直接走控制流                             │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ 进阶场景：多个 select ──────────────────────────────────────────┐
  │  当同一个条件对应多个 select 时，它们全部合并到同一个 if 中。    │
  │                                                                  │
  │  变换前:                                                        │
  │    %s0 = arith.select %cnd, %a, %b : tensor<64xi32>             │
  │    %s1 = arith.select %cnd, %c, %d : tensor<64xf32>             │
  │    %r  = scf.if %cnd -> (tensor<64xi32>) { ... }               │
  │                                                                  │
  │  变换后:                                                        │
  │    %r:3 = scf.if %cnd -> (tensor<64xi32>, tensor<64xi32>,       │
  │                                          tensor<64xf32>) {      │
  │      scf.yield %x, %a, %c        ← 多个额外 yield               │
  │    } else {                                                      │
  │      scf.yield %y, %b, %d                                       │
  │    }                                                             │
  └──────────────────────────────────────────────────────────────────┘
""")


# ===========================================================================
# 示例 2: 用 triton-opt 直接运行 pass
# ===========================================================================

def _find_triton_opt():
    """查找 triton-opt 可执行文件路径。"""
    # 尝试常见的安装路径
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


TRANSFORM_TEST_MLIR = """\
// RUN: triton-opt %s -tritongpu-combine-tensor-select-and-if | FileCheck %s

// CHECK-LABEL: @select_if_combine
tt.func public @select_if_combine(%arg0: tensor<64xf32>, %dst_ptr: tensor<64x!tt.ptr<f32>>, %cnd: i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<64xf32>
  // CHECK-NOT: arith.select
  %sel = arith.select %cnd, %cst, %cst_1 : tensor<64xf32>
  // CHECK: %[[R:.*]] = scf.if %{{.*}} -> (tensor<64xf32>)
  scf.if %cnd {
    tt.store %dst_ptr, %arg0 : tensor<64x!tt.ptr<f32>>
  }
  // CHECK: tt.store %{{.*}}, %[[R]]
  tt.store %dst_ptr, %sel : tensor<64x!tt.ptr<f32>>
  tt.return
}
"""

TRANSFORM_TEST_MLIR2 = """\
// RUN: triton-opt %s -tritongpu-combine-tensor-select-and-if | FileCheck %s

// CHECK-LABEL: @multiple_selects
tt.func @multiple_selects(%cnd: i1, %a: tensor<64xi32>, %b: tensor<64xi32>,
                          %c: tensor<64xf32>, %d: tensor<64xf32>,
                          %e: tensor<64xi32>)
    -> (tensor<64xi32>, tensor<64xf32>, tensor<64xi32>) {
  // CHECK-NOT: arith.select
  %s0 = arith.select %cnd, %a, %b : tensor<64xi32>
  %s1 = arith.select %cnd, %c, %d : tensor<64xf32>
  // CHECK: %[[R:.*]]:3 = scf.if %{{.*}} -> (tensor<64xi32>, tensor<64xi32>, tensor<64xf32>)
  %r = scf.if %cnd -> (tensor<64xi32>) {
    %t = arith.subi %a, %b : tensor<64xi32>
    scf.yield %t : tensor<64xi32>
  } else {
    scf.yield %e : tensor<64xi32>
  }
  // CHECK: tt.return %[[R]]#1, %[[R]]#2, %[[R]]#0
  tt.return %s0, %s1, %r : tensor<64xi32>, tensor<64xf32>, tensor<64xi32>
}
"""


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

    for i, mlir_src in enumerate([TRANSFORM_TEST_MLIR, TRANSFORM_TEST_MLIR2], 1):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as f:
            f.write(mlir_src)
            f.flush()
            mlir_path = f.name

        try:
            print(f"\n子测试 {i}:")

            # 先打印输入
            result_before = subprocess.run(
                [triton_opt, mlir_path, "--mlir-print-ir-before=tritongpu-combine-tensor-select-and-if"],
                capture_output=True, text=True, timeout=30
            )
            # 运行 pass
            result_after = subprocess.run(
                [triton_opt, mlir_path, "-tritongpu-combine-tensor-select-and-if"],
                capture_output=True, text=True, timeout=30
            )

            if result_after.returncode == 0:
                output = result_after.stdout
                # 统计 select 数量
                select_count = output.count("arith.select")
                if_output = output.count("scf.if")
                print(f"    输出中的 arith.select: {select_count}")
                print(f"    输出中的 scf.if:      {if_output}")

                # 打印关键部分
                lines = output.strip().split("\n")
                print("    IR 输出（核心部分）：")
                for line in lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("//"):
                        print(f"      {stripped}")
            else:
                print(f"    ❌ triton-opt 运行失败: {result_after.stderr}")

        except Exception as e:
            print(f"    ❌ 错误: {e}")
        finally:
            os.unlink(mlir_path)


# ===========================================================================
# 示例 3: 概念说明 — pattern 在 Triton 流程中的出现场景
# ===========================================================================

def demo_concept():
    """解释 arith.select 是什么、它和 scf.if 的关系、以及 pattern 的来源。"""
    print("=" * 72)
    print('示例 3: arith.select 的本质 \u2014 scf.if 的\u201c短路\u201d形式')
    print("=" * 72)

    print("""
  arith.select 本质上就是 scf.if 的"线性短路形式"：

    两个版本表达同样的语义："如果 %cond 为真取 %a，否则取 %b"

    ┌─ scf.if（控制流）──────┐    ┌─ arith.select（数据流）─────┐
    │  %r = scf.if %cond {    │    │  %r = arith.select          │
    │    scf.yield %a         │    │    %cond, %a, %b : i64     │
    │  } else {               │    │                             │
    │    scf.yield %b         │    │  一条指令，无基本块，无分支 │
    │  }                      │    │  两个操作数都先求值再选择   │
    └─────────────────────────┘    └─────────────────────────────┘

    关键区别——标量 vs 张量：

    标量 arith.select %cond, %a, %b : i32
      → 一条 PTX 指令 selp.s32，延迟 ~1 cycle
      → 两边都算好但代价极低，不值得消除 ✓

    张量 arith.select %cond, %a, %b : tensor<128xf32>
      → 128 个元素逐元素条件选择
      → %a 和 %b 两个 tensor 都必须先在寄存器中就位
      → 意味着：即使某个分支不会被使用，它的值也已经算好了！
      → 如果 %a 或 %b 来自昂贵的计算（如 matmul），浪费巨大 ✗

  ┌─ arith.select 的三大来源 ──────────────────────────────────┐
  │                                                             │
  │  ① tl.where(cond, x, y) — 最直接的来源                     │
  │     Triton 源码中的 tl.where 直接生成 arith.select          │
  │                                                             │
  │  ② scf.if 被 canonicalization 折叠而成                    │
  │     简单的 scf.if（内部无副作用）会被 arith canonicalizer   │
  │     折叠成 arith.select（if-conversion）                    │
  │                                                             │
  │  ③ 优化 pass 的产物                                        │
  │     SCCP / CSE / LICM 等 pass 在优化过程中会产生新的       │
  │     arith.select，并与已有的 scf.if 共享同一条件            │
  └─────────────────────────────────────────────────────────────┘

  ┌─ 本 pass 的定位：逆 if-conversion（但不是完全逆） ────────┐
  │                                                             │
  │  正常的 if-conversion（LLVM/MLIR 标准流程）：               │
  │    scf.if  →  arith.select    (简单的 if 变成一条指令)      │
  │                                                             │
  │  本 pass（特殊场景的精简）：                                │
  │    arith.select → scf.if       (仅当旁边已有同条件 if 时)  │
  │                                                             │
  │  它有意义是因为：scf.if 已经存在，把 select 吸收进来        │
  │  不增加分支开销，却避免了张量的双重物化。                   │
  └─────────────────────────────────────────────────────────────┘
""")


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  tritongpu-combine-tensor-select-and-if Pass 演示                  ║
║                                                                    ║
║  作用：将共享条件的 arith.select（张量）合并到 scf.if 中，        ║
║        让 if 直接 yield 相应的 true/false 值，                     ║
║        从而消除多余的 select 操作及其产生的中间张量。              ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    demo_text()
    demo_triton_opt()
    demo_concept()

    print("=" * 72)
    print("总结")
    print("=" * 72)
    print("""
  combine-tensor-select-and-if pass 的核心价值：

  ① 消除冗余操作 → arith.select 在张量上逐元素执行，会物化一个
     中间张量。合并到 scf.if 后，true/false 值由 yield 直接输出。

  ② 降低寄存器/共享内存压力 → 减少 SSA 值的数量。

  ③ 使能后续优化 → 合并后的 scf.if 可以被 TMEM hoisting 等 pass
     进一步优化（如将分配提到 if 之外复用）。

  实现要点：
    ▸ 仅处理张量类型的 select（标量 select 代价低不值得合并）
    ▸ select 和 if 必须在同一基本块，select 必须支配 if
    ▸ if 必须支配 select 的所有用户（保证 SSA 合法性）
    ▸ 预处理步骤 canonicalizeSelectUsersInSCFIf 处理 select 的
      用户在 if 内部的情况
""")
