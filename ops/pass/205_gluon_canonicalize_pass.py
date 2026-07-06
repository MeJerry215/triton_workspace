"""
测试 gluon-canonicalize pass：验证 CanonicalizeMaskedLoadPattern 等 Canonicalization Pattern 的效果。

gluon_to_ttgir 管线中，gluon-canonicalize 被调用两次（第 5 和第 8 个 pass）。
参见 markdowns/205_gluon_canonicalize.md。

本脚本以 CanonicalizeMaskedLoadPattern 为例，演示一个具体的 Canonicalization Pattern 的工作原理：

  模式 1: load(ptr, splat(true), ...)  →  load(ptr, ...)              [去掉恒真 mask]
  模式 2: load(ptr, splat(false), other, ...) →  other                [恒假 mask → 替换为 other]
  模式 3: store(ptr, value, splat(true), ...) →  store(ptr, value, ...) [去掉恒真 mask]
  模式 4: store(ptr, value, splat(false), ...) →  [无操作]             [恒假 mask → 删除 store]

IR 变换示例（CanonicalizeMaskedLoadPattern）：

  // 变换前：mask 是 arith.constant dense<true>（全 1）
  %mask = arith.constant dense<true> : tensor<128xi1>
  %val = tt.load %ptr, %mask
      : tensor<128x!tt.ptr<f32>> → tensor<128xf32>

  // 变换后：mask 被消除
  %val = tt.load %ptr
      : tensor<128x!tt.ptr<f32>> → tensor<128xf32>

广播/ExpandDims 模式（BroadcastSplatPattern / ExpandDimsCanonicalize）：

  // 变换前
  %s = tt.splat %x : i32 → tensor<1xi32>
  %b = tt.broadcast %s : tensor<1xi32> → tensor<128xi32>

  // 变换后：broadcast(splat(x)) → splat(x)
  %b = tt.splat %x : i32 → tensor<128xi32>
"""

import os
import re
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# ---------------------------------------------------------------------------
# Kernel — 1D copy with explicit mask
# ---------------------------------------------------------------------------
#
# 本 kernel 在 load/store 时使用 mask 进行边界检查。
# 当 numel 是 BLOCK 的整数倍时，mask 恒为 true。
# gluon-canonicalize 中的 CanonicalizeMaskedLoadPattern 和
# CanonicalizeMaskedStorePattern 应当消除这些冗余的 mask 操作。
#
# 但注意：在 Triton 的编译流程中，mask 通常由 arith.cmpi 比较产生
# （运行时计算的），而非 arith.constant 常量，因此 CanonicalizeMaskedLoadPattern
# 在实际 pipeline 中可能不会立即触发。该 pattern 主要处理前端或其他 pass
# 已经常量折叠后产生的 splat(true)/splat(false) mask。

@gluon.jit
def masked_copy_kernel(in_ptr, out_ptr, numel, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    offsets = pid * BLOCK + gl.arange(0, BLOCK, gl.CoalescedLayout())
    mask = offsets < numel
    value = gl.load(in_ptr + offsets, mask=mask)
    gl.store(out_ptr + offsets, value, mask=mask)


# ---------------------------------------------------------------------------
# Kernel — 展示 arith 常量折叠模式
# ---------------------------------------------------------------------------
#
# triton 中的 scalar 计算（如 pid * BLOCK + constant）会被 gluon-canonicalize
# 中的 arith 规范化模式简化（加 0 消除、乘 1 消除、常量折叠等）。

@gluon.jit
def constant_fold_kernel(out_ptr, numel, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    offsets = pid * BLOCK + gl.arange(0, BLOCK, gl.CoalescedLayout())
    mask = offsets < numel
    # arith canonicalization: select(true, a, b) → a
    # 如果 mask 恒真，select 会退化为直接传递 a
    value = gl.full([BLOCK], 3.14, dtype=gl.float32)
    gl.store(out_ptr + offsets, value, mask=mask)


# ---------------------------------------------------------------------------
# 运行与验证
# ---------------------------------------------------------------------------

def main():
    dump_path = os.environ.get("MLIR_DUMP_PATH")

    print("=" * 60)
    print("Gluon Canonicalize Pass 测试")
    print("=" * 60)
    print(f"TRITON_ALWAYS_COMPILE = {os.environ.get('TRITON_ALWAYS_COMPILE', '(unset)')}")
    print(f"MLIR_ENABLE_DUMP      = {os.environ.get('MLIR_ENABLE_DUMP', '(unset)')}")
    print(f"MLIR_DUMP_PATH        = {dump_path or '(unset → stderr)'}")
    print()

    BLOCK = 1024
    numel = 4096
    grid = (triton.cdiv(numel, BLOCK),)

    # ---- 测试 1: masked_copy_kernel ----
    print("-" * 60)
    print("测试 1: masked_copy_kernel")
    print("-" * 60)

    input_1d = torch.randn(numel, device="cuda")
    output_1d = torch.zeros_like(input_1d)

    compiled = masked_copy_kernel[grid](
        input_1d, output_1d,
        numel, BLOCK, num_warps=4,
    )

    torch.testing.assert_close(output_1d, input_1d, atol=1e-5, rtol=1e-5)
    print(f"[OK] masked_copy_kernel: numel={numel}, BLOCK={BLOCK}")

    # ---- 测试 2: constant_fold_kernel ----
    print("-" * 60)
    print("测试 2: constant_fold_kernel")
    print("-" * 60)

    output_2 = torch.zeros(numel, device="cuda")
    ref_2 = torch.full((numel,), 3.14, device="cuda")

    compiled2 = constant_fold_kernel[grid](
        output_2, numel, BLOCK, num_warps=4,
    )

    torch.testing.assert_close(output_2, ref_2, atol=1e-5, rtol=1e-5)
    print(f"[OK] constant_fold_kernel: numel={numel}, BLOCK={BLOCK}")

    # ---- 检查 dump 文件中 gluon-canonicalize pass 的 dump ----
    if dump_path and os.path.exists(dump_path):
        with open(dump_path) as f:
            content = f.read()

        # 统计所有的 pass dump 条目
        markers = re.findall(
            r"IR Dump Before.*?\(.*?gluon-canonicalize.*?\).*?\)",
            content,
        )
        print(f"\ndump 文件中包含 {len(markers)} 个 gluon-canonicalize 相关的 dump 条目:")
        for m in markers:
            print(f"  ▸ {m}")

        if not markers:
            print("  (未找到 gluon-canonicalize 的 dump 条目)")
            # 列出所有 pass dump 帮助诊断
            all_markers = re.findall(
                r"IR Dump Before.*?\(.*?\).*?\)", content
            )
            print(f"\n共找到 {len(all_markers)} 个 pass dump:")
            for m in all_markers:
                print(f"    {m}")

        print(f"\n完整 dump: {dump_path}")

    print()
    print("所有测试通过。")


# ---------------------------------------------------------------------------
# CanonicalizeMaskedLoadPattern 详解
# ---------------------------------------------------------------------------
#
# 以下是从 triton/lib/Dialect/Triton/IR/Ops.cpp 提取的完整 C++ 实现，方便对照阅读：
#
# // load(ptr, splat(1), ...)        -> load(ptr, ...)
# // load(ptr, splat(0), other, ...) -> other
# struct CanonicalizeMaskedLoadPattern : public OpRewritePattern<LoadOp> {
#   CanonicalizeMaskedLoadPattern(MLIRContext *context)
#       : OpRewritePattern<LoadOp>(context, 1) {}
#
#   LogicalResult matchAndRewrite(LoadOp loadOp,
#                                 PatternRewriter &rewriter) const override {
#     auto mask = loadOp.getMask();
#     if (!mask) return failure();
#
#     auto constantMask = mask.getDefiningOp<arith::ConstantOp>();
#     if (!constantMask) return failure();
#
#     auto splatMask = mlir::dyn_cast<SplatElementsAttr>(constantMask.getValue());
#     if (!splatMask) return failure();
#
#     if (splatMask.getSplatValue<IntegerAttr>().getValue() == true) {
#       // mask = splat(1) → 去掉 mask
#       rewriter.replaceOpWithNewOp<LoadOp>(
#           loadOp, loadOp.getType(), loadOp.getPtr(),
#           Value(), Value(),
#           loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
#           loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
#     } else {
#       // mask = splat(0) → 替换为 other
#       auto otherVal = loadOp.getOther();
#       if (!otherVal) return failure();
#       rewriter.replaceOp(loadOp, otherVal);
#     }
#     return success();
#   }
# };
#
# 关键设计决策：
# 1. 当 mask = splat(true) 时：生成一个全新的 LoadOp，去掉 mask 和 other 参数
# 2. 当 mask = splat(false) 时：用 other 值替换整个 load op
# 3. 只处理 arith.constant + SplatElementsAttr 产生的常量 mask，
#    不处理运行时计算的 mask（如 arith.cmpi 产生的动态 mask）


if __name__ == "__main__":
    main()
