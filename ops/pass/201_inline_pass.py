"""
测试 gluon-inline pass：验证 `@gluon.jit` helper function 被正确内联。

gluon_to_ttgir 管线中，gluon-inline 是第 1 个 pass（见 markdowns/201_inline_pass.md）。

核心原理：
  - AutoLayout / CoalescedLayout 等 encoding 不允许跨越 function boundary。
  - 因此 gluon_to_ttgir 的入口处必须先 inline 所有嵌套 @gluon.jit 调用，
    使 layout 推断（infer_coalesced_encodings、resolve_auto_encodings）能
    在单一函数体内正常工作。

本脚本：
  1. 定义 helper `@gluon.jit` 函数 get_offsets_and_mask
  2. 定义主 kernel `inline_coalesced_kernel` 调用 helper
  3. 设置 MLIR_ENABLE_DUMP=1 → dump 每个 pass 前的 IR
  4. 编译并运行 kernel，验证功能正确性
"""

import os
import re
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# ---------------------------------------------------------------------------
# 1) Helper 函数（会被 gluon-inline 内联进调用者）
# ---------------------------------------------------------------------------

@gluon.jit
def get_offsets_and_mask(pid, numel, BLOCK: gl.constexpr):
    """
    计算一维 tile 的 offsets 和 mask。

    使用 CoalescedLayout() — inline 之后 infer_coalesced_encodings 会
    算出具体的 #blocked layout，使 gmem 访存合并。
    """
    offsets = pid * BLOCK + gl.arange(0, BLOCK, gl.CoalescedLayout())
    mask = offsets < numel
    return offsets, mask


# ---------------------------------------------------------------------------
# 2) 主 Kernel — 调用 helper
# ---------------------------------------------------------------------------

@gluon.jit
def inline_coalesced_kernel(in_ptr, out_ptr, numel, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    offsets, mask = get_offsets_and_mask(pid, numel, BLOCK)
    value = gl.load(in_ptr + offsets, mask=mask)
    gl.store(out_ptr + offsets, value, mask=mask)


# ---------------------------------------------------------------------------
# 3) 运行与验证
# ---------------------------------------------------------------------------

def main():
    dump_path = os.environ.get("MLIR_DUMP_PATH")

    print("=" * 60)
    print("Gluon Inline Pass 测试")
    print("=" * 60)
    print(f"TRITON_ALWAYS_COMPILE = {os.environ.get('TRITON_ALWAYS_COMPILE', '(unset)')}")
    print(f"MLIR_ENABLE_DUMP      = {os.environ.get('MLIR_ENABLE_DUMP', '(unset)')}")
    print(f"MLIR_DUMP_PATH        = {dump_path or '(unset → stderr)'}")
    print()

    numel, BLOCK = 4096, 1024
    grid = (triton.cdiv(numel, BLOCK),)

    in_ptr = torch.randn(numel, device="cuda")
    out_ptr = torch.empty_like(in_ptr)

    compiled = inline_coalesced_kernel[grid](
        in_ptr, out_ptr, numel, BLOCK, num_warps=4
    )

    torch.testing.assert_close(in_ptr, out_ptr, atol=0, rtol=0)
    print(f"[OK] inline_coalesced_kernel: numel={numel}, BLOCK={BLOCK}")

    # 检查 dump 文件
    if dump_path and os.path.exists(dump_path):
        with open(dump_path) as f:
            content = f.read()
        markers = re.findall(r"IR Dump Before.*?\(.*?\).*?\)", content)
        print(f"\ndump 文件中包含 {len(markers)} 个 pass dump:")
        for m in markers[:9]:  # gluon_to_ttgir 一共 9 个 pass
            print(f"  {m}")
        print(f"\n完整 dump: {dump_path}")

    print()
    print("测试通过。")


if __name__ == "__main__":
    main()
