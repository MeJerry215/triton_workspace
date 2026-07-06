"""
测试 gluon-resolve-auto-encodings pass：验证符号化 AutoEncodingAttr 通过 set_auto_layout 锚点
被正确传播并解析为具体编码，随后 set_auto_layout 操作被消除。

gluon_to_ttgir 管线中，gluon-resolve-auto-encodings 是第 3 个 pass（紧随 inline 和 infer-coalesced-encodings）。
参见 markdowns/203_resolve_auto_encoding_pass.md。

核心原理：
  - gl.AutoLayout() 产生一个符号化的 `gluon::AutoEncodingAttr`，不包含具体布局信息。
  - 用户通过 `gl.set_auto_layout(tensor, concrete_layout)` 在 auto-encoded 张量和具体 layout 之间建立桥接。
  - gluon-resolve-auto-encodings pass：
      1. 以 set_auto_layout 的 src value → result encoding 为种子（seed），
         通过 inferLayout 沿 SSA 图向后/向前传播具体编码。
      2. 传播完成后删除所有 set_auto_layout 操作（此时 src 与 dst 类型一致）。
      3. 通过 doubleCheckEncodings 确保不再有任何 AutoEncodingAttr 残留。

本脚本使用 2D kernel：arange 使用 AutoLayout，指针和 mask 通过 set_auto_layout
桥接到 CoalescedLayout。gluon-to-ttgir 管线依次：
  1. gluon-inline                     — 无 helper 调用，无变化
  2. gluon-infer-coalesced-encodings  — CoalescedLayout → BlockedEncodingAttr
  3. gluon-resolve-auto-encodings     — AutoEncodingAttr 传播 + set_auto_layout 消除
  （后续 pass 完成 TMA lowering、canonicalize 等）
"""

import os
import re
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# ---------------------------------------------------------------------------
# Kernel — 2D AutoLayout with explicit set_auto_layout
# ---------------------------------------------------------------------------
#
# arange 使用 AutoLayout()，产生 tensor<..., #gluon.auto_encoding>。
# set_auto_layout 将 auto-encoded 的 ptr/mask 桥接到 CoalescedLayout，
# 这是 ResolveAutoEncodingsPass 的种子来源。

@gluon.jit
def auto_layout_2d_kernel(in_ptr, out_ptr,
                          xnumel, ynumel,
                          xstride_in, ystride_in,
                          xstride_out, ystride_out,
                          XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    # indices 使用 AutoLayout → tensor<128xi32, #gluon.auto_encoding>
    indices_x = pid_x * XBLOCK + gl.arange(0, XBLOCK, gl.AutoLayout())
    indices_y = pid_y * YBLOCK + gl.arange(0, YBLOCK, gl.AutoLayout())

    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    # mask 也是 auto layout
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    # set_auto_layout 将 auto-encoded ptr 桥接到 CoalescedLayout
    # 这是 ResolveAutoEncodingsPass 的种子来源
    in_ptrs = gl.set_auto_layout(in_ptr + in_offsets, gl.CoalescedLayout())
    value = gl.load(in_ptrs, mask=mask)

    value = gl.sin(value)
    value = gl.maximum(value, 0.0)

    out_ptrs = gl.set_auto_layout(out_ptr + out_offsets, gl.CoalescedLayout())
    out_mask_layouted = gl.set_auto_layout(mask, gl.CoalescedLayout())
    gl.store(out_ptrs, value, mask=out_mask_layouted)


# ---------------------------------------------------------------------------
# 运行与验证
# ---------------------------------------------------------------------------

def main():
    dump_path = os.environ.get("MLIR_DUMP_PATH")

    print("=" * 60)
    print("Gluon ResolveAutoEncodings Pass 测试")
    print("=" * 60)
    print(f"TRITON_ALWAYS_COMPILE = {os.environ.get('TRITON_ALWAYS_COMPILE', '(unset)')}")
    print(f"MLIR_ENABLE_DUMP      = {os.environ.get('MLIR_ENABLE_DUMP', '(unset)')}")
    print(f"MLIR_DUMP_PATH        = {dump_path or '(unset → stderr)'}")
    print()

    XBLOCK, YBLOCK = 128, 256
    xnumel, ynumel = 1000, 2000
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))

    input_2d = torch.randn((xnumel, ynumel), device="cuda")
    output_2d = torch.zeros_like(input_2d)
    ref_2d = torch.maximum(torch.sin(input_2d), torch.tensor(0.0, device="cuda"))

    compiled = auto_layout_2d_kernel[grid](
        input_2d, output_2d,
        xnumel, ynumel,
        *input_2d.stride(), *output_2d.stride(),
        XBLOCK, YBLOCK, num_warps=4,
    )

    torch.testing.assert_close(output_2d, ref_2d, atol=1e-5, rtol=1e-5)
    print(f"[OK] auto_layout_2d_kernel: xnumel={xnumel}, ynumel={ynumel}, "
          f"XBLOCK={XBLOCK}, YBLOCK={YBLOCK}")

    # 检查 dump 文件中是否包含相关 pass 的 dump
    if dump_path and os.path.exists(dump_path):
        with open(dump_path) as f:
            content = f.read()
        markers = re.findall(r"IR Dump Before.*?\(.*?\).*?\)", content)
        print(f"\ndump 文件中包含 {len(markers)} 个 pass dump:")

        resolve_dumps = [m for m in markers if "resolve-auto" in m.lower()]
        if resolve_dumps:
            print(f"  gluon-resolve-auto-encodings dump 条目:")
            for m in resolve_dumps:
                print(f"    {m}")
        else:
            print(f"  (未找到 gluon-resolve-auto-encodings 的 dump 条目)")

        print(f"\n完整 dump: {dump_path}")

    print()
    print("测试通过。")


if __name__ == "__main__":
    main()
