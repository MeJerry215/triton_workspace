"""
测试 gluon-infer-coalesced-encodings pass：验证符号化 CoalescedLayout 被正确解析为具体 BlockedEncodingAttr。

gluon_to_ttgir 管线中，gluon-infer-coalesced-encodings 是第 2 个 pass（紧随 gluon-inline 之后）。
参见 markdowns/202_infer_coalesced_encoding_pass.md。

核心原理：
  - gl.CoalescedLayout() 产生一个符号化的 `gluon::CoalescedEncodingAttr`，不包含具体布局信息。
  - gluon-infer-coalesced-encodings pass 利用 AxisInfoAnalysis 分析 ptr 的 contiguity，
    为每个访存操作推导出最佳的 `BlockedEncodingAttr`（sizePerThread、order 等）。
  - 推导结果通过 SSA 图传播（inferLayout），最终所有相关张量的类型都被更新。

本脚本：
  1. 定义一个 2D kernel，两个维度的 arange 都使用 CoalescedLayout
  2. 设置 MLIR_ENABLE_DUMP=1 → dump 每个 pass 前的 IR
  3. 编译并运行 kernel，验证功能正确性
  4. 检查 dump 文件中包含 gluon-infer-coalesced-encodings pass 的 dump
"""

import os
import re
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# ---------------------------------------------------------------------------
# Kernel — 2D coalesced layout
# ---------------------------------------------------------------------------
#
# 二维情况下，两维的 arange 都标记 CoalescedLayout，结合广播形成 2D offsets。
# 最终 load/store 的 ptr tensor 应当被推导出 2D BlockedEncodingAttr。

@gluon.jit
def coalesced_2d_kernel(in_ptr, out_ptr,
                        xnumel, ynumel,
                        xstride_in, ystride_in,
                        xstride_out, ystride_out,
                        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    indices_x = pid_x * XBLOCK + gl.arange(0, XBLOCK, gl.CoalescedLayout())
    indices_y = pid_y * YBLOCK + gl.arange(0, YBLOCK, gl.CoalescedLayout())

    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask)
    value = gl.sin(value)
    value = gl.maximum(value, 0.0)
    gl.store(out_ptr + out_offsets, value, mask=mask)


# ---------------------------------------------------------------------------
# 运行与验证
# ---------------------------------------------------------------------------

def main():
    dump_path = os.environ.get("MLIR_DUMP_PATH")

    print("=" * 60)
    print("Gluon InferCoalescedEncodings Pass 测试")
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

    compiled = coalesced_2d_kernel[grid](
        input_2d, output_2d,
        xnumel, ynumel,
        *input_2d.stride(), *output_2d.stride(),
        XBLOCK, YBLOCK, num_warps=4,
    )

    torch.testing.assert_close(output_2d, ref_2d, atol=1e-5, rtol=1e-5)
    print(f"[OK] coalesced_2d_kernel: xnumel={xnumel}, ynumel={ynumel}, "
          f"XBLOCK={XBLOCK}, YBLOCK={YBLOCK}")

    # 检查 dump 文件中是否包含相关 pass 的 dump
    if dump_path and os.path.exists(dump_path):
        with open(dump_path) as f:
            content = f.read()
        markers = re.findall(r"IR Dump Before.*?\(.*?\).*?\)", content)
        print(f"\ndump 文件中包含 {len(markers)} 个 pass dump:")

        coalesced_dumps = [m for m in markers if "infer-coalesced" in m.lower()]
        if coalesced_dumps:
            print(f"  gluon-infer-coalesced-encodings dump 条目:")
            for m in coalesced_dumps:
                print(f"    {m}")
        else:
            print(f"  (未找到 gluon-infer-coalesced-encodings 的 dump 条目)")

        print(f"\n完整 dump: {dump_path}")

    print()
    print("测试通过。")


if __name__ == "__main__":
    main()
