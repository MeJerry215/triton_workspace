"""
Test: verify that tl.reduce(axis=...) + [:,None]/[None,:] indexing
works or fails as expected per SliceEncodingAttr semantics.

Expected:
  reduce(axis=1) + [:, None]  → OK   (slice dim=1 == expand axis=1)
  reduce(axis=0) + [None, :]  → OK   (slice dim=0 == expand axis=0)
  reduce(axis=1) + [None, :]  → FAIL (slice dim=1 != expand axis=0)
  reduce(axis=0) + [:, None]  → FAIL (slice dim=0 != expand axis=1)
"""

import torch
import triton
import triton.language as tl
import traceback


# ── 匹配 case: reduce(axis=1) + [:, None] ──────────────────────
@triton.jit
def kernel_match_axis1(
    x_ptr, y_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.arange(0, BLOCK_M)[:, None]
    col = tl.arange(0, BLOCK_N)[None, :]
    mask = (row < M) & (col < N)
    x = tl.load(x_ptr + row * N + col, mask=mask)

    sum_val = tl.sum(x, axis=1)        # [M], #slice<dim=1, parent=...>
    expanded = sum_val[:, None]         # [M, 1], expand axis=1 == slice dim=1 → OK

    # pointer = y_ptr + row has shape [M, 1]
    tl.store(y_ptr + row, expanded, mask=row < M)


# ── 匹配 case: reduce(axis=0) + [None, :] ──────────────────────
@triton.jit
def kernel_match_axis0(
    x_ptr, y_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.arange(0, BLOCK_M)[:, None]
    col = tl.arange(0, BLOCK_N)[None, :]
    mask = (row < M) & (col < N)
    x = tl.load(x_ptr + row * N + col, mask=mask)

    sum_val = tl.sum(x, axis=0)        # [N], #slice<dim=0, parent=...>
    expanded = sum_val[None, :]         # [1, N], expand axis=0 == slice dim=0 → OK

    # pointer = y_ptr + col has shape [1, N]
    tl.store(y_ptr + col, expanded, mask=col < N)


# ── 不匹配 case: reduce(axis=1) + [None, :] ────────────────────
@triton.jit
def kernel_mismatch_axis1_none_colon(
    x_ptr, y_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.arange(0, BLOCK_M)[:, None]
    col = tl.arange(0, BLOCK_N)[None, :]
    mask = (row < M) & (col < N)
    x = tl.load(x_ptr + row * N + col, mask=mask)

    sum_val = tl.sum(x, axis=1)        # [M], #slice<dim=1, ...>
    expanded = sum_val[None, :]         # [1, M], expand axis=0 ≠ slice dim=1 → EXPECT FAIL

    tl.store(y_ptr + col, expanded, mask=col < M)


# ── 不匹配 case: reduce(axis=0) + [:, None] ────────────────────
@triton.jit
def kernel_mismatch_axis0_colon_none(
    x_ptr, y_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.arange(0, BLOCK_M)[:, None]
    col = tl.arange(0, BLOCK_N)[None, :]
    mask = (row < M) & (col < N)
    x = tl.load(x_ptr + row * N + col, mask=mask)

    sum_val = tl.sum(x, axis=0)        # [N], #slice<dim=0, ...>
    expanded = sum_val[:, None]         # [N, 1], expand axis=1 ≠ slice dim=0 → EXPECT FAIL

    tl.store(y_ptr + row, expanded, mask=row < N)


def try_compile(kernel_fn, label: str, name: str):
    """Try to compile and run a kernel. Print detailed results."""
    M, N = 32, 64
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    y = torch.empty(M, N, device='cuda', dtype=torch.float32)
    try:
        kernel_fn[(1,)](x, y, M=M, N=N, BLOCK_M=M, BLOCK_N=N)
        print(f"  ✅ {label}: 编译 + 运行成功")
        return True
    except Exception as e:
        msg = str(e)
        # Extract first meaningful line
        first_line = msg.strip().split('\n')[0]
        # Check for the specific layout inference error
        if ("ExpandDimsOp" in msg or "Incompatible" in msg
            or "slice" in msg.lower()):
            print(f"  ❌ {label}: 编译失败（layout 推断拒绝）")
            print(f"      {first_line}")
        else:
            print(f"  ⚠️  {label}: 编译失败（其它原因）")
            print(f"      {first_line}")
        return False


def main():
    print("=" * 60)
    print("SliceEncodingAttr expand_dims 兼容性测试")
    print("=" * 60)
    print()

    # (kernel, label, expect_success)
    cases = [
        (kernel_match_axis1,
         "reduce(axis=1) + [:, None]   (dim=1 == axis=1)", True),
        (kernel_match_axis0,
         "reduce(axis=0) + [None, :]   (dim=0 == axis=0)", True),
        (kernel_mismatch_axis1_none_colon,
         "reduce(axis=1) + [None, :]   (dim=1 ≠ axis=0)", False),
        (kernel_mismatch_axis0_colon_none,
         "reduce(axis=0) + [:, None]   (dim=0 ≠ axis=1)", False),
    ]

    results = []
    for kernel_fn, label, expect_ok in cases:
        ok = try_compile(kernel_fn, label, kernel_fn.__name__)
        results.append((label, ok, expect_ok))

    print()
    print("-" * 60)
    all_pass = all(
        (ok == expect_ok) for _, ok, expect_ok in results
    )
    for label, ok, expect_ok in results:
        status = "PASS" if ok == expect_ok else "FAIL"
        actual = "成功" if ok else "失败"
        expected = "成功" if expect_ok else "失败（axis 不匹配）"
        print(f"  [{status}] {label}")
        print(f"          实际: {actual} | 预期: {expected}")

    print()
    if all_pass:
        print("✅ 全部通过！完全符合 SliceEncodingAttr 的 expand_dims 语义")
    else:
        print("💥 存在不符合语义的用例")


if __name__ == "__main__":
    main()
