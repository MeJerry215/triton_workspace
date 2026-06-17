"""
64×64×64 矩阵乘（M×K @ K×N），使用当前推荐的 tl.dot 显式 API：
  - out_dtype：指定点积累加/输出元素类型（常为 fp32 累加）
  - 可选 acc= 做多阶段 K 切分；此处 K=64 一次吃满，无需 K 循环
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def dot_64_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BN: tl.constexpr,
):
    """单 program 覆盖整块 64×64×64：C = A @ B，A:[M,K], B:[K,N], C:[M,N]。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BN + tl.arange(0, BN)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BN)
    a = tl.load(
        a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
    )
    b = tl.load(
        b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
    )
    # 新 API 要点：显式 out_dtype（fp16×fp16 → fp32 累加）
    c = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
    )


def matmul_64x64x64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == (64, 64) and b.shape == (64, 64)
    m, k = a.shape
    k2, n = b.shape
    assert k == k2 == 64 and n == 64
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    grid = (1, 1)
    dot_64_kernel[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BN=64,
        num_warps=4,
    )
    return c


def test_matmul_64_matches_torch():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example.")
    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    got = matmul_64x64x64(a, b)
    ref = torch.matmul(a, b).float()
    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_matmul_64_matches_torch()
    print("ok: 64×64×64 tl.dot matches torch.matmul")
