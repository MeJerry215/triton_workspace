#
# Copyright (c) 2025 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Licensed under the Apache License, Version 2.0
#

"""PyTorch reference for sparse GQA MLA Flash-Attention (FA) with KV cache.

This module mirrors ``gqa_flash_mla_sparse_fa_kernel`` in
``30_gqa_flash_mla_with_kv_cache.py``: sparse top-k gather from a paged KV
cache, GQA (multiple Q heads share one KV head), MLA (K and V share storage).

Online softmax derivation
-------------------------

Goal: compute attention output without materializing the full score matrix.

.. math::

    O = \\mathrm{softmax}(S) V, \\quad S_{ij} = \\mathrm{scale} \\cdot q_i^\\top k_j

Split the key indices into groups :math:`G_1, G_2, \\ldots, G_T` (here each
group has up to ``GROUP_SIZE`` sparse tokens).  For group :math:`t` define

.. math::

    m_t = \\max_{j \\in G_t} S_{ij}, \\quad
    \\tilde{P}_{ij} = \\exp(S_{ij} - m_t), \\quad
    \\ell_t = \\sum_{j \\in G_t} \\tilde{P}_{ij}, \\quad
    \\tilde{O}_t = \\tilde{P}_t V_t

:math:`\\tilde{P}_t` is **not** normalized inside the group; that is intentional.

Maintain running state :math:`(m, \\ell, \\mathrm{acc})` across groups:

.. math::

    m^{(0)} = -\\infty,\\; \\ell^{(0)} = 0,\\; \\mathrm{acc}^{(0)} = 0

For each group :math:`t`:

.. math::

    m' &= \\max(m^{(t-1)}, m_t) \\\\
    \\alpha &= \\exp(m^{(t-1)} - m') \\\\
    \\beta   &= \\exp(m_t - m') \\\\
    \\mathrm{acc}^{(t)} &= \\mathrm{acc}^{(t-1)} \\cdot \\alpha
                         + \\tilde{O}_t \\cdot \\beta \\\\
    \\ell^{(t)} &= \\ell^{(t-1)} \\cdot \\alpha + \\ell_t \\cdot \\beta \\\\
    m^{(t)} &= m'

After all groups, :math:`O = \\mathrm{acc}^{(T)} / \\ell^{(T)}`.

**Correctness sketch.**  Let :math:`M = \\max_t m_t` be the global max.  After
processing groups :math:`1..t`,

.. math::

    \\ell^{(t)} = \\sum_{g=1}^{t} \\sum_{j \\in G_g} \\exp(S_{ij} - M), \\quad
    \\mathrm{acc}^{(t)} = \\sum_{g=1}^{t} \\sum_{j \\in G_g}
        \\exp(S_{ij} - M)\\, v_j

The rescale factors :math:`\\alpha, \\beta` re-base earlier partial sums from
:math:`m^{(t-1)}` to the new global max :math:`m'`, so each update is equivalent
to extending the softmax denominator and numerator under a common
:math:`\\exp(\\cdot - M)` scale.  This is the same recurrence used in Flash
Attention block merging and in the Triton FA kernel
(``running_max``, ``running_denom``, ``acc``).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

TOPK = 2048
_DEFAULT_GROUP_SIZE = 64


def _load_triton_module():
    """Import ``30_gqa_flash_mla_with_kv_cache`` without package layout."""
    path = Path(__file__).with_name("30_gqa_flash_mla_with_kv_cache.py")
    spec = importlib.util.spec_from_file_location("gqa_flash_mla_30", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gather_kv(
    kv_cache: torch.Tensor,
    token_indices: torch.Tensor,
    page_block_size: int,
) -> torch.Tensor:
    """Gather KV rows from a paged cache.

    Args:
        kv_cache: ``[num_blocks, page_size, num_kv_heads, D_V]``
        token_indices: ``[G]`` global token indices (int32/int64)
        page_block_size: tokens per page

    Returns:
        ``[G, D_V]`` gathered rows (kv head 0).
    """
    block_idx = token_indices // page_block_size
    pos = token_indices - block_idx * page_block_size
    return kv_cache[block_idx, pos, 0, :]


@torch.inference_mode()
def gqa_flash_mla_sparse_with_kv_cache_fa_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    *,
    group_size: int = _DEFAULT_GROUP_SIZE,
    d_v: int = 512,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FA-style sparse GQA MLA with online softmax (group-by-group merge).

    Semantics match ``gqa_flash_mla_sparse_with_kv_cache_fa`` in file 30.
    """
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dim() == 4 and q.size(1) == 1 and q.size(3) == d_v
    num_tokens, _, num_heads, head_dim = q.shape
    assert head_dim == d_v
    page_block_size = kv_cache.size(1)

    if out is None:
        out = torch.empty((num_tokens, 1, num_heads, d_v), dtype=q.dtype, device=q.device)
    else:
        assert out.shape == (num_tokens, 1, num_heads, d_v)

    neg_inf = torch.tensor(float("-inf"), device=q.device)

    for batch_idx in range(num_tokens):
        context_len = int(topk_length[batch_idx].item())
        if context_len <= 0:
            out[batch_idx].zero_()
            continue

        idx_all = indices[batch_idx, 0, :context_len]
        num_groups = (context_len + group_size - 1) // group_size

        # All heads share the same sparse indices (GQA); batch Q across heads.
        q_heads = q[batch_idx, 0, :, :].float()  # [H_Q, D_V]
        running_max = torch.full((num_heads,), float("-inf"), device=q.device)
        running_denom = torch.zeros(num_heads, dtype=torch.float32, device=q.device)
        acc = torch.zeros(num_heads, head_dim, dtype=torch.float32, device=q.device)

        for group_idx in range(num_groups):
            slot_start = group_idx * group_size
            slot_end = min(slot_start + group_size, context_len)

            idx_vals = idx_all[slot_start:slot_end]
            valid = idx_vals >= 0
            if not valid.any():
                continue

            safe_idx = torch.where(valid, idx_vals, torch.zeros_like(idx_vals))
            k = _gather_kv(kv_cache, safe_idx, page_block_size).float()  # [G, D_V]
            scores = (q_heads @ k.T) * sm_scale  # [H_Q, G]
            scores = torch.where(valid.unsqueeze(0), scores, neg_inf)

            group_max = scores.max(dim=-1).values  # [H_Q]
            exp_scores = torch.where(
                valid.unsqueeze(0),
                torch.exp(scores - group_max.unsqueeze(-1)),
                torch.zeros_like(scores),
            )
            exp_sum = exp_scores.sum(dim=-1)  # [H_Q]
            head_active = exp_sum > 0
            if not head_active.any():
                continue

            partial_unnorm = exp_scores @ k  # [H_Q, D_V]

            new_max = torch.where(head_active, torch.maximum(running_max, group_max), running_max)
            scale_old = torch.exp(running_max - new_max)
            scale_group = torch.where(head_active, torch.exp(group_max - new_max), torch.zeros_like(group_max))
            acc = acc * scale_old.unsqueeze(-1) + partial_unnorm * scale_group.unsqueeze(-1)
            running_denom = running_denom * scale_old + exp_sum * scale_group
            running_max = new_max

        has_valid = running_denom > 0
        out[batch_idx, 0, :, :] = torch.where(
            has_valid.unsqueeze(-1),
            (acc / running_denom.unsqueeze(-1)).to(q.dtype),
            torch.zeros(num_heads, head_dim, dtype=q.dtype, device=q.device),
        )

    return out


@torch.inference_mode()
def gqa_flash_mla_sparse_with_kv_cache_reference_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    *,
    d_v: int = 512,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Naive reference: gather all valid keys, full softmax, then PV."""
    assert q.dtype in (torch.float16, torch.bfloat16)
    num_tokens, _, num_heads, head_dim = q.shape
    page_block_size = kv_cache.size(1)

    if out is None:
        out = torch.empty((num_tokens, 1, num_heads, d_v), dtype=q.dtype, device=q.device)
    else:
        assert out.shape == (num_tokens, 1, num_heads, d_v)

    for batch_idx in range(num_tokens):
        context_len = int(topk_length[batch_idx].item())
        if context_len <= 0:
            out[batch_idx].zero_()
            continue

        idx_all = indices[batch_idx, 0, :context_len]
        valid_mask = idx_all >= 0
        if not valid_mask.any():
            out[batch_idx].zero_()
            continue

        idx_valid = idx_all[valid_mask]
        k = _gather_kv(kv_cache, idx_valid, page_block_size).float()  # [N, D_V]
        q_heads = q[batch_idx, 0, :, :].float()  # [H_Q, D_V]
        scores = (q_heads @ k.T) * sm_scale  # [H_Q, N]
        probs = torch.softmax(scores, dim=-1)
        out[batch_idx, 0, :, :] = (probs @ k).to(q.dtype)

    return out


def _parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(" ", "").split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check PyTorch FA GQA reference vs Triton FA kernel.",
    )
    parser.add_argument("--batch-size", "--bs", default="4")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--valid-topk", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=_DEFAULT_GROUP_SIZE)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--check",
        choices=("fa", "ref", "all"),
        default="all",
        help="fa=torch vs triton FA; ref=torch FA vs naive; all=both",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for Triton comparison; running CPU-only self-check skipped.")
        return

    mod = _load_triton_module()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    batch_sizes = _parse_int_list(args.batch_size)
    if not batch_sizes:
        parser.error("--batch-size must contain at least one integer")

    torch.manual_seed(0)
    for batch_size in batch_sizes:
        q, kv_cache, indices, sm_scale, topk_length = mod.make_decode_inputs(
            valid_topk=args.valid_topk,
            batch_size=batch_size,
            num_heads=args.num_heads,
            dtype=dtype,
        )
        common = (q, kv_cache, indices, topk_length, sm_scale)

        fa_torch = gqa_flash_mla_sparse_with_kv_cache_fa_torch(
            *common,
            group_size=args.group_size,
        )
        torch.cuda.synchronize()

        if args.check in ("ref", "all"):
            ref_torch = gqa_flash_mla_sparse_with_kv_cache_reference_torch(*common)
            torch.testing.assert_close(fa_torch, ref_torch, atol=1e-2, rtol=1e-2)
            print(
                f"check passed: online FA torch vs naive softmax "
                f"(bs={batch_size}, topk={args.valid_topk}, group={args.group_size})"
            )

        if args.check in ("fa", "all"):
            fa_triton = mod.gqa_flash_mla_sparse_with_kv_cache_fa(*common)
            torch.cuda.synchronize()
            torch.testing.assert_close(fa_torch, fa_triton, atol=1e-2, rtol=1e-2)
            print(
                f"check passed: torch FA vs triton FA "
                f"(bs={batch_size}, topk={args.valid_topk}, group={args.group_size})"
            )


if __name__ == "__main__":
    main()


import triton
import triton.language as tl


