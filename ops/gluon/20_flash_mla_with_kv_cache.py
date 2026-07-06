#
# Copyright (c) 2025 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Licensed under the Apache License, Version 2.0
#

from typing import Iterable, Literal, Optional

import argparse
import math
import sys

import torch
import triton
import triton.language as tl
from triton.experimental.gluon import language as gl
from triton.experimental import gluon
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

# Set True to autotune kernel configs; False uses _DEFAULT_*_KERNEL_CONFIG below.
# Can also enable by passing --tune on the command line.
ENABLE_AUTOTUNE = False
if "--tune" in sys.argv:
    ENABLE_AUTOTUNE = True


def ensure_fp16_bf16_contiguous(x: torch.Tensor) -> None:
    assert x.is_contiguous()


def ensure_fp32_contiguous(x: torch.Tensor) -> None:
    assert x.is_contiguous()


def ensure_int32_contiguous(x: torch.Tensor) -> None:
    assert x.is_contiguous()


def align_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) // alignment * alignment


_DEFAULT_FD_KERNEL_CONFIG = triton.Config(
    {"BLOCK_H": 16, "GROUP_SIZE": 64, "BLOCK_D": 128, "BLOCK_V": 128},
    num_warps=8,
    num_stages=1,
)
_DEFAULT_RD_KERNEL_CONFIG = triton.Config({"BLOCK_V": 256}, num_warps=1)

_DEFAULT_FD_CONFIG = {
    **_DEFAULT_FD_KERNEL_CONFIG.kwargs,
    "num_warps": _DEFAULT_FD_KERNEL_CONFIG.num_warps,
}

_DEFAULT_FA_KERNEL_CONFIG = triton.Config(
    {"BLOCK_H": 16, "GROUP_SIZE": 64, "BLOCK_D": 128, "BLOCK_V": 128},
    num_warps=8,
    num_stages=1,
)
_DEFAULT_V2_GLUON_KERNEL_CONFIG = triton.Config(
    {"BLOCK_H": 16, "GROUP_SIZE": 64, "BLOCK_D": 128, "BLOCK_V": 128, "CTA_WARPS": 4},
    num_warps=4,
    num_stages=1,
)

# tl.arange(0, N) requires N to be a power of 2.
_BLOCK_HS = [16, 32, 64]
_GROUP_SIZES = [16, 32, 64]
_BLOCK_DS = [16, 32, 64,]
_BLOCK_VS = [128, 256, 512]
_FD_NUM_WARPS = [1, 2, 4, 8]

# FA single-kernel: larger GROUP_SIZE (in-kernel loop) and num_stages search.
_FA_GROUP_SIZES = [128, ]
_FA_NUM_WARPS = [1, 2, 4, 8, 16]
_FA_NUM_STAGES = [1,]

_AUTOTUNE_CONFIGS: list[triton.Config] = []
for block_h in _BLOCK_HS:
    for group_size in _GROUP_SIZES:
        for block_d in _BLOCK_DS:
            for block_v in _BLOCK_VS:
                for num_warps in _FD_NUM_WARPS:
                    _AUTOTUNE_CONFIGS.append(
                        triton.Config(
                            {
                                "BLOCK_H": block_h,
                                "GROUP_SIZE": group_size,
                                "BLOCK_D": block_d,
                                "BLOCK_V": block_v,
                            },
                            num_warps=num_warps,
                            num_stages=1,
                        )
                    )

_FA_AUTOTUNE_CONFIGS: list[triton.Config] = []
for block_h in _BLOCK_HS:
    for group_size in _FA_GROUP_SIZES:
        for block_d in _BLOCK_DS:
            for block_v in _BLOCK_VS:
                for num_warps in _FA_NUM_WARPS:
                    for num_stages in _FA_NUM_STAGES:
                        _FA_AUTOTUNE_CONFIGS.append(
                            triton.Config(
                                {
                                    "BLOCK_H": block_h,
                                    "GROUP_SIZE": group_size,
                                    "BLOCK_D": block_d,
                                    "BLOCK_V": block_v,
                                },
                                num_warps=num_warps,
                                num_stages=num_stages,
                            )
                        )

_V2_GLUON_BASE_CONFIGS = [
    {"BLOCK_H": 16, "GROUP_SIZE": 32, "BLOCK_D": 32, "BLOCK_V": 128},
    {"BLOCK_H": 16, "GROUP_SIZE": 32, "BLOCK_D": 64, "BLOCK_V": 128},
    {"BLOCK_H": 16, "GROUP_SIZE": 64, "BLOCK_D": 32, "BLOCK_V": 128},
    {"BLOCK_H": 16, "GROUP_SIZE": 64, "BLOCK_D": 64, "BLOCK_V": 128},
    {"BLOCK_H": 16, "GROUP_SIZE": 64, "BLOCK_D": 128, "BLOCK_V": 128},
    {"BLOCK_H": 16, "GROUP_SIZE": 64, "BLOCK_D": 128, "BLOCK_V": 256},
    {"BLOCK_H": 16, "GROUP_SIZE": 128, "BLOCK_D": 64, "BLOCK_V": 128},
    {"BLOCK_H": 16, "GROUP_SIZE": 128, "BLOCK_D": 128, "BLOCK_V": 128},
    {"BLOCK_H": 16, "GROUP_SIZE": 128, "BLOCK_D": 128, "BLOCK_V": 256},
    {"BLOCK_H": 16, "GROUP_SIZE": 256, "BLOCK_D": 64, "BLOCK_V": 128},
    {"BLOCK_H": 16, "GROUP_SIZE": 256, "BLOCK_D": 128, "BLOCK_V": 128},
    {"BLOCK_H": 16, "GROUP_SIZE": 256, "BLOCK_D": 128, "BLOCK_V": 256},
    {"BLOCK_H": 16, "GROUP_SIZE": 256, "BLOCK_D": 128, "BLOCK_V": 512},
]
_V2_GLUON_NUM_WARPS = [2, 4, 8]
_V2_GLUON_CONFIGS = [
    triton.Config({**kwargs, "CTA_WARPS": num_warps}, num_warps=num_warps, num_stages=1)
    for kwargs in _V2_GLUON_BASE_CONFIGS
    for num_warps in _V2_GLUON_NUM_WARPS
]

_TUNE_FD = False
_TUNE_FA = False
_TUNE_RD = False
_TUNE_V2_GLUON = ENABLE_AUTOTUNE

_FD_KERNEL_CONFIGS = _AUTOTUNE_CONFIGS if _TUNE_FD else [_DEFAULT_FD_KERNEL_CONFIG]
_FA_KERNEL_CONFIGS = _FA_AUTOTUNE_CONFIGS if _TUNE_FA else [_DEFAULT_FA_KERNEL_CONFIG]
_V2_GLUON_KERNEL_CONFIGS = _V2_GLUON_CONFIGS if _TUNE_V2_GLUON else [_DEFAULT_V2_GLUON_KERNEL_CONFIG]


def _is_power_of_2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _prune_fd_configs(configs, nargs, **kwargs):
    args = {**nargs, **kwargs}
    max_block_h = align_up(args["H_Q"], 16)
    d_v = args["D_V"]
    pruned = []
    for c in configs:
        block_h = c.kwargs["BLOCK_H"]
        group_size = c.kwargs["GROUP_SIZE"]
        block_d = c.kwargs["BLOCK_D"]
        block_v = c.kwargs["BLOCK_V"]
        if block_h > max_block_h:
            continue
        # tl.arange(0, N) requires N to be a power of 2.
        if not all(_is_power_of_2(v) for v in (block_h, group_size, block_d, block_v)):
            continue
        # tl.dot on fp16/bf16 requires K >= 16; GROUP_SIZE is K for probs@v, BLOCK_D is K for q@k.
        if group_size < 16 or block_d < 16:
            continue
        if group_size % 8 != 0:
            continue
        if block_d > d_v or d_v % block_d != 0:
            continue
        if block_v > d_v or d_v % block_v != 0:
            continue
        pruned.append(c)
    return pruned if pruned else configs

_RD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_V": 32}, num_warps=1),
    triton.Config({"BLOCK_V": 64}, num_warps=1),
    triton.Config({"BLOCK_V": 128}, num_warps=1),
    triton.Config({"BLOCK_V": 256}, num_warps=1),
    triton.Config({"BLOCK_V": 256}, num_warps=2),
    triton.Config({"BLOCK_V": 256}, num_warps=4),
    triton.Config({"BLOCK_V": 512}, num_warps=1),
    triton.Config({"BLOCK_V": 512}, num_warps=2),
    triton.Config({"BLOCK_V": 512}, num_warps=4),
]

_RD_KERNEL_CONFIGS = _RD_AUTOTUNE_CONFIGS if _TUNE_RD else [_DEFAULT_RD_KERNEL_CONFIG]


def _best_fd_param(key: str, default):
    best_config = getattr(flash_mla_sparse_with_kv_cache_fd_kernel, "best_config", None)
    if best_config is None:
        return _DEFAULT_FD_CONFIG.get(key, default)
    if key == "num_warps":
        return best_config.num_warps
    return best_config.kwargs.get(key, default)


def _best_v2_gluon_param(key: str, default):
    best_config = getattr(flash_mla_sparse_with_kv_cache_v2_gluon_kernel, "best_config", None)
    if best_config is None:
        return _DEFAULT_V2_GLUON_KERNEL_CONFIG.kwargs.get(key, default)
    if key == "num_warps":
        return best_config.num_warps
    return best_config.kwargs.get(key, default)


def _format_kernel_config(name: str, config: triton.Config) -> str:
    kwargs = ", ".join(f"{k}={v}" for k, v in config.kwargs.items() if k != "CTA_WARPS")
    line = f"  {name}: {{{kwargs}}}, num_warps={config.num_warps}"
    if name in ("fd_kernel", "fa_kernel", "v2_gluon_kernel"):
        line += f", num_stages={config.num_stages}"
    return line


def _print_autotune_configs(*, include_fa: bool = False, include_v2: bool = False) -> None:
    entries = [
        ("fd_kernel", flash_mla_sparse_with_kv_cache_fd_kernel, _DEFAULT_FD_KERNEL_CONFIG, _TUNE_FD),
        ("rd_kernel", flash_mla_sparse_with_kv_cache_rd_kernel, _DEFAULT_RD_KERNEL_CONFIG, _TUNE_RD),
    ]
    if include_fa:
        entries.append(
            ("fa_kernel", flash_mla_sparse_with_kv_cache_fa_kernel, _DEFAULT_FA_KERNEL_CONFIG, _TUNE_FA),
        )
    if include_v2:
        entries.append(
            (
                "v2_gluon_kernel",
                flash_mla_sparse_with_kv_cache_v2_gluon_kernel,
                _DEFAULT_V2_GLUON_KERNEL_CONFIG,
                _TUNE_V2_GLUON,
            ),
        )
    for name, kernel, default, use_best in entries:
        if use_best:
            config = getattr(kernel, "best_config", None)
            if config is None:
                print(f"  {name}: (autotune not run)")
                continue
            print(_format_kernel_config(name, config))
        else:
            print(_format_kernel_config(name, default))


@triton.jit
def _load_model1_dense_kv_for_dot(
    kv_ptr,
    block_idx,
    pos_in_block,
    offsets,
    mask,
    block_stride: tl.constexpr,
    pos_stride: tl.constexpr):
    return tl.load(
        kv_ptr + block_idx * block_stride + pos_in_block * pos_stride + offsets,
        mask=mask,
        other=0.0)


@triton.autotune(
    configs=_FD_KERNEL_CONFIGS,
    key=["H_Q", "D_V", "TOPK", "EXTRA_TOPK"],
    prune_configs_by={"early_config_prune": _prune_fd_configs} if ENABLE_AUTOTUNE else None,
)
@triton.jit
def flash_mla_sparse_with_kv_cache_fd_kernel(
    q_ptr,
    k_cache_ptr,
    indices_ptr,
    topk_length_ptr,
    extra_k_cache_ptr,
    extra_indices_ptr,
    extra_topk_length_ptr,
    middle_ptr,
    group_maxs_ptr,
    group_expsums_ptr,
    sm_scale,
    TOPK: gl.constexpr,
    EXTRA_TOPK: gl.constexpr,
    PAGE_BLOCK_SIZE: gl.constexpr,
    EXTRA_PAGE_BLOCK_SIZE: gl.constexpr,
    HAS_EXTRA_K_CACHE: gl.constexpr,
    H_Q: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr):
    head_block_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    group_idx = tl.program_id(2)
    batch_idx = query_idx

    num_queries = tl.num_programs(1)
    max_groups = tl.num_programs(2)
    stride_q_b = H_Q * D_V
    stride_q_h = D_V
    stride_k_block = PAGE_BLOCK_SIZE * D_V
    stride_extra_k_block = EXTRA_PAGE_BLOCK_SIZE * D_V
    stride_indices_b = TOPK
    stride_extra_indices_b = EXTRA_TOPK
    stride_middle_g = num_queries * H_Q * D_V
    stride_middle_q = H_Q * D_V
    stride_middle_h = D_V
    stride_group_q = H_Q * max_groups
    stride_group_h = max_groups

    head_offsets = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    group_offsets = tl.arange(0, GROUP_SIZE)
    d_offsets = tl.arange(0, BLOCK_D)
    v_offsets = tl.arange(0, BLOCK_V)

    context_len = tl.load(topk_length_ptr + batch_idx)
    extra_context_len = 0
    if HAS_EXTRA_K_CACHE:
        extra_context_len = tl.load(extra_topk_length_ptr + batch_idx)

    swa_groups = (context_len + GROUP_SIZE - 1) // GROUP_SIZE
    extra_groups = 0
    if HAS_EXTRA_K_CACHE:
        extra_groups = (extra_context_len + GROUP_SIZE - 1) // GROUP_SIZE

    if group_idx >= swa_groups + extra_groups:
        return

    is_extra_group = group_idx >= swa_groups
    if is_extra_group:
        slot_offsets = (group_idx - swa_groups) * GROUP_SIZE + group_offsets
        idx_vals = tl.load(
            extra_indices_ptr + batch_idx * stride_extra_indices_b + slot_offsets,
            mask=slot_offsets < extra_context_len,
            other=-1)
        valid = (slot_offsets < extra_context_len) & (idx_vals >= 0)
        safe_idx = tl.where(valid, idx_vals, 0)
        kv_block_idx = safe_idx // EXTRA_PAGE_BLOCK_SIZE
        kv_pos = safe_idx - kv_block_idx * EXTRA_PAGE_BLOCK_SIZE
        kv_stride = stride_extra_k_block
    else:
        slot_offsets = group_idx * GROUP_SIZE + group_offsets
        idx_vals = tl.load(
            indices_ptr + batch_idx * stride_indices_b + slot_offsets,
            mask=slot_offsets < context_len,
            other=-1)
        valid = (slot_offsets < context_len) & (idx_vals >= 0)
        safe_idx = tl.where(valid, idx_vals, 0)
        kv_block_idx = safe_idx // PAGE_BLOCK_SIZE
        kv_pos = safe_idx - kv_block_idx * PAGE_BLOCK_SIZE
        kv_stride = stride_k_block

    scores = tl.zeros((BLOCK_H, GROUP_SIZE), dtype=tl.float32)
    if is_extra_group:
        for d_start in tl.range(0, D_V, BLOCK_D):
            ds = d_start + d_offsets
            q_vals = tl.load(
                q_ptr
                + batch_idx * stride_q_b
                + head_offsets[:, None] * stride_q_h
                + ds[None, :],
                mask=(head_offsets[:, None] < H_Q) & (ds[None, :] < D_V),
                other=0.0)
            k_vals = _load_model1_dense_kv_for_dot(
                extra_k_cache_ptr,
                kv_block_idx[None, :],
                kv_pos[None, :],
                ds[:, None],
                valid[None, :] & (ds[:, None] < D_V),
                kv_stride,
                D_V)
            scores += tl.dot(q_vals, k_vals, out_dtype=tl.float32) * sm_scale
    else:
        for d_start in tl.range(0, D_V, BLOCK_D):
            ds = d_start + d_offsets
            q_vals = tl.load(
                q_ptr
                + batch_idx * stride_q_b
                + head_offsets[:, None] * stride_q_h
                + ds[None, :],
                mask=(head_offsets[:, None] < H_Q) & (ds[None, :] < D_V),
                other=0.0)
            k_vals = _load_model1_dense_kv_for_dot(
                k_cache_ptr,
                kv_block_idx[None, :],
                kv_pos[None, :],
                ds[:, None],
                valid[None, :] & (ds[:, None] < D_V),
                kv_stride,
                D_V)
            scores += tl.dot(q_vals, k_vals, out_dtype=tl.float32) * sm_scale

    scores = tl.where(valid[None, :] & (head_offsets[:, None] < H_Q), scores, -float("inf"))
    group_max = tl.max(scores, axis=1)
    group_has_valid = tl.sum(tl.where(valid, 1, 0), axis=0) > 0
    safe_group_max = tl.where(group_has_valid, group_max, 0.0)
    exp_scores = tl.where(valid[None, :], tl.exp(scores - safe_group_max[:, None]), 0.0)
    exp_sum = tl.sum(exp_scores, axis=1)
    inv_exp_sum = tl.where(exp_sum > 0.0, 1.0 / exp_sum, 0.0)
    probs = exp_scores * inv_exp_sum[:, None]
    if q_ptr.dtype.element_ty == tl.bfloat16:
        probs_for_dot = probs.to(tl.bfloat16)
    else:
        probs_for_dot = probs.to(tl.float16)

    group_offsets_out = query_idx * stride_group_q + head_offsets * stride_group_h + group_idx
    tl.store(group_maxs_ptr + group_offsets_out, group_max, mask=head_offsets < H_Q)
    tl.store(group_expsums_ptr + group_offsets_out, exp_sum, mask=head_offsets < H_Q)

    if is_extra_group:
        for v_start in tl.range(0, D_V, BLOCK_V):
            vs = v_start + v_offsets
            v_vals = _load_model1_dense_kv_for_dot(
                extra_k_cache_ptr,
                kv_block_idx[:, None],
                kv_pos[:, None],
                vs[None, :],
                valid[:, None] & (vs[None, :] < D_V),
                kv_stride,
                D_V)
            partial = tl.dot(probs_for_dot, v_vals, out_dtype=tl.float32)
            tl.store(
                middle_ptr
                + group_idx * stride_middle_g
                + query_idx * stride_middle_q
                + head_offsets[:, None] * stride_middle_h
                + vs[None, :],
                partial,
                mask=(head_offsets[:, None] < H_Q) & (vs[None, :] < D_V))
    else:
        for v_start in tl.range(0, D_V, BLOCK_V):
            vs = v_start + v_offsets
            v_vals = _load_model1_dense_kv_for_dot(
                k_cache_ptr,
                kv_block_idx[:, None],
                kv_pos[:, None],
                vs[None, :],
                valid[:, None] & (vs[None, :] < D_V),
                kv_stride,
                D_V)
            partial = tl.dot(probs_for_dot, v_vals, out_dtype=tl.float32)
            tl.store(
                middle_ptr
                + group_idx * stride_middle_g
                + query_idx * stride_middle_q
                + head_offsets[:, None] * stride_middle_h
                + vs[None, :],
                partial,
                mask=(head_offsets[:, None] < H_Q) & (vs[None, :] < D_V))


@triton.autotune(
    configs=_RD_KERNEL_CONFIGS,
    key=["H_Q", "D_V", "CEIL_GROUPS", "HAS_EXTRA_K_CACHE"],
)
@triton.jit
def flash_mla_sparse_with_kv_cache_rd_kernel(
    middle_ptr,
    group_maxs_ptr,
    group_expsums_ptr,
    attn_sink_ptr,
    out_ptr,
    topk_length_ptr,
    extra_topk_length_ptr,
    H_Q: tl.constexpr,
    D_V: tl.constexpr,
    CEIL_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_EXTRA_K_CACHE: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_V: tl.constexpr):
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    v_block_idx = tl.program_id(2)
    batch_idx = query_idx

    num_queries = tl.num_programs(0)
    stride_middle_g = num_queries * H_Q * D_V
    stride_middle_q = H_Q * D_V
    stride_middle_h = D_V
    stride_group_q = H_Q * CEIL_GROUPS
    stride_group_h = CEIL_GROUPS
    stride_out_b = H_Q * D_V
    stride_out_h = D_V

    v_offsets = v_block_idx * BLOCK_V + tl.arange(0, BLOCK_V)

    context_len = tl.load(topk_length_ptr + batch_idx)
    extra_context_len = 0
    if HAS_EXTRA_K_CACHE:
        extra_context_len = tl.load(extra_topk_length_ptr + batch_idx)
    swa_groups = (context_len + GROUP_SIZE - 1) // GROUP_SIZE
    extra_groups = 0
    if HAS_EXTRA_K_CACHE:
        extra_groups = (extra_context_len + GROUP_SIZE - 1) // GROUP_SIZE
    effective_groups = swa_groups + extra_groups

    final_max = tl.full((), -float("inf"), dtype=tl.float32)
    for group_idx in tl.range(0, effective_groups):
        group_max = tl.load(
            group_maxs_ptr + query_idx * stride_group_q + head_idx * stride_group_h + group_idx,
            mask=head_idx < H_Q,
            other=-float("inf"))  # scalar
        final_max = tl.maximum(final_max, group_max)

    denom = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for group_idx in tl.range(0, effective_groups):
        group_max = tl.load(
            group_maxs_ptr + query_idx * stride_group_q + head_idx * stride_group_h + group_idx,
            mask=head_idx < H_Q,
            other=-float("inf"))  # scalar
        group_exp_sum = tl.load(
            group_expsums_ptr
            + query_idx * stride_group_q
            + head_idx * stride_group_h
            + group_idx,
            mask=head_idx < H_Q,
            other=0.0)  # scalar
        weight = tl.exp(group_max - final_max) * group_exp_sum
        partial = tl.load(
            middle_ptr
            + group_idx * stride_middle_g
            + query_idx * stride_middle_q
            + head_idx * stride_middle_h
            + v_offsets,
            mask=(head_idx < H_Q) & (v_offsets < D_V),
            other=0.0)  # block_shape: (BLOCK_V)
        acc += partial * weight
        denom += weight

    has_valid = denom > 0.0
    final_denom = denom
    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + head_idx, mask=head_idx < H_Q, other=0.0).to(tl.float32)
        sink_weight = tl.exp(sink - final_max)
        final_denom += tl.where(has_valid, sink_weight, 0.0)

    out_vals = tl.where(has_valid, acc / final_denom, 0.0)
    tl.store(
        out_ptr
        + batch_idx * stride_out_b
        + head_idx * stride_out_h
        + v_offsets,
        out_vals,
        mask=(head_idx < H_Q) & (v_offsets < D_V))


def _best_fa_param(key: str, default):
    best_config = getattr(flash_mla_sparse_with_kv_cache_fa_kernel, "best_config", None)
    if best_config is None:
        return _DEFAULT_FA_KERNEL_CONFIG.kwargs.get(key, _DEFAULT_FD_CONFIG.get(key, default))
    if key == "num_warps":
        return best_config.num_warps
    return best_config.kwargs.get(key, default)


@triton.autotune(
    configs=_FA_KERNEL_CONFIGS,
    key=["H_Q", "D_V", "TOPK", "EXTRA_TOPK"],
    prune_configs_by={"early_config_prune": _prune_fd_configs},
)
@triton.jit
def flash_mla_sparse_with_kv_cache_fa_kernel(
    q_ptr,
    k_cache_ptr,
    indices_ptr,
    topk_length_ptr,
    extra_k_cache_ptr,
    extra_indices_ptr,
    extra_topk_length_ptr,
    attn_sink_ptr,
    out_ptr,
    sm_scale,
    TOPK: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
    EXTRA_PAGE_BLOCK_SIZE: tl.constexpr,
    HAS_EXTRA_K_CACHE: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    H_Q: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr
):
    """End-to-end sparse MLA: QK^T, online softmax across groups, PV, attn_sink — no rd_kernel."""
    head_block_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    batch_idx = query_idx

    stride_q_b = H_Q * D_V
    stride_q_h = D_V
    stride_k_block = PAGE_BLOCK_SIZE * D_V
    stride_extra_k_block = EXTRA_PAGE_BLOCK_SIZE * D_V
    stride_indices_b = TOPK
    stride_extra_indices_b = EXTRA_TOPK
    stride_out_b = H_Q * D_V
    stride_out_h = D_V

    head_offsets = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    group_offsets = tl.arange(0, GROUP_SIZE)
    d_offsets = tl.arange(0, BLOCK_D)
    v_offsets = tl.arange(0, D_V)

    context_len = tl.load(topk_length_ptr + batch_idx)
    extra_context_len = 0
    if HAS_EXTRA_K_CACHE:
        extra_context_len = tl.load(extra_topk_length_ptr + batch_idx)
    swa_groups = (context_len + GROUP_SIZE - 1) // GROUP_SIZE
    extra_groups = 0
    if HAS_EXTRA_K_CACHE:
        extra_groups = (extra_context_len + GROUP_SIZE - 1) // GROUP_SIZE
    effective_groups = swa_groups + extra_groups

    running_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    running_denom = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D_V), dtype=tl.float32)

    for group_idx in tl.range(0, effective_groups):
        is_extra_group = group_idx >= swa_groups
        if is_extra_group:
            slot_offsets = (group_idx - swa_groups) * GROUP_SIZE + group_offsets
            idx_vals = tl.load(
                extra_indices_ptr + batch_idx * stride_extra_indices_b + slot_offsets,
                mask=slot_offsets < extra_context_len,
                other=-1)
            valid = (slot_offsets < extra_context_len) & (idx_vals >= 0)
            safe_idx = tl.where(valid, idx_vals, 0)
            kv_block_idx = safe_idx // EXTRA_PAGE_BLOCK_SIZE
            kv_pos = safe_idx - kv_block_idx * EXTRA_PAGE_BLOCK_SIZE
            kv_ptr = extra_k_cache_ptr
            kv_stride = stride_extra_k_block
        else:
            slot_offsets = group_idx * GROUP_SIZE + group_offsets
            idx_vals = tl.load(
                indices_ptr + batch_idx * stride_indices_b + slot_offsets,
                mask=slot_offsets < context_len,
                other=-1)
            valid = (slot_offsets < context_len) & (idx_vals >= 0)
            safe_idx = tl.where(valid, idx_vals, 0)
            kv_block_idx = safe_idx // PAGE_BLOCK_SIZE
            kv_pos = safe_idx - kv_block_idx * PAGE_BLOCK_SIZE
            kv_ptr = k_cache_ptr
            kv_stride = stride_k_block

        scores = tl.zeros((BLOCK_H, GROUP_SIZE), dtype=tl.float32)
        for d_start in tl.range(0, D_V, BLOCK_D):
            ds = d_start + d_offsets
            q_vals = tl.load(
                q_ptr
                + batch_idx * stride_q_b
                + head_offsets[:, None] * stride_q_h
                + ds[None, :],
                mask=(head_offsets[:, None] < H_Q) & (ds[None, :] < D_V),
                other=0.0)
            k_vals = _load_model1_dense_kv_for_dot(
                kv_ptr,
                kv_block_idx[None, :],
                kv_pos[None, :],
                ds[:, None],
                valid[None, :] & (ds[:, None] < D_V),
                kv_stride,
                D_V)
            scores += tl.dot(q_vals, k_vals, out_dtype=tl.float32) * sm_scale

        scores = tl.where(valid[None, :] & (head_offsets[:, None] < H_Q), scores, -float("inf"))
        group_max = tl.max(scores, axis=1)
        exp_scores = tl.where(valid[None, :], tl.exp(scores - group_max[:, None]), 0.0)
        exp_sum = tl.sum(exp_scores, axis=1)
        head_active = exp_sum > 0.0

        if q_ptr.dtype.element_ty == tl.bfloat16:
            exp_for_dot = exp_scores.to(tl.bfloat16)
        else:
            exp_for_dot = exp_scores.to(tl.float16)

        v_vals = _load_model1_dense_kv_for_dot(
            kv_ptr,
            kv_block_idx[:, None],
            kv_pos[:, None],
            v_offsets[None, :],
            valid[:, None] & (v_offsets[None, :] < D_V),
            kv_stride,
            D_V)
        partial_unnorm = tl.dot(exp_for_dot, v_vals, out_dtype=tl.float32)

        new_max = tl.where(head_active, tl.maximum(running_max, group_max), running_max)
        scale_old = tl.exp(running_max - new_max)
        scale_group = tl.where(head_active, tl.exp(group_max - new_max), 0.0)
        acc = acc * scale_old[:, None] + partial_unnorm * scale_group[:, None]
        running_denom = running_denom * scale_old + exp_sum * scale_group
        running_max = new_max

    has_valid = running_denom > 0.0
    final_denom = running_denom
    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + head_offsets, mask=head_offsets < H_Q, other=0.0).to(tl.float32)
        sink_weight = tl.exp(sink - running_max)
        final_denom = running_denom + tl.where(has_valid, sink_weight, 0.0)

    out_vals = tl.where(has_valid[:, None], acc / final_denom[:, None], 0.0)
    tl.store(
        out_ptr
        + batch_idx * stride_out_b
        + head_offsets[:, None] * stride_out_h
        + v_offsets[None, :],
        out_vals,
        mask=(head_offsets[:, None] < H_Q) & (v_offsets[None, :] < D_V))


@triton.autotune(
    configs=_V2_GLUON_KERNEL_CONFIGS,
    key=["H_Q", "NUM_KV_HEADS", "D_V", "TOPK", "EXTRA_TOPK"],
    prune_configs_by={"early_config_prune": _prune_fd_configs} if ENABLE_AUTOTUNE else None,
)
@gluon.jit
def flash_mla_sparse_with_kv_cache_v2_gluon_kernel(
    q_ptr,
    k_cache_ptr,
    indices_ptr,
    topk_length_ptr,
    extra_k_cache_ptr,
    extra_indices_ptr,
    extra_topk_length_ptr,
    middle_ptr,
    group_maxs_ptr,
    group_expsums_ptr,
    sm_scale,
    TOPK: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
    EXTRA_PAGE_BLOCK_SIZE: tl.constexpr,
    HAS_EXTRA_K_CACHE: tl.constexpr,
    H_Q: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    D_V: gl.constexpr,
    BLOCK_H: gl.constexpr,
    GROUP_SIZE: gl.constexpr,
    BLOCK_D: gl.constexpr,
    BLOCK_V: gl.constexpr,
    CTA_WARPS: gl.constexpr
):
    # grid: [num_seqs, num_groups, num_kv_head_groups].
    # Each CTA computes one sparse group for a block of query heads that share a KV head.
    q_layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [CTA_WARPS, 1], [1, 0])
    k_layout: gl.constexpr = gl.BlockedLayout([8, 1], [16, 2], [1, CTA_WARPS], [0, 1])
    v_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, CTA_WARPS], [1, 0])
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, CTA_WARPS], instr_shape=[16, 8])
    q_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
    k_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[0, 1])
    v_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
    f32_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
    elem_ty: gl.constexpr = q_ptr.dtype.element_ty
    k_width: gl.constexpr = 32 // elem_ty.primitive_bitwidth
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=k_width)
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=k_width)

    batch_idx = gl.program_id(0)
    group_idx = gl.program_id(1)
    kv_group_program_idx = gl.program_id(2)

    actual_group_heads: gl.constexpr = H_Q // NUM_KV_HEADS
    q_blocks_per_kv: gl.constexpr = (actual_group_heads + BLOCK_H - 1) // BLOCK_H
    kv_head_idx = kv_group_program_idx // q_blocks_per_kv
    q_block_idx = kv_group_program_idx - kv_head_idx * q_blocks_per_kv
    q_head_base = kv_head_idx * actual_group_heads + q_block_idx * BLOCK_H

    num_queries = gl.num_programs(0)
    max_groups = gl.num_programs(1)
    stride_q_b: gl.constexpr = H_Q * D_V
    stride_q_h: gl.constexpr = D_V
    stride_k_block: gl.constexpr = PAGE_BLOCK_SIZE * NUM_KV_HEADS * D_V
    stride_k_pos: gl.constexpr = NUM_KV_HEADS * D_V
    stride_extra_k_block: gl.constexpr = EXTRA_PAGE_BLOCK_SIZE * NUM_KV_HEADS * D_V
    stride_extra_k_pos: gl.constexpr = NUM_KV_HEADS * D_V
    stride_indices_b: gl.constexpr = NUM_KV_HEADS * TOPK
    stride_extra_indices_b: gl.constexpr = NUM_KV_HEADS * EXTRA_TOPK
    stride_middle_g = num_queries * H_Q * D_V
    stride_middle_q: gl.constexpr = H_Q * D_V
    stride_middle_h: gl.constexpr = D_V
    stride_group_q = H_Q * max_groups
    stride_group_h = max_groups

    head_offsets = q_head_base + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, q_layout))
    head_offsets_v = q_head_base + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, v_layout))
    head_offsets_mma = q_head_base + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mma_layout))
    group_offsets = gl.arange(0, GROUP_SIZE, layout=gl.SliceLayout(0, k_layout))
    group_offsets_v = gl.arange(0, GROUP_SIZE, layout=gl.SliceLayout(1, v_layout))
    d_offsets = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, q_layout))
    d_offsets_k = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(1, k_layout))
    v_offsets = gl.arange(0, BLOCK_V, layout=gl.SliceLayout(0, v_layout))

    q_smem = gl.allocate_shared_memory(elem_ty, [2, BLOCK_H, BLOCK_D], layout=q_smem_layout)
    k_smem = gl.allocate_shared_memory(elem_ty, [2, BLOCK_D, GROUP_SIZE], layout=k_smem_layout)
    v_smem = gl.allocate_shared_memory(elem_ty, [GROUP_SIZE, BLOCK_V], layout=v_smem_layout)
    scores_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_H, GROUP_SIZE], layout=f32_smem_layout)
    exp_smem = gl.allocate_shared_memory(elem_ty, [BLOCK_H, GROUP_SIZE], layout=q_smem_layout)

    context_len = gl.load(topk_length_ptr + batch_idx)
    extra_context_len = 0
    if HAS_EXTRA_K_CACHE:
        extra_context_len = gl.load(extra_topk_length_ptr + batch_idx)
    swa_groups = (context_len + GROUP_SIZE - 1) // GROUP_SIZE
    extra_groups = 0
    if HAS_EXTRA_K_CACHE:
        extra_groups = (extra_context_len + GROUP_SIZE - 1) // GROUP_SIZE
    effective_groups = swa_groups + extra_groups
    is_active_group = group_idx < effective_groups
    is_extra_group = group_idx >= swa_groups

    slot_offsets = group_idx * GROUP_SIZE + group_offsets
    valid_len = context_len
    indices_base = indices_ptr + batch_idx * stride_indices_b + kv_head_idx * TOPK
    kv_ptr = k_cache_ptr
    page_size = PAGE_BLOCK_SIZE
    kv_block_stride = stride_k_block
    kv_pos_stride = stride_k_pos
    if HAS_EXTRA_K_CACHE:
        if is_extra_group:
            slot_offsets = (group_idx - swa_groups) * GROUP_SIZE + group_offsets
            valid_len = extra_context_len
            indices_base = extra_indices_ptr + batch_idx * stride_extra_indices_b + kv_head_idx * EXTRA_TOPK
            kv_ptr = extra_k_cache_ptr
            page_size = EXTRA_PAGE_BLOCK_SIZE
            kv_block_stride = stride_extra_k_block
            kv_pos_stride = stride_extra_k_pos

    idx_vals = gl.load(indices_base + slot_offsets, mask=is_active_group & (slot_offsets < valid_len), other=-1)
    valid = is_active_group & (slot_offsets < valid_len) & (idx_vals >= 0)
    valid_mma = gl.convert_layout(valid, gl.SliceLayout(0, mma_layout))
    valid_v = gl.convert_layout(valid, gl.SliceLayout(1, v_layout))
    safe_idx = gl.where(valid, idx_vals, 0)
    kv_block_idx = safe_idx // page_size
    kv_pos = safe_idx - kv_block_idx * page_size

    scores_smem.store(gl.zeros([BLOCK_H, GROUP_SIZE], gl.float32, layout=mma_layout))
    q_base = q_ptr + batch_idx * stride_q_b
    q_head_limit = kv_head_idx * actual_group_heads + actual_group_heads
    head_mask_q = (head_offsets[:, None] < H_Q) & (head_offsets[:, None] < q_head_limit)
    head_mask_mma = (head_offsets_mma[:, None] < H_Q) & (head_offsets_mma[:, None] < q_head_limit)
    head_mask_v = (head_offsets_v[:, None] < H_Q) & (head_offsets_v[:, None] < q_head_limit)

    qk_copy_idx = 0
    qk_read_idx = 0
    ds = qk_copy_idx * BLOCK_D + d_offsets
    ds_k = qk_copy_idx * BLOCK_D + d_offsets_k
    q_mask = head_mask_q & (ds[None, :] < D_V)
    q_ptrs = q_base + head_offsets[:, None] * stride_q_h + ds[None, :]
    cp.async_copy_global_to_shared(q_smem.index(qk_copy_idx % 2), q_ptrs, mask=q_mask)
    k_mask = valid[None, :] & (ds_k[:, None] < D_V)
    k_ptrs = kv_ptr + kv_block_idx[None, :] * kv_block_stride + kv_pos[None, :] * kv_pos_stride + kv_head_idx * D_V + ds_k[:, None]
    cp.async_copy_global_to_shared(k_smem.index(qk_copy_idx % 2), k_ptrs, mask=k_mask)
    cp.commit_group()
    qk_copy_idx += 1
    for _ in range(1, D_V // BLOCK_D):
        ds = qk_copy_idx * BLOCK_D + d_offsets
        ds_k = qk_copy_idx * BLOCK_D + d_offsets_k
        q_mask = head_mask_q & (ds[None, :] < D_V)
        q_ptrs = q_base + head_offsets[:, None] * stride_q_h + ds[None, :]
        cp.async_copy_global_to_shared(q_smem.index(qk_copy_idx % 2), q_ptrs, mask=q_mask)
        k_mask = valid[None, :] & (ds_k[:, None] < D_V)
        k_ptrs = kv_ptr + kv_block_idx[None, :] * kv_block_stride + kv_pos[None, :] * kv_pos_stride + kv_head_idx * D_V + ds_k[:, None]
        cp.async_copy_global_to_shared(k_smem.index(qk_copy_idx % 2), k_ptrs, mask=k_mask)
        cp.commit_group()
        cp.wait_group(1)
        q_vals = q_smem.index(qk_read_idx % 2).load(q_dot_layout)
        k_vals = k_smem.index(qk_read_idx % 2).load(k_dot_layout)
        scores = scores_smem.load(mma_layout)
        partial = mma_v2(q_vals, k_vals, gl.zeros([BLOCK_H, GROUP_SIZE], gl.float32, layout=mma_layout), input_precision="tf32")
        scores_smem.store(scores + partial * sm_scale)
        qk_copy_idx += 1
        qk_read_idx += 1
    cp.wait_group(0)
    q_vals = q_smem.index(qk_read_idx % 2).load(q_dot_layout)
    k_vals = k_smem.index(qk_read_idx % 2).load(k_dot_layout)
    scores = scores_smem.load(mma_layout)
    partial = mma_v2(q_vals, k_vals, gl.zeros([BLOCK_H, GROUP_SIZE], gl.float32, layout=mma_layout), input_precision="tf32")
    scores = scores + partial * sm_scale
    scores = gl.where(valid_mma[None, :] & head_mask_mma, scores, -float("inf"))
    scores_smem.store(scores)

    scores = scores_smem.load(mma_layout)
    group_max = gl.max(scores, axis=1)
    exp_scores = gl.where(valid_mma[None, :], gl.exp(scores - group_max[:, None]), 0.0)
    exp_sum = gl.sum(exp_scores, axis=1)
    safe_group_max = gl.where(exp_sum > 0.0, group_max, -float("inf"))
    inv_exp_sum = gl.where(exp_sum > 0.0, 1.0 / exp_sum, 0.0)
    probs = exp_scores * inv_exp_sum[:, None]
    if elem_ty == gl.bfloat16:
        probs_for_dot = probs.to(gl.bfloat16)
    else:
        probs_for_dot = probs.to(gl.float16)
    exp_smem.store(gl.convert_layout(probs_for_dot, q_layout))
    probs_dot = exp_smem.load(q_dot_layout)

    group_offsets_out = batch_idx * stride_group_q + head_offsets_mma * stride_group_h + group_idx
    gl.store(group_maxs_ptr + group_offsets_out, safe_group_max, mask=is_active_group & (head_offsets_mma < H_Q) & (head_offsets_mma < q_head_limit))
    gl.store(group_expsums_ptr + group_offsets_out, exp_sum, mask=is_active_group & (head_offsets_mma < H_Q) & (head_offsets_mma < q_head_limit))

    for v_start in gl.static_range(0, D_V, BLOCK_V):
        vs = v_start + v_offsets
        kv_block_idx_v = gl.convert_layout(kv_block_idx, gl.SliceLayout(1, v_layout))
        kv_pos_v = gl.convert_layout(kv_pos, gl.SliceLayout(1, v_layout))
        v_mask = valid_v[:, None] & (vs[None, :] < D_V)
        v_ptrs = kv_ptr + kv_block_idx_v[:, None] * kv_block_stride + kv_pos_v[:, None] * kv_pos_stride + kv_head_idx * D_V + vs[None, :]
        cp.async_copy_global_to_shared(v_smem, v_ptrs, mask=v_mask)
        cp.commit_group()
        cp.wait_group(0)
        v_vals = v_smem.load(k_dot_layout)
        partial = mma_v2(probs_dot, v_vals, gl.zeros([BLOCK_H, BLOCK_V], gl.float32, layout=mma_layout), input_precision="tf32")
        partial = gl.convert_layout(partial, v_layout)
        gl.store(
            middle_ptr
            + group_idx * stride_middle_g
            + batch_idx * stride_middle_q
            + head_offsets_v[:, None] * stride_middle_h
            + vs[None, :],
            partial,
            mask=is_active_group & head_mask_v & (vs[None, :] < D_V))


def _sparse_kvcache_run_config(num_tokens: int, num_heads: int, topk: int, extra_topk: int, extra_page_size: int) -> dict:
    return dict(_DEFAULT_FD_CONFIG)


def flash_mla_sparse_with_kv_cache_v2_gluon(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor,
    d_v: int = 512,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gluon split implementation with grid=(num_seqs, num_groups, kv/q-head groups)."""
    assert q.is_contiguous() and q.dtype in (torch.float16, torch.bfloat16)
    assert q.dim() == 4 and q.size(1) == 1 and q.size(3) == d_v, (
        f"q must be contiguous fp16/bf16 [num_tokens, 1, num_heads, head_dim], got {tuple(q.shape)}"
    )
    num_tokens = q.size(0)
    num_heads = q.size(2)
    head_dim = q.size(3)
    assert d_v == head_dim, "d_v must equal q head_dim"
    assert kv_cache.is_contiguous() and kv_cache.dtype == q.dtype and kv_cache.dim() == 4
    assert kv_cache.size(3) == head_dim, (
        f"kv_cache must be [num_kv, page_size, num_kv_heads, head_dim], got {tuple(kv_cache.shape)}"
    )
    num_kv_heads = kv_cache.size(2)
    page_size = kv_cache.size(1)
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    assert indices.is_contiguous() and indices.dtype is torch.int32 and indices.dim() == 3
    assert indices.size(0) == num_tokens and indices.size(1) == num_kv_heads, (
        f"indices must be [num_tokens, num_kv_heads, topk], got {tuple(indices.shape)}"
    )
    topk = indices.size(2)
    assert topk_length.is_contiguous() and topk_length.dtype is torch.int32
    assert topk_length.dim() == 1 and topk_length.size(0) == num_tokens, (
        f"topk_length must be [num_tokens], got {tuple(topk_length.shape)}"
    )
    assert attn_sink.is_contiguous() and attn_sink.dtype is torch.float32
    assert attn_sink.dim() == 1 and attn_sink.size(0) == num_heads, (
        f"attn_sink must be [num_heads], got {tuple(attn_sink.shape)}"
    )

    has_extra = extra_kv_cache is not None
    assert has_extra == (extra_indices is not None) and has_extra == (extra_topk_length is not None), (
        "extra_kv_cache, extra_indices, and extra_topk_length must be provided together"
    )
    if has_extra:
        assert extra_kv_cache.is_contiguous() and extra_kv_cache.dtype == q.dtype
        assert extra_kv_cache.dim() == 4 and extra_kv_cache.size(2) == num_kv_heads
        assert extra_kv_cache.size(3) == head_dim
        assert extra_indices.is_contiguous() and extra_indices.dtype is torch.int32
        assert extra_indices.dim() == 3 and extra_indices.size(0) == num_tokens and extra_indices.size(1) == num_kv_heads
        assert extra_topk_length.is_contiguous() and extra_topk_length.dtype is torch.int32
        assert extra_topk_length.dim() == 1 and extra_topk_length.size(0) == num_tokens
        extra_topk = extra_indices.size(2)
        extra_page_size = extra_kv_cache.size(1)
    else:
        extra_topk = 0
        extra_page_size = 1

    ensure_fp16_bf16_contiguous(q)
    ensure_fp16_bf16_contiguous(kv_cache)
    ensure_int32_contiguous(indices)
    ensure_int32_contiguous(topk_length)
    ensure_fp32_contiguous(attn_sink)
    if has_extra:
        ensure_fp16_bf16_contiguous(extra_kv_cache)
        ensure_int32_contiguous(extra_indices)
        ensure_int32_contiguous(extra_topk_length)

    if out is None:
        out = torch.empty((num_tokens, 1, num_heads, d_v), dtype=q.dtype, device=q.device)
    else:
        assert out.is_contiguous() and out.dtype == q.dtype
        assert out.shape == (num_tokens, 1, num_heads, d_v), f"out shape mismatch: {tuple(out.shape)}"
        ensure_fp16_bf16_contiguous(out)

    min_group_size = min(config.kwargs["GROUP_SIZE"] for config in _V2_GLUON_KERNEL_CONFIGS)
    max_groups = triton.cdiv(topk + extra_topk, min_group_size)
    middle = torch.empty((max_groups, num_tokens, num_heads, head_dim), dtype=torch.float32, device=q.device)
    group_maxs = torch.empty((num_tokens, num_heads, max_groups), dtype=torch.float32, device=q.device)
    group_expsums = torch.empty((num_tokens, num_heads, max_groups), dtype=torch.float32, device=q.device)

    actual_group_heads = num_heads // num_kv_heads
    grid = lambda meta: (
        num_tokens,
        triton.cdiv(topk + extra_topk, meta["GROUP_SIZE"]),
        num_kv_heads * triton.cdiv(actual_group_heads, meta["BLOCK_H"]),
    )
    flash_mla_sparse_with_kv_cache_v2_gluon_kernel[grid](
        q,
        kv_cache,
        indices,
        topk_length,
        extra_kv_cache if has_extra else kv_cache,
        extra_indices if has_extra else indices,
        extra_topk_length if has_extra else topk_length,
        middle,
        group_maxs,
        group_expsums,
        sm_scale,
        TOPK=topk,
        EXTRA_TOPK=extra_topk,
        PAGE_BLOCK_SIZE=page_size,
        EXTRA_PAGE_BLOCK_SIZE=extra_page_size,
        HAS_EXTRA_K_CACHE=has_extra,
        H_Q=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        D_V=head_dim,
    )

    group_size = _best_v2_gluon_param("GROUP_SIZE", _DEFAULT_V2_GLUON_KERNEL_CONFIG.kwargs["GROUP_SIZE"])
    ceil_groups = triton.cdiv(topk + extra_topk, group_size)
    rd_grid = lambda meta: (
        num_tokens,
        num_heads,
        triton.cdiv(head_dim, meta["BLOCK_V"]),
    )
    flash_mla_sparse_with_kv_cache_rd_kernel[rd_grid](
        middle,
        group_maxs,
        group_expsums,
        attn_sink,
        out,
        topk_length,
        extra_topk_length if has_extra else topk_length,
        H_Q=num_heads,
        D_V=head_dim,
        CEIL_GROUPS=ceil_groups,
        GROUP_SIZE=group_size,
        HAS_EXTRA_K_CACHE=has_extra,
        HAS_ATTN_SINK=True,
    )
    return out



def flash_mla_sparse_with_kv_cache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor,
    d_v: int = 512,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert q.is_contiguous() and q.dtype in (torch.float16, torch.bfloat16)
    assert q.dim() == 4 and q.size(1) == 1 and q.size(3) == d_v, (
        f"q must be contiguous fp16/bf16 [num_tokens, 1, num_heads, head_dim], got {tuple(q.shape)}"
    )
    num_tokens = q.size(0)
    num_heads = q.size(2)
    head_dim = q.size(3)
    assert d_v == head_dim, "d_v must equal q head_dim"
    assert kv_cache.is_contiguous() and kv_cache.dtype == q.dtype and kv_cache.dim() == 4
    assert kv_cache.size(2) == 1 and kv_cache.size(3) == head_dim, (
        f"kv_cache must be [num_kv, page_size, 1, head_dim], got {tuple(kv_cache.shape)}"
    )
    num_kv_heads = kv_cache.size(2)
    page_size = kv_cache.size(1)
    assert indices.is_contiguous() and indices.dtype is torch.int32 and indices.dim() == 3
    assert indices.size(0) == num_tokens and indices.size(1) == num_kv_heads, (
        f"indices must be [num_tokens, num_kv_heads, topk], got {tuple(indices.shape)}"
    )
    topk = indices.size(2)
    assert topk_length.is_contiguous() and topk_length.dtype is torch.int32
    assert topk_length.dim() == 1 and topk_length.size(0) == num_tokens, (
        f"topk_length must be [num_tokens], got {tuple(topk_length.shape)}"
    )
    assert attn_sink.is_contiguous() and attn_sink.dtype is torch.float32
    assert attn_sink.dim() == 1 and attn_sink.size(0) == num_heads, (
        f"attn_sink must be [num_heads], got {tuple(attn_sink.shape)}"
    )
    has_extra = extra_kv_cache is not None
    assert has_extra == (extra_indices is not None) and has_extra == (extra_topk_length is not None), (
        "extra_kv_cache, extra_indices, and extra_topk_length must be provided together"
    )
    if has_extra:
        assert extra_kv_cache.is_contiguous() and extra_kv_cache.dtype == q.dtype
        assert extra_kv_cache.dim() == 4 and extra_kv_cache.size(2) == 1 and extra_kv_cache.size(3) == head_dim
        assert extra_indices.is_contiguous() and extra_indices.dtype is torch.int32
        assert extra_indices.dim() == 3 and extra_indices.size(0) == num_tokens and extra_indices.size(1) == num_kv_heads
        assert extra_topk_length.is_contiguous() and extra_topk_length.dtype is torch.int32
        assert extra_topk_length.dim() == 1 and extra_topk_length.size(0) == num_tokens
        extra_topk = extra_indices.size(2)
        extra_page_size = extra_kv_cache.size(1)
    else:
        extra_topk = 0
        extra_page_size = 1

    ensure_fp16_bf16_contiguous(q)
    ensure_fp16_bf16_contiguous(kv_cache)
    ensure_int32_contiguous(indices)
    ensure_int32_contiguous(topk_length)
    ensure_fp32_contiguous(attn_sink)
    if has_extra:
        ensure_fp16_bf16_contiguous(extra_kv_cache)
        ensure_int32_contiguous(extra_indices)
        ensure_int32_contiguous(extra_topk_length)

    if out is None:
        out = torch.empty((num_tokens, 1, num_heads, d_v), dtype=q.dtype, device=q.device)
    else:
        assert out.is_contiguous() and out.dtype == q.dtype
        assert out.shape == (num_tokens, 1, num_heads, d_v), f"out shape mismatch: {tuple(out.shape)}"
        ensure_fp16_bf16_contiguous(out)

    if ENABLE_AUTOTUNE:
        min_group_size = min(config.kwargs["GROUP_SIZE"] for config in _FD_KERNEL_CONFIGS)
    else:
        min_group_size = _DEFAULT_FD_KERNEL_CONFIG.kwargs["GROUP_SIZE"]
    max_groups = triton.cdiv(topk + extra_topk, min_group_size)

    middle = torch.empty((max_groups, num_tokens, num_heads, head_dim), dtype=torch.float32, device=q.device)
    group_maxs = torch.empty((num_tokens, num_heads, max_groups), dtype=torch.float32, device=q.device)
    group_expsums = torch.empty((num_tokens, num_heads, max_groups), dtype=torch.float32, device=q.device)

    grid = lambda meta: (
        triton.cdiv(num_heads, meta["BLOCK_H"]),
        num_tokens,
        triton.cdiv(topk + extra_topk, meta["GROUP_SIZE"]),
    )
    flash_mla_sparse_with_kv_cache_fd_kernel[grid](
        q,
        kv_cache,
        indices,
        topk_length,
        extra_kv_cache if has_extra else kv_cache,
        extra_indices if has_extra else indices,
        extra_topk_length if has_extra else topk_length,
        middle,
        group_maxs,
        group_expsums,
        sm_scale,
        TOPK=topk,
        EXTRA_TOPK=extra_topk,
        PAGE_BLOCK_SIZE=page_size,
        EXTRA_PAGE_BLOCK_SIZE=extra_page_size,
        HAS_EXTRA_K_CACHE=has_extra,
        H_Q=num_heads,
        D_V=head_dim)

    group_size = _best_fd_param("GROUP_SIZE", _DEFAULT_FD_CONFIG["GROUP_SIZE"])
    ceil_groups = triton.cdiv(topk + extra_topk, group_size)
    rd_grid = lambda meta: (
        num_tokens,
        num_heads,
        triton.cdiv(head_dim, meta["BLOCK_V"]),
    )
    flash_mla_sparse_with_kv_cache_rd_kernel[rd_grid](
        middle,
        group_maxs,
        group_expsums,
        attn_sink,
        out,
        topk_length,
        extra_topk_length if has_extra else topk_length,
        H_Q=num_heads,
        D_V=head_dim,
        CEIL_GROUPS=ceil_groups,
        GROUP_SIZE=group_size,
        HAS_EXTRA_K_CACHE=has_extra,
        HAS_ATTN_SINK=True)
    return out


def flash_mla_sparse_with_kv_cache_fa(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor,
    d_v: int = 512,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Single-kernel FA variant: end-to-end attention without fd/rd split."""
    assert q.is_contiguous() and q.dtype in (torch.float16, torch.bfloat16)
    assert q.dim() == 4 and q.size(1) == 1 and q.size(3) == d_v, (
        f"q must be contiguous fp16/bf16 [num_tokens, 1, num_heads, head_dim], got {tuple(q.shape)}"
    )
    num_tokens = q.size(0)
    num_heads = q.size(2)
    head_dim = q.size(3)
    assert d_v == head_dim, "d_v must equal q head_dim"
    assert kv_cache.is_contiguous() and kv_cache.dtype == q.dtype and kv_cache.dim() == 4
    assert kv_cache.size(2) == 1 and kv_cache.size(3) == head_dim, (
        f"kv_cache must be [num_kv, page_size, 1, head_dim], got {tuple(kv_cache.shape)}"
    )
    page_size = kv_cache.size(1)
    assert indices.is_contiguous() and indices.dtype is torch.int32 and indices.dim() == 3
    assert indices.size(0) == num_tokens and indices.size(1) == kv_cache.size(2), (
        f"indices must be [num_tokens, num_kv_heads, topk], got {tuple(indices.shape)}"
    )
    topk = indices.size(2)
    assert topk_length.is_contiguous() and topk_length.dtype is torch.int32
    assert topk_length.dim() == 1 and topk_length.size(0) == num_tokens, (
        f"topk_length must be [num_tokens], got {tuple(topk_length.shape)}"
    )
    assert attn_sink.is_contiguous() and attn_sink.dtype is torch.float32
    assert attn_sink.dim() == 1 and attn_sink.size(0) == num_heads, (
        f"attn_sink must be [num_heads], got {tuple(attn_sink.shape)}"
    )
    has_extra = extra_kv_cache is not None
    assert has_extra == (extra_indices is not None) and has_extra == (extra_topk_length is not None), (
        "extra_kv_cache, extra_indices, and extra_topk_length must be provided together"
    )
    if has_extra:
        assert extra_kv_cache.is_contiguous() and extra_kv_cache.dtype == q.dtype
        assert extra_kv_cache.dim() == 4 and extra_kv_cache.size(2) == 1 and extra_kv_cache.size(3) == head_dim
        assert extra_indices.is_contiguous() and extra_indices.dtype is torch.int32
        assert extra_indices.dim() == 3 and extra_indices.size(0) == num_tokens and extra_indices.size(1) == kv_cache.size(2)
        assert extra_topk_length.is_contiguous() and extra_topk_length.dtype is torch.int32
        assert extra_topk_length.dim() == 1 and extra_topk_length.size(0) == num_tokens
        extra_topk = extra_indices.size(2)
        extra_page_size = extra_kv_cache.size(1)
    else:
        extra_topk = 0
        extra_page_size = 1

    ensure_fp16_bf16_contiguous(q)
    ensure_fp16_bf16_contiguous(kv_cache)
    ensure_int32_contiguous(indices)
    ensure_int32_contiguous(topk_length)
    ensure_fp32_contiguous(attn_sink)
    if has_extra:
        ensure_fp16_bf16_contiguous(extra_kv_cache)
        ensure_int32_contiguous(extra_indices)
        ensure_int32_contiguous(extra_topk_length)

    if out is None:
        out = torch.empty((num_tokens, 1, num_heads, d_v), dtype=q.dtype, device=q.device)
    else:
        assert out.is_contiguous() and out.dtype == q.dtype
        assert out.shape == (num_tokens, 1, num_heads, d_v), f"out shape mismatch: {tuple(out.shape)}"
        ensure_fp16_bf16_contiguous(out)

    grid = lambda meta: (
        triton.cdiv(num_heads, meta["BLOCK_H"]),
        num_tokens,
    )
    flash_mla_sparse_with_kv_cache_fa_kernel[grid](
        q,
        kv_cache,
        indices,
        topk_length,
        extra_kv_cache if has_extra else kv_cache,
        extra_indices if has_extra else indices,
        extra_topk_length if has_extra else topk_length,
        attn_sink,
        out,
        sm_scale,
        TOPK=topk,
        EXTRA_TOPK=extra_topk,
        PAGE_BLOCK_SIZE=page_size,
        EXTRA_PAGE_BLOCK_SIZE=extra_page_size,
        HAS_EXTRA_K_CACHE=has_extra,
        HAS_ATTN_SINK=True,
        H_Q=num_heads,
        D_V=head_dim)
    return out


SWA_TOPK = 128


def default_extra_topk(
    compress_ratio: int,
    *,
    mode: Literal["flash", "pro"] = "flash",
) -> int:
    """Return fixed extra_topk capacity for c1/c4 layers."""
    if compress_ratio <= 1:
        return 0
    if compress_ratio == 4:
        return 1024 if mode == "pro" else 512
    raise ValueError("c128 layer requires explicit extra_topk")


def extra_page_block_size_for_compress_ratio(compress_ratio: int, page_block_size: int) -> int:
    if compress_ratio <= 1:
        return 1
    if compress_ratio == 4:
        return page_block_size
    if compress_ratio == 128:
        return 2
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected 1, 4, or 128")


def _fill_indices(
    indices: torch.Tensor,
    valid_len: int,
    *,
    cache_tokens: int,
    seed_stride: int,
) -> None:
    max_start = max(1, cache_tokens - valid_len)
    for batch_idx in range(indices.shape[0]):
        for kv_head_idx in range(indices.shape[1]):
            start = (2176 + batch_idx * 8191 + kv_head_idx * seed_stride) % max_start
            indices[batch_idx, kv_head_idx, :valid_len] = torch.arange(
                start,
                start + valid_len,
                device=indices.device,
                dtype=torch.int32,
            )


def make_decode_inputs(
    *,
    valid_topk: int = SWA_TOPK,
    extra_topk: int = 0,
    valid_extra_topk: int = 0,
    compress_ratio: int = 1,
    mode: Literal["flash", "pro"] = "flash",
    batch_size: int = 1,
    num_heads: int = 8,
    num_kv_heads: int = 1,
    head_dim: int = 512,
    num_blocks: int = 17606,
    page_block_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    use_attn_sink: bool = True,
    device: str = "cuda",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Build decode inputs for DeepSeek v4 c1 / c4 / c128 layers.

    c1: extra_topk=0, valid_extra_topk=0, SWA topk fixed at 128.
    c4: extra_topk defaults to 512 (flash) or 1024 (pro) when omitted via compress_ratio=4.
    c128: caller must pass extra_topk and valid_extra_topk explicitly.
    """
    if compress_ratio == 128 and extra_topk == 0:
        raise ValueError("c128 layer requires explicit extra_topk")
    if compress_ratio <= 1:
        extra_topk = 0
        valid_extra_topk = 0
    elif extra_topk == 0:
        extra_topk = default_extra_topk(compress_ratio, mode=mode)

    valid_topk = min(valid_topk, SWA_TOPK)
    valid_extra_topk = min(valid_extra_topk, extra_topk)
    extra_page_block_size = extra_page_block_size_for_compress_ratio(compress_ratio, page_block_size)
    sm_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn((batch_size, 1, num_heads, head_dim), device=device, dtype=dtype) / 10
    kv_cache = torch.randn(
        (num_blocks, page_block_size, num_kv_heads, head_dim),
        device=device,
        dtype=dtype,
    ) / 10
    indices = torch.full((batch_size, num_kv_heads, SWA_TOPK), -1, device=device, dtype=torch.int32)
    _fill_indices(
        indices,
        valid_topk,
        cache_tokens=num_blocks * page_block_size,
        seed_stride=4099,
    )
    topk_length = torch.full((batch_size,), valid_topk, device=device, dtype=torch.int32)

    extra_kv_cache = None
    extra_indices = None
    extra_topk_length = None
    if extra_topk > 0 and valid_extra_topk > 0:
        extra_kv_cache = torch.randn(
            (num_blocks, extra_page_block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        ) / 10
        extra_indices = torch.full(
            (batch_size, num_kv_heads, extra_topk),
            -1,
            device=device,
            dtype=torch.int32,
        )
        _fill_indices(
            extra_indices,
            valid_extra_topk,
            cache_tokens=num_blocks * extra_page_block_size,
            seed_stride=6151,
        )
        extra_topk_length = torch.full((batch_size,), valid_extra_topk, device=device, dtype=torch.int32)

    if use_attn_sink:
        attn_sink = torch.randn((num_heads,), device=device, dtype=torch.float32) / 10
    else:
        attn_sink = torch.zeros((num_heads,), device=device, dtype=torch.float32)

    return (
        q,
        kv_cache,
        indices,
        sm_scale,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
    )


flash_mla_with_kvcache = flash_mla_sparse_with_kv_cache


def _parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(" ", "").split(",") if item]


def _default_cases_for_compress_ratio(
    compress_ratio: int,
    mode: Literal["flash", "pro"],
) -> list[tuple[int, int]]:
    if compress_ratio <= 1:
        return [(0, 0)]
    if compress_ratio == 4:
        extra_topk = default_extra_topk(4, mode=mode)
        return [(extra_topk, 512), (extra_topk, min(640, extra_topk))]
    if compress_ratio == 128:
        return [(128, 128), (256, 128), (256, 256)]
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}")


def _benchmark_kernel(fn, args=(), *, warmup=25, runs=100, use_cudagraph=False, **kwargs) -> float:
    run = (lambda: fn(*args, **kwargs)) if kwargs else (lambda: fn(*args))
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    if use_cudagraph:
        return triton.testing.do_bench_cudagraph(run, rep=runs, return_mode="median")
    return triton.testing.do_bench(run, warmup=warmup, rep=runs, return_mode="median")


def _run_benchmark_case(extra_topk: int, valid_extra_topk: int, args: argparse.Namespace) -> None:
    (
        q,
        kv_cache,
        indices,
        sm_scale,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
    ) = make_decode_inputs(
        valid_topk=SWA_TOPK,
        extra_topk=extra_topk,
        valid_extra_topk=valid_extra_topk,
        compress_ratio=args.compress_ratio,
        mode=args.mode,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_blocks=args.num_blocks,
        page_block_size=args.page_block_size,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16,
        use_attn_sink=not args.no_attn_sink,
    )

    swa_len = int(topk_length[0].item())
    extra_len = 0 if extra_topk_length is None else int(extra_topk_length[0].item())
    print(
        f"cr={args.compress_ratio} mode={args.mode} "
        f"swa_len={swa_len} extra_topk={extra_topk} valid_extra_topk={extra_len} "
        f"total={swa_len + extra_len} bs={args.batch_size} h={args.num_heads} "
        f"page={kv_cache.shape[1]} "
        f"extra_page={None if extra_kv_cache is None else extra_kv_cache.shape[1]}"
    )

    common_kwargs = dict(
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
    )
    common_args = (q, kv_cache, indices, topk_length, sm_scale, attn_sink)

    run_check = args.check or args.impl == "compare"
    if run_check or args.impl in ("compare", "fa"):
        fa_out = flash_mla_sparse_with_kv_cache_fa(*common_args, **common_kwargs)
        torch.cuda.synchronize()
        assert fa_out.shape == q.shape

    if run_check or args.impl in ("compare", "split"):
        split_out = flash_mla_sparse_with_kv_cache(*common_args, **common_kwargs)
        torch.cuda.synchronize()
        assert split_out.shape == q.shape

    if run_check or args.impl in ("compare", "v2"):
        v2_out = flash_mla_sparse_with_kv_cache_v2_gluon(*common_args, **common_kwargs)
        torch.cuda.synchronize()
        assert v2_out.shape == q.shape

    if run_check:
        torch.testing.assert_close(split_out, fa_out, atol=1e-2, rtol=1e-2)
        print("check: split vs fa passed")
        torch.testing.assert_close(v2_out, fa_out, atol=1e-2, rtol=1e-2)
        print("check: v2 gluon vs fa passed")

    impls: list[tuple[str, object]] = []
    if args.impl in ("split", "compare"):
        impls.append(("split", flash_mla_sparse_with_kv_cache))
    if args.impl in ("fa", "compare"):
        impls.append(("fa", flash_mla_sparse_with_kv_cache_fa))
    if args.impl in ("v2", "compare"):
        impls.append(("v2", flash_mla_sparse_with_kv_cache_v2_gluon))

    for tag, fn in impls:
        ms = _benchmark_kernel(
            fn,
            args=common_args,
            warmup=args.warmup,
            runs=args.rep,
            use_cudagraph=not args.no_cudagraph,
            **common_kwargs,
        )
        print(
            f"extra_topk={extra_topk:4d} valid_extra_topk={valid_extra_topk:4d} "
            f"impl={tag} latency={ms:.4f} ms"
        )
    print("kernel_config:")
    _print_autotune_configs(
        include_fa=args.impl in ("fa", "compare"),
        include_v2=args.impl in ("v2", "compare"),
    )


def main(cases: Iterable[tuple[int, int]] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark flash_mla_sparse_with_kv_cache for DeepSeek v4 c1/c4/c128 layers."
    )
    parser.add_argument(
        "--extra-topk",
        default=None,
        help="Comma-separated extra_topk capacities. Defaults per --compress-ratio.",
    )
    parser.add_argument(
        "--valid-extra-topk",
        default=None,
        help="Comma-separated valid extra lengths, paired with --extra-topk.",
    )
    parser.add_argument(
        "--compress-ratio",
        default="1,4,128",
        help="Comma-separated compress ratios: 1 (c1), 4 (c4), 128 (c128).",
    )
    parser.add_argument(
        "--mode",
        choices=("flash", "pro"),
        default="flash",
        help="c4 default extra_topk: flash=512, pro=1024.",
    )
    parser.add_argument(
        "--batch-size",
        "--bs",
        default="256",
        help="Comma-separated batch sizes.",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, default=17606)
    parser.add_argument("--page-block-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--no-attn-sink", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--no-cudagraph", action="store_true")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable kernel autotuning (overrides ENABLE_AUTOTUNE=False at import time).",
    )
    parser.add_argument(
        "--impl",
        choices=("split", "fa", "v2", "compare"),
        default="compare",
        help="compare=benchmark split (fd+rd), fa, and v2 Gluon; split=fd+rd only, fa=single-kernel only.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=None,
        help="Assert split and fa outputs match before benchmarking (on by default for --impl compare).",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip correctness check even when --impl compare.",
    )
    args = parser.parse_args()
    if args.no_check:
        args.check = False
    elif args.check is None:
        args.check = args.impl == "compare"

    compress_ratios = _parse_int_list(str(args.compress_ratio))
    if not compress_ratios:
        parser.error("--compress-ratio must contain at least one integer")
    unsupported = [r for r in compress_ratios if r not in (1, 4, 128)]
    if unsupported:
        parser.error(f"Unsupported --compress-ratio values: {unsupported}; expected 1, 4, or 128")

    explicit_extra_topk = _parse_int_list(args.extra_topk) if args.extra_topk else None
    explicit_valid_extra_topk = _parse_int_list(args.valid_extra_topk) if args.valid_extra_topk else None
    if (explicit_extra_topk is None) ^ (explicit_valid_extra_topk is None):
        parser.error("--extra-topk and --valid-extra-topk must be provided together")
    if explicit_extra_topk and len(explicit_extra_topk) != len(explicit_valid_extra_topk):
        parser.error("--extra-topk and --valid-extra-topk must have the same length")

    batch_sizes = _parse_int_list(args.batch_size)
    if not batch_sizes:
        parser.error("--batch-size/--bs must contain at least one integer")

    torch.manual_seed(0)
    for compress_ratio in compress_ratios:
        args.compress_ratio = compress_ratio
        if cases is not None:
            run_cases = list(cases)
        elif explicit_extra_topk is not None:
            run_cases = list(zip(explicit_extra_topk, explicit_valid_extra_topk))
        else:
            run_cases = _default_cases_for_compress_ratio(compress_ratio, args.mode)
        for batch_size in batch_sizes:
            args.batch_size = batch_size
            for extra_topk, valid_extra_topk in run_cases:
                _run_benchmark_case(extra_topk, valid_extra_topk, args)


if __name__ == "__main__":
    main()
