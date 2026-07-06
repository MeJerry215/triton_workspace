#
# Copyright (c) 2025 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Licensed under the Apache License, Version 2.0
#

import argparse
import math
import os
import sys

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

ENABLE_AUTOTUNE = False
if "--tune" in sys.argv:
    ENABLE_AUTOTUNE = True

TOPK = 2048

_DEFAULT_FD_KERNEL_CONFIG = triton.Config(
    {"BLOCK_H": 16, "GROUP_SIZE": 64, "BLOCK_D": 128, "BLOCK_V": 256},
    num_warps=4,
    num_stages=1,
)
_DEFAULT_RD_KERNEL_CONFIG = triton.Config({"BLOCK_V": 256}, num_warps=1)
_DEFAULT_FA_KERNEL_CONFIG = triton.Config(
    {"BLOCK_H": 16, "GROUP_SIZE": 64, "BLOCK_D": 128, "BLOCK_V": 256},
    num_warps=4,
    num_stages=1,
)

_DEFAULT_FD_CONFIG = {
    **_DEFAULT_FD_KERNEL_CONFIG.kwargs,
    "num_warps": _DEFAULT_FD_KERNEL_CONFIG.num_warps,
}

_BLOCK_HS = [16,]
_GROUP_SIZES = [16, 32, 64, 128]
_BLOCK_DS = [16, 32, 64, 128]
_BLOCK_VS = [128, 256, 512]
_FD_NUM_WARPS = [1, 2, 4, 8]
_FA_GROUP_SIZES = [64, 128, 256]
_FA_NUM_WARPS = [1, 2, 4, 8, 16]
_FA_NUM_STAGES = [1]

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

_FD_KERNEL_CONFIGS = _AUTOTUNE_CONFIGS if ENABLE_AUTOTUNE else [_DEFAULT_FD_KERNEL_CONFIG]
_FA_KERNEL_CONFIGS = _FA_AUTOTUNE_CONFIGS if ENABLE_AUTOTUNE else [_DEFAULT_FA_KERNEL_CONFIG]


def align_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) // alignment * alignment


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
        if not all(_is_power_of_2(v) for v in (block_h, group_size, block_d, block_v)):
            continue
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
_RD_KERNEL_CONFIGS = _RD_AUTOTUNE_CONFIGS if ENABLE_AUTOTUNE else [_DEFAULT_RD_KERNEL_CONFIG]


def _best_fd_param(key: str, default):
    best_config = getattr(gqa_flash_mla_sparse_fd_kernel, "best_config", None)
    if best_config is None:
        return _DEFAULT_FD_CONFIG.get(key, default)
    if key == "num_warps":
        return best_config.num_warps
    return best_config.kwargs.get(key, default)


def _format_kernel_config(name: str, config: triton.Config) -> str:
    kwargs = ", ".join(f"{k}={v}" for k, v in config.kwargs.items())
    line = f"  {name}: {{{kwargs}}}, num_warps={config.num_warps}"
    if name in ("fd_kernel", "fa_kernel"):
        line += f", num_stages={config.num_stages}"
    return line


def _print_autotune_configs(*, include_fa: bool = False) -> None:
    entries = [
        ("fd_kernel", gqa_flash_mla_sparse_fd_kernel, _DEFAULT_FD_KERNEL_CONFIG, ENABLE_AUTOTUNE),
        ("rd_kernel", gqa_flash_mla_sparse_rd_kernel, _DEFAULT_RD_KERNEL_CONFIG, ENABLE_AUTOTUNE),
    ]
    if include_fa:
        entries.append(
            ("fa_kernel", gqa_flash_mla_sparse_fa_kernel, _DEFAULT_FA_KERNEL_CONFIG, ENABLE_AUTOTUNE),
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
def _load_dense_kv_for_dot(
    kv_ptr,
    block_idx,
    pos_in_block,
    offsets,
    mask,
    block_stride: tl.constexpr,
    pos_stride: tl.constexpr):
    return tl.load(  # shape: offsets.shape (e.g. [BLOCK_D, GROUP_SIZE] or [GROUP_SIZE, D_V])
        kv_ptr + block_idx * block_stride + pos_in_block * pos_stride + offsets,
        mask=mask,
        other=0.0)


@triton.autotune(
    configs=_FD_KERNEL_CONFIGS,
    key=["H_Q", "D_V", "TOPK"],
    prune_configs_by={"early_config_prune": _prune_fd_configs} if ENABLE_AUTOTUNE else None,
)
@triton.jit
def gqa_flash_mla_sparse_fd_kernel(
    q_ptr,
    k_cache_ptr,
    indices_ptr,
    topk_length_ptr,
    middle_ptr,
    group_maxs_ptr,
    group_expsums_ptr,
    sm_scale,
    TOPK: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
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
    stride_indices_b = TOPK
    stride_middle_g = num_queries * H_Q * D_V
    stride_middle_q = H_Q * D_V
    stride_middle_h = D_V
    stride_group_q = H_Q * max_groups
    stride_group_h = max_groups

    head_offsets = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    group_offsets = tl.arange(0, GROUP_SIZE)
    d_offsets = tl.arange(0, BLOCK_D)
    v_offsets = tl.arange(0, BLOCK_V)

    context_len = tl.load(topk_length_ptr + batch_idx)  # shape: scalar
    num_groups = (context_len + GROUP_SIZE - 1) // GROUP_SIZE
    if group_idx >= num_groups:
        return

    slot_offsets = group_idx * GROUP_SIZE + group_offsets
    idx_vals = tl.load(  # shape: [GROUP_SIZE]
        indices_ptr + batch_idx * stride_indices_b + slot_offsets,
        mask=slot_offsets < context_len,
        other=-1)
    valid = (slot_offsets < context_len) & (idx_vals >= 0)
    safe_idx = tl.where(valid, idx_vals, 0)
    kv_block_idx = safe_idx // PAGE_BLOCK_SIZE
    kv_pos = safe_idx - kv_block_idx * PAGE_BLOCK_SIZE

    scores = tl.zeros((BLOCK_H, GROUP_SIZE), dtype=tl.float32)
    for d_start in tl.range(0, D_V, BLOCK_D):
        ds = d_start + d_offsets
        q_vals = tl.load(  # shape: [BLOCK_H, BLOCK_D]
            q_ptr
            + batch_idx * stride_q_b
            + head_offsets[:, None] * stride_q_h
            + ds[None, :],
            mask=(head_offsets[:, None] < H_Q) & (ds[None, :] < D_V),
            other=0.0)
        k_vals = _load_dense_kv_for_dot(  # shape: [BLOCK_D, GROUP_SIZE]
            k_cache_ptr,
            kv_block_idx[None, :],
            kv_pos[None, :],
            ds[:, None],
            valid[None, :] & (ds[:, None] < D_V),
            stride_k_block,
            D_V)
        scores += tl.dot(q_vals, k_vals, out_dtype=tl.float32) * sm_scale  # [BLOCK_H, BLOCK_D] @ [BLOCK_D, GROUP_SIZE] -> [BLOCK_H, GROUP_SIZE]

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

    for v_start in tl.range(0, D_V, BLOCK_V):
        vs = v_start + v_offsets
        v_vals = _load_dense_kv_for_dot(  # shape: [GROUP_SIZE, BLOCK_V]
            k_cache_ptr,
            kv_block_idx[:, None],
            kv_pos[:, None],
            vs[None, :],
            valid[:, None] & (vs[None, :] < D_V),
            stride_k_block,
            D_V)
        partial = tl.dot(probs_for_dot, v_vals, out_dtype=tl.float32)  # [BLOCK_H, GROUP_SIZE] @ [GROUP_SIZE, BLOCK_V] -> [BLOCK_H, BLOCK_V]
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
    key=["H_Q", "D_V", "CEIL_GROUPS"],
)
@triton.jit
def gqa_flash_mla_sparse_rd_kernel(
    middle_ptr,
    group_maxs_ptr,
    group_expsums_ptr,
    out_ptr,
    topk_length_ptr,
    H_Q: tl.constexpr,
    D_V: tl.constexpr,
    CEIL_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
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

    context_len = tl.load(topk_length_ptr + batch_idx)  # shape: scalar
    effective_groups = (context_len + GROUP_SIZE - 1) // GROUP_SIZE

    final_max = tl.full((), -float("inf"), dtype=tl.float32)
    for group_idx in tl.range(0, effective_groups):
        group_max = tl.load(  # shape: scalar
            group_maxs_ptr + query_idx * stride_group_q + head_idx * stride_group_h + group_idx,
            mask=head_idx < H_Q,
            other=-float("inf"))
        final_max = tl.maximum(final_max, group_max)

    denom = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for group_idx in tl.range(0, effective_groups):
        group_max = tl.load(  # shape: scalar
            group_maxs_ptr + query_idx * stride_group_q + head_idx * stride_group_h + group_idx,
            mask=head_idx < H_Q,
            other=-float("inf"))
        group_exp_sum = tl.load(  # shape: scalar
            group_expsums_ptr
            + query_idx * stride_group_q
            + head_idx * stride_group_h
            + group_idx,
            mask=head_idx < H_Q,
            other=0.0)
        weight = tl.exp(group_max - final_max) * group_exp_sum
        partial = tl.load(  # shape: [BLOCK_V]
            middle_ptr
            + group_idx * stride_middle_g
            + query_idx * stride_middle_q
            + head_idx * stride_middle_h
            + v_offsets,
            mask=(head_idx < H_Q) & (v_offsets < D_V),
            other=0.0)
        acc += partial * weight
        denom += weight

    has_valid = denom > 0.0
    out_vals = tl.where(has_valid, acc / denom, 0.0)
    tl.store(
        out_ptr
        + batch_idx * stride_out_b
        + head_idx * stride_out_h
        + v_offsets,
        out_vals,
        mask=(head_idx < H_Q) & (v_offsets < D_V))


@triton.autotune(
    configs=_FA_KERNEL_CONFIGS,
    key=["H_Q", "D_V", "TOPK"],
    prune_configs_by={"early_config_prune": _prune_fd_configs},
)
@triton.jit
def gqa_flash_mla_sparse_fa_kernel(
    q_ptr,
    k_cache_ptr,
    indices_ptr,
    topk_length_ptr,
    out_ptr,
    sm_scale,
    TOPK: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
    H_Q: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr):
    head_block_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    batch_idx = query_idx

    stride_q_b = H_Q * D_V
    stride_q_h = D_V
    stride_k_block = PAGE_BLOCK_SIZE * D_V
    stride_indices_b = TOPK
    stride_out_b = H_Q * D_V
    stride_out_h = D_V

    head_offsets = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    group_offsets = tl.arange(0, GROUP_SIZE)
    d_offsets = tl.arange(0, BLOCK_D)
    v_offsets = tl.arange(0, D_V)

    context_len = tl.load(topk_length_ptr + batch_idx)  # shape: scalar
    effective_groups = (context_len + GROUP_SIZE - 1) // GROUP_SIZE

    running_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    running_denom = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D_V), dtype=tl.float32)

    for group_idx in tl.range(0, effective_groups):
        slot_offsets = group_idx * GROUP_SIZE + group_offsets
        idx_vals = tl.load(  # shape: [GROUP_SIZE]
            indices_ptr + batch_idx * stride_indices_b + slot_offsets,
            mask=slot_offsets < context_len,
            other=-1)
        valid = (slot_offsets < context_len) & (idx_vals >= 0)
        safe_idx = tl.where(valid, idx_vals, 0)
        kv_block_idx = safe_idx // PAGE_BLOCK_SIZE
        kv_pos = safe_idx - kv_block_idx * PAGE_BLOCK_SIZE

        scores = tl.zeros((BLOCK_H, GROUP_SIZE), dtype=tl.float32)
        for d_start in tl.range(0, D_V, BLOCK_D):
            ds = d_start + d_offsets
            q_vals = tl.load(  # shape: [BLOCK_H, BLOCK_D]
                q_ptr
                + batch_idx * stride_q_b
                + head_offsets[:, None] * stride_q_h
                + ds[None, :],
                mask=(head_offsets[:, None] < H_Q) & (ds[None, :] < D_V),
                other=0.0)
            k_vals = _load_dense_kv_for_dot(  # shape: [BLOCK_D, GROUP_SIZE]
                k_cache_ptr,
                kv_block_idx[None, :],
                kv_pos[None, :],
                ds[:, None],
                valid[None, :] & (ds[:, None] < D_V),
                stride_k_block,
                D_V)
            scores += tl.dot(q_vals, k_vals, out_dtype=tl.float32) * sm_scale  # [BLOCK_H, BLOCK_D] @ [BLOCK_D, GROUP_SIZE] -> [BLOCK_H, GROUP_SIZE]

        scores = tl.where(valid[None, :] & (head_offsets[:, None] < H_Q), scores, -float("inf"))
        group_max = tl.max(scores, axis=1)
        exp_scores = tl.where(valid[None, :], tl.exp(scores - group_max[:, None]), 0.0)
        exp_sum = tl.sum(exp_scores, axis=1)
        head_active = exp_sum > 0.0

        if q_ptr.dtype.element_ty == tl.bfloat16:
            exp_for_dot = exp_scores.to(tl.bfloat16)
        else:
            exp_for_dot = exp_scores.to(tl.float16)

        v_vals = _load_dense_kv_for_dot(  # shape: [GROUP_SIZE, D_V]
            k_cache_ptr,
            kv_block_idx[:, None],
            kv_pos[:, None],
            v_offsets[None, :],
            valid[:, None] & (v_offsets[None, :] < D_V),
            stride_k_block,
            D_V)
        partial_unnorm = tl.dot(exp_for_dot, v_vals, out_dtype=tl.float32)  # [BLOCK_H, GROUP_SIZE] @ [GROUP_SIZE, D_V] -> [BLOCK_H, D_V]

        new_max = tl.where(head_active, tl.maximum(running_max, group_max), running_max)
        scale_old = tl.exp(running_max - new_max)
        scale_group = tl.where(head_active, tl.exp(group_max - new_max), 0.0)
        acc = acc * scale_old[:, None] + partial_unnorm * scale_group[:, None]
        running_denom = running_denom * scale_old + exp_sum * scale_group
        running_max = new_max

    has_valid = running_denom > 0.0
    out_vals = tl.where(has_valid[:, None], acc / running_denom[:, None], 0.0)
    tl.store(
        out_ptr
        + batch_idx * stride_out_b
        + head_offsets[:, None] * stride_out_h
        + v_offsets[None, :],
        out_vals,
        mask=(head_offsets[:, None] < H_Q) & (v_offsets[None, :] < D_V))


@gluon.jit
def gqa_flash_mla_sparse_fa_gluon_kernel(q_ptr, k_cache_ptr, indices_ptr, topk_length_ptr, out_ptr, sm_scale, TOPK: gl.constexpr, PAGE_BLOCK_SIZE: gl.constexpr, H_Q: gl.constexpr, D_V: gl.constexpr, BLOCK_H: gl.constexpr, GROUP_SIZE: gl.constexpr, BLOCK_D: gl.constexpr, BLOCK_V: gl.constexpr):
    # Layouts from gqa_flash_mla_sparse_fa_kernel.ttgir (BLOCK_H=16, GROUP_SIZE=64, BLOCK_D=128, D_V=512, num_warps=4).
    q_layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    k_layout: gl.constexpr = gl.BlockedLayout([8, 1], [16, 2], [1, 4], [0, 1])
    v_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [2, 2], [1, 0])
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    q_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
    k_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[0, 1])
    v_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
    f32_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
    elem_ty: gl.constexpr = q_ptr.dtype.element_ty
    k_width: gl.constexpr = 32 // elem_ty.primitive_bitwidth
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=k_width)
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=k_width)
    head_block_idx = gl.program_id(0)
    query_idx = gl.program_id(1)
    batch_idx = query_idx
    stride_q_b = H_Q * D_V
    stride_q_h = D_V
    stride_k_block = PAGE_BLOCK_SIZE * D_V
    stride_indices_b = TOPK
    stride_out_b = H_Q * D_V
    stride_out_h = D_V
    head_offsets = head_block_idx * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, q_layout))
    head_offsets_v = head_block_idx * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, v_layout))
    head_offsets_mma = head_block_idx * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mma_layout))
    group_offsets = gl.arange(0, GROUP_SIZE, layout=gl.SliceLayout(0, k_layout))
    group_offsets_v = gl.arange(0, GROUP_SIZE, layout=gl.SliceLayout(1, v_layout))
    d_offsets = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, q_layout))
    d_offsets_k = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(1, k_layout))
    v_offsets = gl.arange(0, D_V, layout=gl.SliceLayout(0, v_layout))
    q_smem = gl.allocate_shared_memory(elem_ty, [2, BLOCK_H, BLOCK_D], layout=q_smem_layout)
    k_smem = gl.allocate_shared_memory(elem_ty, [2, BLOCK_D, GROUP_SIZE], layout=k_smem_layout)
    v_smem = gl.allocate_shared_memory(elem_ty, [GROUP_SIZE, D_V], layout=v_smem_layout)
    scores_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_H, GROUP_SIZE], layout=f32_smem_layout)
    acc_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_H, D_V], layout=f32_smem_layout)
    scores_smem.store(gl.zeros([BLOCK_H, GROUP_SIZE], gl.float32, layout=mma_layout))
    acc_smem.store(gl.zeros([BLOCK_H, D_V], gl.float32, layout=mma_layout))
    context_len = gl.load(topk_length_ptr + batch_idx)
    effective_groups = (context_len + GROUP_SIZE - 1) // GROUP_SIZE
    running_max = gl.full([BLOCK_H], -float("inf"), gl.float32, layout=gl.SliceLayout(1, mma_layout))
    running_denom = gl.zeros([BLOCK_H], gl.float32, layout=gl.SliceLayout(1, mma_layout))
    q_base = q_ptr + batch_idx * stride_q_b
    indices_base = indices_ptr + batch_idx * stride_indices_b
    head_mask_q = head_offsets[:, None] < H_Q
    head_mask_mma = head_offsets_mma[:, None] < H_Q
    head_mask_v = head_offsets_v[:, None] < H_Q
    for group_idx in range(0, effective_groups):
        slot_offsets = group_idx * GROUP_SIZE + group_offsets
        idx_vals = gl.load(indices_base + slot_offsets, mask=slot_offsets < context_len, other=-1)
        valid = (slot_offsets < context_len) & (idx_vals >= 0)
        valid_mma = gl.convert_layout(valid, gl.SliceLayout(0, mma_layout))
        safe_idx = gl.where(valid, idx_vals, 0)
        kv_block_idx = safe_idx // PAGE_BLOCK_SIZE
        kv_pos = safe_idx - kv_block_idx * PAGE_BLOCK_SIZE
        scores_smem.store(gl.zeros([BLOCK_H, GROUP_SIZE], gl.float32, layout=mma_layout))
        qk_copy_idx = 0
        qk_read_idx = 0
        ds = qk_copy_idx * BLOCK_D + d_offsets
        ds_k = qk_copy_idx * BLOCK_D + d_offsets_k
        q_mask = head_mask_q & (ds[None, :] < D_V)
        q_ptrs = q_base + head_offsets[:, None] * stride_q_h + ds[None, :]
        cp.async_copy_global_to_shared(q_smem.index(qk_copy_idx % 2), q_ptrs, mask=q_mask)
        k_mask = valid[None, :] & (ds_k[:, None] < D_V)
        k_ptrs = k_cache_ptr + kv_block_idx[None, :] * stride_k_block + kv_pos[None, :] * D_V + ds_k[:, None]
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
            k_ptrs = k_cache_ptr + kv_block_idx[None, :] * stride_k_block + kv_pos[None, :] * D_V + ds_k[:, None]
            cp.async_copy_global_to_shared(k_smem.index(qk_copy_idx % 2), k_ptrs, mask=k_mask)
            cp.commit_group()
            cp.wait_group(1)
            q_vals = q_smem.index(qk_read_idx % 2).load(q_dot_layout)
            k_vals = k_smem.index(qk_read_idx % 2).load(k_dot_layout)
            scores = scores_smem.load(mma_layout)
            partial = mma_v2(q_vals, k_vals, gl.zeros([BLOCK_H, GROUP_SIZE], gl.float32, layout=mma_layout), input_precision="tf32")
            scores = scores + partial * sm_scale
            scores_smem.store(scores)
            qk_copy_idx += 1
            qk_read_idx += 1
        cp.wait_group(0)
        q_vals = q_smem.index(qk_read_idx % 2).load(q_dot_layout)
        k_vals = k_smem.index(qk_read_idx % 2).load(k_dot_layout)
        scores = scores_smem.load(mma_layout)
        partial = mma_v2(q_vals, k_vals, gl.zeros([BLOCK_H, GROUP_SIZE], gl.float32, layout=mma_layout), input_precision="tf32")
        scores = scores + partial * sm_scale
        scores_smem.store(scores)
        scores = scores_smem.load(mma_layout)
        scores = gl.where(valid_mma[None, :] & head_mask_mma, scores, -float("inf"))
        scores_smem.store(scores)
        scores = scores_smem.load(mma_layout)
        group_max = gl.max(scores, axis=1)
        exp_scores = gl.where(valid_mma[None, :], gl.exp(scores - group_max[:, None]), 0.0)
        exp_sum = gl.sum(exp_scores, axis=1)
        head_active = exp_sum > 0.0
        if elem_ty == gl.bfloat16:
            exp_for_dot = exp_scores.to(gl.bfloat16)
        else:
            exp_for_dot = exp_scores.to(gl.float16)
        valid_v = gl.convert_layout(valid, gl.SliceLayout(1, v_layout))
        kv_block_idx_v = gl.convert_layout(kv_block_idx, gl.SliceLayout(1, v_layout))
        kv_pos_v = gl.convert_layout(kv_pos, gl.SliceLayout(1, v_layout))
        v_mask = valid_v[:, None] & (v_offsets[None, :] < D_V)
        v_ptrs = k_cache_ptr + kv_block_idx_v[:, None] * stride_k_block + kv_pos_v[:, None] * D_V + v_offsets[None, :]
        cp.async_copy_global_to_shared(v_smem, v_ptrs, mask=v_mask)
        cp.commit_group()
        cp.wait_group(0)
        exp_smem = gl.allocate_shared_memory(elem_ty, [BLOCK_H, GROUP_SIZE], layout=q_smem_layout)
        exp_smem.store(gl.convert_layout(exp_for_dot, q_layout))
        exp_dot = exp_smem.load(q_dot_layout)
        v_vals = v_smem.load(k_dot_layout)
        partial_unnorm = mma_v2(exp_dot, v_vals, gl.zeros([BLOCK_H, D_V], gl.float32, layout=mma_layout), input_precision="tf32")
        new_max = gl.where(head_active, gl.maximum(running_max, group_max), running_max)
        scale_old = gl.exp(running_max - new_max)
        scale_group = gl.where(head_active, gl.exp(group_max - new_max), 0.0)
        acc = acc_smem.load(mma_layout)
        acc = acc * scale_old[:, None] + partial_unnorm * scale_group[:, None]
        acc_smem.store(acc)
        running_denom = running_denom * scale_old + exp_sum * scale_group
        running_max = new_max
    acc = acc_smem.load(mma_layout)
    has_valid = running_denom > 0.0
    out_vals = gl.where(has_valid[:, None], acc / running_denom[:, None], 0.0)
    out_vals = out_vals.to(elem_ty)
    out_vals = gl.convert_layout(out_vals, v_layout)
    out_mask = head_mask_v & (v_offsets[None, :] < D_V)
    gl.store(out_ptr + batch_idx * stride_out_b + head_offsets_v[:, None] * stride_out_h + v_offsets[None, :], out_vals, mask=out_mask)


def gqa_flash_mla_sparse_with_kv_cache_fa_gluon(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.is_contiguous() and q.dtype in (torch.float16, torch.bfloat16)
    assert q.dim() == 4 and q.size(1) == 1 and q.size(3) == d_v
    num_tokens = q.size(0)
    num_heads = q.size(2)
    head_dim = q.size(3)
    assert d_v == head_dim
    assert kv_cache.is_contiguous() and kv_cache.dtype == q.dtype and kv_cache.dim() == 4
    assert kv_cache.size(2) == 1 and kv_cache.size(3) == head_dim
    page_size = kv_cache.size(1)
    assert indices.is_contiguous() and indices.dtype is torch.int32 and indices.dim() == 3
    assert indices.size(0) == num_tokens and indices.size(1) == 1
    topk = indices.size(2)
    assert topk == TOPK
    assert topk_length.is_contiguous() and topk_length.dtype is torch.int32
    assert topk_length.dim() == 1 and topk_length.size(0) == num_tokens

    if out is None:
        out = torch.empty((num_tokens, 1, num_heads, d_v), dtype=q.dtype, device=q.device)
    else:
        assert out.is_contiguous() and out.dtype == q.dtype
        assert out.shape == (num_tokens, 1, num_heads, d_v)

    grid = lambda meta: (
        triton.cdiv(num_heads, meta["BLOCK_H"]),
        num_tokens,
    )
    gqa_flash_mla_sparse_fa_gluon_kernel[grid](
        q,
        kv_cache,
        indices,
        topk_length,
        out,
        sm_scale,
        TOPK=topk,
        PAGE_BLOCK_SIZE=page_size,
        H_Q=num_heads,
        D_V=head_dim,
        num_warps=_DEFAULT_FA_KERNEL_CONFIG.num_warps,
        **_DEFAULT_FA_KERNEL_CONFIG.kwargs,
    )
    return out


def gqa_flash_mla_sparse_with_kv_cache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    out: torch.Tensor | None = None) -> torch.Tensor:
    assert q.is_contiguous() and q.dtype in (torch.float16, torch.bfloat16)
    assert q.dim() == 4 and q.size(1) == 1 and q.size(3) == d_v
    num_tokens = q.size(0)
    num_heads = q.size(2)
    head_dim = q.size(3)
    assert d_v == head_dim
    assert kv_cache.is_contiguous() and kv_cache.dtype == q.dtype and kv_cache.dim() == 4
    assert kv_cache.size(2) == 1 and kv_cache.size(3) == head_dim
    page_size = kv_cache.size(1)
    assert indices.is_contiguous() and indices.dtype is torch.int32 and indices.dim() == 3
    assert indices.size(0) == num_tokens and indices.size(1) == 1
    topk = indices.size(2)
    assert topk == TOPK
    assert topk_length.is_contiguous() and topk_length.dtype is torch.int32
    assert topk_length.dim() == 1 and topk_length.size(0) == num_tokens

    if out is None:
        out = torch.empty((num_tokens, 1, num_heads, d_v), dtype=q.dtype, device=q.device)
    else:
        assert out.is_contiguous() and out.dtype == q.dtype
        assert out.shape == (num_tokens, 1, num_heads, d_v)

    if ENABLE_AUTOTUNE:
        min_group_size = min(config.kwargs["GROUP_SIZE"] for config in _FD_KERNEL_CONFIGS)
    else:
        min_group_size = _DEFAULT_FD_KERNEL_CONFIG.kwargs["GROUP_SIZE"]
    max_groups = triton.cdiv(topk, min_group_size)

    middle = torch.empty((max_groups, num_tokens, num_heads, head_dim), dtype=torch.float32, device=q.device)
    group_maxs = torch.empty((num_tokens, num_heads, max_groups), dtype=torch.float32, device=q.device)
    group_expsums = torch.empty((num_tokens, num_heads, max_groups), dtype=torch.float32, device=q.device)

    grid = lambda meta: (
        triton.cdiv(num_heads, meta["BLOCK_H"]),
        num_tokens,
        triton.cdiv(topk, meta["GROUP_SIZE"]),
    )
    gqa_flash_mla_sparse_fd_kernel[grid](
        q,
        kv_cache,
        indices,
        topk_length,
        middle,
        group_maxs,
        group_expsums,
        sm_scale,
        TOPK=topk,
        PAGE_BLOCK_SIZE=page_size,
        H_Q=num_heads,
        D_V=head_dim)

    group_size = _best_fd_param("GROUP_SIZE", _DEFAULT_FD_CONFIG["GROUP_SIZE"])
    ceil_groups = triton.cdiv(topk, group_size)
    rd_grid = lambda meta: (
        num_tokens,
        num_heads,
        triton.cdiv(head_dim, meta["BLOCK_V"]),
    )
    gqa_flash_mla_sparse_rd_kernel[rd_grid](
        middle,
        group_maxs,
        group_expsums,
        out,
        topk_length,
        H_Q=num_heads,
        D_V=head_dim,
        CEIL_GROUPS=ceil_groups,
        GROUP_SIZE=group_size)
    return out


def gqa_flash_mla_sparse_with_kv_cache_fa(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    out: torch.Tensor | None = None) -> torch.Tensor:
    assert q.is_contiguous() and q.dtype in (torch.float16, torch.bfloat16)
    assert q.dim() == 4 and q.size(1) == 1 and q.size(3) == d_v
    num_tokens = q.size(0)
    num_heads = q.size(2)
    head_dim = q.size(3)
    assert d_v == head_dim
    assert kv_cache.is_contiguous() and kv_cache.dtype == q.dtype and kv_cache.dim() == 4
    assert kv_cache.size(2) == 1 and kv_cache.size(3) == head_dim
    page_size = kv_cache.size(1)
    assert indices.is_contiguous() and indices.dtype is torch.int32 and indices.dim() == 3
    assert indices.size(0) == num_tokens and indices.size(1) == 1
    topk = indices.size(2)
    assert topk == TOPK
    assert topk_length.is_contiguous() and topk_length.dtype is torch.int32
    assert topk_length.dim() == 1 and topk_length.size(0) == num_tokens

    if out is None:
        out = torch.empty((num_tokens, 1, num_heads, d_v), dtype=q.dtype, device=q.device)
    else:
        assert out.is_contiguous() and out.dtype == q.dtype
        assert out.shape == (num_tokens, 1, num_heads, d_v)

    grid = lambda meta: (
        triton.cdiv(num_heads, meta["BLOCK_H"]),
        num_tokens,
    )
    gqa_flash_mla_sparse_fa_kernel[grid](
        q,
        kv_cache,
        indices,
        topk_length,
        out,
        sm_scale,
        TOPK=topk,
        PAGE_BLOCK_SIZE=page_size,
        H_Q=num_heads,
        D_V=head_dim)
    return out


def _fill_indices(indices: torch.Tensor, valid_len: int, *, cache_tokens: int, seed_stride: int) -> None:
    max_start = max(1, cache_tokens - valid_len)
    for batch_idx in range(indices.shape[0]):
        start = (2176 + batch_idx * 8191) % max_start
        indices[batch_idx, 0, :valid_len] = torch.arange(
            start,
            start + valid_len,
            device=indices.device,
            dtype=torch.int32,
        )


def make_decode_inputs(
    *,
    valid_topk: int = TOPK,
    batch_size: int = 128,
    num_heads: int = 8,
    head_dim: int = 512,
    num_blocks: int = 17606,
    page_block_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    valid_topk = min(valid_topk, TOPK)
    sm_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn((batch_size, 1, num_heads, head_dim), device=device, dtype=dtype) / 10
    kv_cache = torch.randn(
        (num_blocks, page_block_size, 1, head_dim),
        device=device,
        dtype=dtype,
    ) / 10
    indices = torch.full((batch_size, 1, TOPK), -1, device=device, dtype=torch.int32)
    _fill_indices(
        indices,
        valid_topk,
        cache_tokens=num_blocks * page_block_size,
        seed_stride=4099,
    )
    topk_length = torch.full((batch_size,), valid_topk, device=device, dtype=torch.int32)
    return q, kv_cache, indices, sm_scale, topk_length


def _is_ampere_or_newer() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def _parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(" ", "").split(",") if item]


def _benchmark_kernel(fn, args=(), *, warmup=25, runs=100, use_cudagraph=False, **kwargs) -> float:
    run = (lambda: fn(*args, **kwargs)) if kwargs else (lambda: fn(*args))
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    if use_cudagraph:
        return triton.testing.do_bench_cudagraph(run, rep=runs, return_mode="median")
    return triton.testing.do_bench(run, warmup=warmup, rep=runs, return_mode="median")


def _profile_kernel(
    fn,
    tag: str,
    args=(),
    *,
    profile_dir: str,
    **kwargs,
) -> None:
    run = (lambda: fn(*args, **kwargs)) if kwargs else (lambda: fn(*args))
    # Compile/JIT warmup (not captured in the trace).
    run()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        run()
    torch.cuda.synchronize()

    os.makedirs(profile_dir, exist_ok=True)
    trace_path = os.path.join(profile_dir, f"gqa_mla_{tag}.json")
    prof.export_chrome_trace(trace_path)
    print(f"profile: impl={tag} trace={trace_path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark standard sparse GQA MLA (topk=2048, no attn_sink).")
    parser.add_argument("--batch-size", "--bs", default="128")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=17606)
    parser.add_argument("--page-block-size", type=int, default=64)
    parser.add_argument("--valid-topk", type=int, default=TOPK)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--no-cudagraph", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--impl",
        choices=("split", "fa", "gluon", "compare"),
        default="compare",
        help="compare=split+fa+gluon (default), split=fd+rd, fa=single-kernel, gluon=gluon FA kernel",
    )
    parser.add_argument("--check", action="store_true", default=None)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile each selected impl once (JIT warmup + 1 traced run); skip benchmark loop.",
    )
    parser.add_argument(
        "--profile-dir",
        default="profile_traces",
        help="Directory for Chrome trace JSON files (default: profile_traces).",
    )
    args = parser.parse_args()
    if args.no_check:
        args.check = False
    elif args.check is None:
        args.check = args.impl == "compare" and not args.profile

    batch_sizes = _parse_int_list(args.batch_size)
    if not batch_sizes:
        parser.error("--batch-size/--bs must contain at least one integer")

    torch.manual_seed(0)
    for batch_size in batch_sizes:
        q, kv_cache, indices, sm_scale, topk_length = make_decode_inputs(
            valid_topk=args.valid_topk,
            batch_size=batch_size,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            page_block_size=args.page_block_size,
            dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16,
        )
        valid_len = int(topk_length[0].item())
        print(
            f"topk={TOPK} valid_topk={valid_len} bs={batch_size} "
            f"h={args.num_heads} page={kv_cache.shape[1]}"
        )

        common_args = (q, kv_cache, indices, topk_length, sm_scale)
        run_gluon = args.impl in ("gluon", "compare") and _is_ampere_or_newer()
        if args.impl in ("gluon", "compare") and not run_gluon:
            print("gluon: skipped (requires Ampere or newer, sm80+)")

        run_check = args.check and not args.profile
        if args.impl in ("compare", "fa") or (run_check and args.impl in ("split", "gluon")):
            fa_out = gqa_flash_mla_sparse_with_kv_cache_fa(*common_args)
            torch.cuda.synchronize()
        if run_check or args.impl in ("compare", "split"):
            split_out = gqa_flash_mla_sparse_with_kv_cache(*common_args)
            torch.cuda.synchronize()
        if run_gluon and (run_check or args.impl == "gluon"):
            gluon_out = gqa_flash_mla_sparse_with_kv_cache_fa_gluon(*common_args)
            torch.cuda.synchronize()
        if run_check:
            if args.impl in ("compare", "split"):
                torch.testing.assert_close(split_out, fa_out, atol=1e-2, rtol=1e-2)
                print("check: split vs fa passed")
            if run_gluon and args.impl in ("compare", "gluon"):
                torch.testing.assert_close(gluon_out, fa_out, atol=1e-2, rtol=1e-2)
                print("check: gluon vs fa passed")

        impls: list[tuple[str, object]] = []
        if args.impl in ("split", "compare"):
            impls.append(("split", gqa_flash_mla_sparse_with_kv_cache))
        if args.impl in ("fa", "compare"):
            impls.append(("fa", gqa_flash_mla_sparse_with_kv_cache_fa))
        if run_gluon and args.impl in ("gluon", "compare"):
            impls.append(("gluon", gqa_flash_mla_sparse_with_kv_cache_fa_gluon))

        if args.profile:
            profile_dir = os.path.join(
                args.profile_dir,
                f"bs{batch_size}_topk{valid_len}",
            )
            print(f"profile: running each impl once -> {profile_dir}")
            for tag, fn in impls:
                _profile_kernel(fn, tag, args=common_args, profile_dir=profile_dir)
        else:
            for tag, fn in impls:
                ms = _benchmark_kernel(
                    fn,
                    args=common_args,
                    warmup=args.warmup,
                    runs=args.rep,
                    use_cudagraph=not args.no_cudagraph,
                )
                print(f"valid_topk={valid_len:4d} impl={tag} latency={ms:.4f} ms")
            print("kernel_config:")
            _print_autotune_configs(include_fa=args.impl in ("fa", "compare"))


if __name__ == "__main__":
    main()
