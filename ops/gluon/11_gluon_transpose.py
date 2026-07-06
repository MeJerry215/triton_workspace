import os
import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

DEFAULT_IBLOCK = 128
DEFAULT_JBLOCK = 128
DEFAULT_NUM_WARPS = 4
DEFAULT_JPIPE = 32
DEFAULT_NUM_BUFFERS = 2


def is_ampere_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def get_smem_layout():
    return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])


def get_layout_for_gmem_access(tensor, num_warps):
    assert len(tensor.shape) == 2, "only 2D tensors are supported"
    assert tensor.stride(1) == 1, "only row-major tensors are supported"
    return gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])


def get_layout_for_transpose_store(num_warps):
    # Mirror of row-major load layout: transpose tile register mapping (#blocked1).
    return gl.BlockedLayout([1, 1], [32, 1], [num_warps, 1], [0, 1])


@gluon.jit
def transpose_kernel(in_ptr, out_ptr,  #
                     M, N, stride_in_m, stride_in_n, stride_out_n, stride_out_m,  #
                     layout_in: gl.constexpr, layout_out: gl.constexpr,  #
                     IBLOCK: gl.constexpr, JBLOCK: gl.constexpr):
    pid_i = gl.program_id(0)
    pid_j = gl.program_id(1)

    start_i = pid_i * IBLOCK
    start_j = pid_j * JBLOCK

    # Input tile: in[start_i:start_i+IBLOCK, start_j:start_j+JBLOCK]
    indices_i = start_i + gl.arange(0, IBLOCK, layout=gl.SliceLayout(dim=1, parent=layout_in))
    indices_j = start_j + gl.arange(0, JBLOCK, layout=gl.SliceLayout(dim=0, parent=layout_in))
    mask_in = (indices_i[:, None] < M) & (indices_j[None, :] < N)
    in_offsets = stride_in_m * indices_i[:, None] + stride_in_n * indices_j[None, :]

    value = gl.load(in_ptr + in_offsets, mask=mask_in)
    value = gl.permute(value, (1, 0))

    # Output tile: out[start_j:start_j+JBLOCK, start_i:start_i+IBLOCK]
    out_rows = start_j + gl.arange(0, JBLOCK, layout=gl.SliceLayout(dim=1, parent=layout_out))
    out_cols = start_i + gl.arange(0, IBLOCK, layout=gl.SliceLayout(dim=0, parent=layout_out))
    mask_out = (out_rows[:, None] < N) & (out_cols[None, :] < M)
    out_offsets = stride_out_n * out_rows[:, None] + stride_out_m * out_cols[None, :]

    value = gl.convert_layout(value, layout_out)
    gl.store(out_ptr + out_offsets, value, mask=mask_out)


@gluon.jit
def transpose_index_kernel(in_ptr, out_ptr,  #
                           M, N, stride_in_m, stride_in_n, stride_out_n, stride_out_m,  #
                           layout_in: gl.constexpr, layout_store: gl.constexpr,  #
                           IBLOCK: gl.constexpr, JBLOCK: gl.constexpr):
    pid_i = gl.program_id(0)
    pid_j = gl.program_id(1)

    start_i = pid_i * IBLOCK
    start_j = pid_j * JBLOCK

    # Input tile: in[start_i:start_i+IBLOCK, start_j:start_j+JBLOCK]
    indices_i = start_i + gl.arange(0, IBLOCK, layout=gl.SliceLayout(dim=1, parent=layout_in))
    indices_j = start_j + gl.arange(0, JBLOCK, layout=gl.SliceLayout(dim=0, parent=layout_in))
    mask_in = (indices_i[:, None] < M) & (indices_j[None, :] < N)
    in_offsets = stride_in_m * indices_i[:, None] + stride_in_n * indices_j[None, :]
    value = gl.load(in_ptr + in_offsets, mask=mask_in)

    # Output tile: out[j, i] with coalesced store under layout_store (#blocked1).
    # gl.BlockedLayout([1, 1], [32, 1], [num_warps, 1], [0, 1])
    # BlockedLayout([1], [1], [1], [1])
    out_rows = start_j + gl.arange(0, JBLOCK, layout=gl.SliceLayout(dim=0, parent=layout_store))
    #  gl.BlockedLayout([1], [32], [num_warps], [0])
    out_cols = start_i + gl.arange(0, IBLOCK, layout=gl.SliceLayout(dim=1, parent=layout_store))
    mask_out = (out_rows[None, :] < N) & (out_cols[:, None] < M)
    out_offsets = stride_out_n * out_rows[None, :] + stride_out_m * out_cols[:, None]
    value = gl.convert_layout(value, layout_store)
    gl.store(out_ptr + out_offsets, value, mask=mask_out)


def transpose(input, output, IBLOCK=DEFAULT_IBLOCK, JBLOCK=DEFAULT_JBLOCK, num_warps=DEFAULT_NUM_WARPS):
    assert len(input.shape) == 2 and len(output.shape) == 2
    M, N = input.shape
    assert output.shape == (N, M)

    layout_in = get_layout_for_gmem_access(input, num_warps)
    layout_out = get_layout_for_gmem_access(output, num_warps)
    grid = (triton.cdiv(M, IBLOCK), triton.cdiv(N, JBLOCK))
    return transpose_kernel[grid](  #
        input, output,  #
        M, N,  #
        input.stride(0), input.stride(1),  #
        output.stride(0), output.stride(1),  #
        layout_in, layout_out,  #
        IBLOCK, JBLOCK, num_warps=num_warps)


@gluon.jit
def transpose_cpasync_kernel(in_ptr, out_ptr,  #
                             M, N, stride_in_m, stride_in_n, stride_out_n, stride_out_m,  #
                             layout_in: gl.constexpr, layout_store: gl.constexpr,  #
                             smem_layout: gl.constexpr,  #
                             IBLOCK: gl.constexpr, JBLOCK: gl.constexpr):
    pid_i = gl.program_id(0)
    pid_j = gl.program_id(1)

    start_i = pid_i * IBLOCK
    start_j = pid_j * JBLOCK

    indices_i = start_i + gl.arange(0, IBLOCK, layout=gl.SliceLayout(dim=1, parent=layout_in))
    indices_j = start_j + gl.arange(0, JBLOCK, layout=gl.SliceLayout(dim=0, parent=layout_in))
    mask_in = (indices_i[:, None] < M) & (indices_j[None, :] < N)
    in_offsets = stride_in_m * indices_i[:, None] + stride_in_n * indices_j[None, :]

    dtype: gl.constexpr = in_ptr.dtype.element_ty
    smem = gl.allocate_shared_memory(dtype, [IBLOCK, JBLOCK], layout=smem_layout)
    cp.async_copy_global_to_shared(smem, in_ptr + in_offsets, mask=mask_in)
    cp.commit_group()
    cp.wait_group(0)

    value = smem.load(layout_in)

    out_rows = start_j + gl.arange(0, JBLOCK, layout=gl.SliceLayout(dim=0, parent=layout_store))
    out_cols = start_i + gl.arange(0, IBLOCK, layout=gl.SliceLayout(dim=1, parent=layout_store))
    mask_out = (out_rows[None, :] < N) & (out_cols[:, None] < M)
    out_offsets = stride_out_n * out_rows[None, :] + stride_out_m * out_cols[:, None]
    value = gl.convert_layout(value, layout_store)
    gl.store(out_ptr + out_offsets, value, mask=mask_out)


@gluon.jit
def issue_transpose_loads(copy_idx, smem, in_ptr, stride_in_m, stride_in_n, indices_i, start_j, j_idx, M, N,
                          layout_in: gl.constexpr, JPIPE: gl.constexpr, num_buffers: gl.constexpr):
    joffs = start_j + copy_idx * JPIPE + j_idx
    mask = (indices_i[:, None] < M) & (joffs[None, :] < N)
    in_offsets = stride_in_m * indices_i[:, None] + stride_in_n * joffs[None, :]
    cp.async_copy_global_to_shared(smem.index(copy_idx % num_buffers), in_ptr + in_offsets, mask=mask)
    cp.commit_group()
    return copy_idx + 1


@gluon.jit
def perform_transpose_store(read_idx, smem, out_ptr, stride_out_n, stride_out_m, start_i, start_j, M, N,
                            layout_in: gl.constexpr, layout_store: gl.constexpr, IBLOCK: gl.constexpr,
                            JPIPE: gl.constexpr, num_buffers: gl.constexpr):
    value = smem.index(read_idx % num_buffers).load(layout_in)
    out_rows = start_j + read_idx * JPIPE + gl.arange(0, JPIPE, layout=gl.SliceLayout(dim=0, parent=layout_store))
    out_cols = start_i + gl.arange(0, IBLOCK, layout=gl.SliceLayout(dim=1, parent=layout_store))
    mask_out = (out_rows[None, :] < N) & (out_cols[:, None] < M)
    out_offsets = stride_out_n * out_rows[None, :] + stride_out_m * out_cols[:, None]
    value = gl.convert_layout(value, layout_store)
    gl.store(out_ptr + out_offsets, value, mask=mask_out)
    return read_idx + 1


@gluon.jit
def transpose_cpasync_pipelined_kernel(in_ptr, out_ptr,  #
                                       M, N, stride_in_m, stride_in_n, stride_out_n, stride_out_m,  #
                                       layout_in: gl.constexpr, layout_store: gl.constexpr,  #
                                       smem_layout: gl.constexpr,  #
                                       IBLOCK: gl.constexpr, JBLOCK: gl.constexpr, JPIPE: gl.constexpr,
                                       num_buffers: gl.constexpr):
    pid_i = gl.program_id(0)
    pid_j = gl.program_id(1)

    start_i = pid_i * IBLOCK
    start_j = pid_j * JBLOCK

    indices_i = start_i + gl.arange(0, IBLOCK, layout=gl.SliceLayout(dim=1, parent=layout_in))
    j_idx = gl.arange(0, JPIPE, layout=gl.SliceLayout(dim=0, parent=layout_in))

    dtype: gl.constexpr = in_ptr.dtype.element_ty
    smem = gl.allocate_shared_memory(dtype, [num_buffers, IBLOCK, JPIPE], layout=smem_layout)
    copy_idx = 0
    read_idx = 0
    num_strips = gl.cdiv(JBLOCK, JPIPE)

    for _ in gl.static_range(num_buffers - 1):
        copy_idx = issue_transpose_loads(copy_idx, smem, in_ptr, stride_in_m, stride_in_n, indices_i, start_j, j_idx,
                                         M, N, layout_in, JPIPE, num_buffers)

    for _ in range(num_strips - (num_buffers - 1)):
        copy_idx = issue_transpose_loads(copy_idx, smem, in_ptr, stride_in_m, stride_in_n, indices_i, start_j, j_idx,
                                         M, N, layout_in, JPIPE, num_buffers)
        cp.wait_group(num_buffers - 1)
        read_idx = perform_transpose_store(read_idx, smem, out_ptr, stride_out_n, stride_out_m, start_i, start_j, M, N,
                                           layout_in, layout_store, IBLOCK, JPIPE, num_buffers)

    for i in gl.static_range(num_buffers - 1):
        cp.wait_group(num_buffers - 2 - i)
        read_idx = perform_transpose_store(read_idx, smem, out_ptr, stride_out_n, stride_out_m, start_i, start_j, M, N,
                                           layout_in, layout_store, IBLOCK, JPIPE, num_buffers)


def transpose_cpasync_pipelined(input, output, IBLOCK=DEFAULT_IBLOCK, JBLOCK=DEFAULT_JBLOCK, JPIPE=DEFAULT_JPIPE,
                                num_buffers=DEFAULT_NUM_BUFFERS, num_warps=DEFAULT_NUM_WARPS):
    assert len(input.shape) == 2 and len(output.shape) == 2
    M, N = input.shape
    assert output.shape == (N, M)
    assert JPIPE <= JBLOCK and JBLOCK % JPIPE == 0

    layout_in = get_layout_for_gmem_access(input, num_warps)
    layout_store = get_layout_for_transpose_store(num_warps)
    smem_layout = get_smem_layout()
    grid = (triton.cdiv(M, IBLOCK), triton.cdiv(N, JBLOCK))
    return transpose_cpasync_pipelined_kernel[grid](  #
        input, output,  #
        M, N,  #
        input.stride(0), input.stride(1),  #
        output.stride(0), output.stride(1),  #
        layout_in, layout_store, smem_layout,  #
        IBLOCK, JBLOCK, JPIPE, num_buffers, num_warps=num_warps)


def transpose_cpasync(input, output, IBLOCK=DEFAULT_IBLOCK, JBLOCK=DEFAULT_JBLOCK, num_warps=DEFAULT_NUM_WARPS):
    assert len(input.shape) == 2 and len(output.shape) == 2
    M, N = input.shape
    assert output.shape == (N, M)

    layout_in = get_layout_for_gmem_access(input, num_warps)
    layout_store = get_layout_for_transpose_store(num_warps)
    smem_layout = get_smem_layout()
    grid = (triton.cdiv(M, IBLOCK), triton.cdiv(N, JBLOCK))
    return transpose_cpasync_kernel[grid](  #
        input, output,  #
        M, N,  #
        input.stride(0), input.stride(1),  #
        output.stride(0), output.stride(1),  #
        layout_in, layout_store, smem_layout,  #
        IBLOCK, JBLOCK, num_warps=num_warps)


def transpose_index(input, output, IBLOCK=DEFAULT_IBLOCK, JBLOCK=DEFAULT_JBLOCK, num_warps=DEFAULT_NUM_WARPS):
    assert len(input.shape) == 2 and len(output.shape) == 2
    M, N = input.shape
    assert output.shape == (N, M)

    layout_in = get_layout_for_gmem_access(input, num_warps)
    grid = (triton.cdiv(M, IBLOCK), triton.cdiv(N, JBLOCK))
    layout_store = get_layout_for_transpose_store(num_warps)
    return transpose_index_kernel[grid](  #
        input, output,  #
        M, N,  #
        input.stride(0), input.stride(1),  #
        output.stride(0), output.stride(1),  #
        layout_in, layout_store,  #
        IBLOCK, JBLOCK, num_warps=num_warps)


@triton.jit
def triton_transpose_kernel(in_ptr, out_ptr,  #
                            M, N, stride_in_m, stride_in_n, stride_out_n, stride_out_m,  #
                            IBLOCK: tl.constexpr, JBLOCK: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    start_i = pid_i * IBLOCK
    start_j = pid_j * JBLOCK

    indices_i = start_i + tl.arange(0, IBLOCK)
    indices_j = start_j + tl.arange(0, JBLOCK)
    mask_in = (indices_i[:, None] < M) & (indices_j[None, :] < N)
    in_offsets = stride_in_m * indices_i[:, None] + stride_in_n * indices_j[None, :]

    value = tl.load(in_ptr + in_offsets, mask=mask_in)
    value = tl.permute(value, (1, 0))

    out_rows = start_j + tl.arange(0, JBLOCK)
    out_cols = start_i + tl.arange(0, IBLOCK)
    mask_out = (out_rows[:, None] < N) & (out_cols[None, :] < M)
    out_offsets = stride_out_n * out_rows[:, None] + stride_out_m * out_cols[None, :]

    tl.store(out_ptr + out_offsets, value, mask=mask_out)


def triton_transpose(input, output, IBLOCK=DEFAULT_IBLOCK, JBLOCK=DEFAULT_JBLOCK, num_warps=DEFAULT_NUM_WARPS):
    assert len(input.shape) == 2 and len(output.shape) == 2
    M, N = input.shape
    assert output.shape == (N, M)

    grid = (triton.cdiv(M, IBLOCK), triton.cdiv(N, JBLOCK))
    return triton_transpose_kernel[grid](  #
        input, output,  #
        M, N,  #
        input.stride(0), input.stride(1),  #
        output.stride(0), output.stride(1),  #
        IBLOCK, JBLOCK, num_warps=num_warps)


@triton.jit
def triton_transpose_index_kernel(in_ptr, out_ptr,  #
                                  M, N, stride_in_m, stride_in_n, stride_out_n, stride_out_m,  #
                                  IBLOCK: tl.constexpr, JBLOCK: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    start_i = pid_i * IBLOCK
    start_j = pid_j * JBLOCK

    # Logical tile coords (a, b): load input[i, j], store output[j, i].
    indices_i = start_i + tl.arange(0, IBLOCK)
    indices_j = start_j + tl.arange(0, JBLOCK)

    mask = (indices_i[:, None] < M) & (indices_j[None, :] < N)
    in_offsets = stride_in_m * indices_i[:, None] + stride_in_n * indices_j[None, :]
    value = tl.load(in_ptr + in_offsets, mask=mask)

    # Swap row/col indices for output without tl.trans / tl.permute.
    out_offsets = stride_out_n * indices_j[None, :] + stride_out_m * indices_i[:, None]
    tl.store(out_ptr + out_offsets, value, mask=mask)


def triton_transpose_index(input, output, IBLOCK=DEFAULT_IBLOCK, JBLOCK=DEFAULT_JBLOCK, num_warps=DEFAULT_NUM_WARPS):
    assert len(input.shape) == 2 and len(output.shape) == 2
    M, N = input.shape
    assert output.shape == (N, M)

    grid = (triton.cdiv(M, IBLOCK), triton.cdiv(N, JBLOCK))
    return triton_transpose_index_kernel[grid](  #
        input, output,  #
        M, N,  #
        input.stride(0), input.stride(1),  #
        output.stride(0), output.stride(1),  #
        IBLOCK, JBLOCK, num_warps=num_warps)


@pytest.mark.parametrize("M, N", [(300, 400), (1000, 200), (127, 255)])
@pytest.mark.parametrize("IBLOCK, JBLOCK", [(128, 128), (64, 256)])
@pytest.mark.parametrize("num_warps", [4])
def test_gluon_transpose(M, N, IBLOCK, JBLOCK, num_warps):
    torch.manual_seed(0)
    input = torch.randn((M, N), device="cuda")
    output = torch.empty((N, M), device="cuda")
    transpose(input, output, IBLOCK, JBLOCK, num_warps=num_warps)
    torch.testing.assert_close(input.T, output, atol=0, rtol=0)


@pytest.mark.parametrize("M, N", [(300, 400), (1000, 200), (127, 255)])
@pytest.mark.parametrize("IBLOCK, JBLOCK", [(128, 128), (64, 256)])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_gluon_transpose_cpasync(M, N, IBLOCK, JBLOCK, num_warps):
    torch.manual_seed(0)
    input = torch.randn((M, N), device="cuda")
    output = torch.empty((N, M), device="cuda")
    transpose_cpasync(input, output, IBLOCK, JBLOCK, num_warps=num_warps)
    torch.testing.assert_close(input.T, output, atol=0, rtol=0)


@pytest.mark.parametrize("M, N", [(300, 400), (1000, 200), (127, 255)])
@pytest.mark.parametrize("IBLOCK, JBLOCK", [(128, 128), (64, 256)])
@pytest.mark.parametrize("JPIPE", [32])
@pytest.mark.parametrize("num_buffers", [2])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_gluon_transpose_cpasync_pipelined(M, N, IBLOCK, JBLOCK, JPIPE, num_buffers, num_warps):
    torch.manual_seed(0)
    input = torch.randn((M, N), device="cuda")
    output = torch.empty((N, M), device="cuda")
    transpose_cpasync_pipelined(input, output, IBLOCK, JBLOCK, JPIPE, num_buffers, num_warps=num_warps)
    torch.testing.assert_close(input.T, output, atol=0, rtol=0)


@pytest.mark.parametrize("M, N", [(300, 400), (1000, 200), (127, 255)])
@pytest.mark.parametrize("IBLOCK, JBLOCK", [(128, 128), (64, 256)])
@pytest.mark.parametrize("num_warps", [4])
def test_gluon_transpose_index(M, N, IBLOCK, JBLOCK, num_warps):
    torch.manual_seed(0)
    input = torch.randn((M, N), device="cuda")
    output = torch.empty((N, M), device="cuda")
    transpose_index(input, output, IBLOCK, JBLOCK, num_warps=num_warps)
    torch.testing.assert_close(input.T, output, atol=0, rtol=0)


@pytest.mark.parametrize("M, N", [(300, 400), (1000, 200), (127, 255)])
@pytest.mark.parametrize("IBLOCK, JBLOCK", [(128, 128), (64, 256)])
@pytest.mark.parametrize("num_warps", [4])
def test_triton_transpose(M, N, IBLOCK, JBLOCK, num_warps):
    torch.manual_seed(0)
    input = torch.randn((M, N), device="cuda")
    output = torch.empty((N, M), device="cuda")
    triton_transpose(input, output, IBLOCK, JBLOCK, num_warps=num_warps)
    torch.testing.assert_close(input.T, output, atol=0, rtol=0)


@pytest.mark.parametrize("M, N", [(300, 400), (1000, 200), (127, 255)])
@pytest.mark.parametrize("IBLOCK, JBLOCK", [(128, 128), (64, 256)])
@pytest.mark.parametrize("num_warps", [4])
def test_triton_transpose_index(M, N, IBLOCK, JBLOCK, num_warps):
    torch.manual_seed(0)
    input = torch.randn((M, N), device="cuda")
    output = torch.empty((N, M), device="cuda")
    triton_transpose_index(input, output, IBLOCK, JBLOCK, num_warps=num_warps)
    torch.testing.assert_close(input.T, output, atol=0, rtol=0)


def get_throughput(tensor, ms):
    tbytes = (tensor.numel() * tensor.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)


def _ttgir_blocked_layouts(ttgir):
    return [line.strip() for line in ttgir.splitlines() if line.startswith("#blocked")]


def _ptx_global_memops(ptx):
    ld = sum(1 for line in ptx.splitlines() if "ld.global" in line)
    st = sum(1 for line in ptx.splitlines() if "st.global" in line)
    vec_ld = sum(1 for line in ptx.splitlines() if "ld.global.v" in line)
    return ld, st, vec_ld


def print_kernel_info(label, kernel):
    md = kernel.metadata
    print(f"\n--- {label} ---")
    print(f"num_warps={md.num_warps}  shared={md.shared} bytes")
    kernel._init_handles()
    print(f"n_regs={kernel.n_regs}  n_spills={kernel.n_spills}")
    ttgir = kernel.asm.get("ttgir", "")
    if ttgir:
        layouts = _ttgir_blocked_layouts(ttgir)
        for layout in layouts:
            print(f"  {layout}")
        trans = sum(1 for line in ttgir.splitlines() if "tt.trans" in line)
        convert_layout = sum(1 for line in ttgir.splitlines() if "convert_layout" in line)
        print(f"  tt.trans ops={trans}  convert_layout ops={convert_layout}")
    ptx = kernel.asm.get("ptx", "")
    if ptx:
        ld, st, vec_ld = _ptx_global_memops(ptx)
        print(f"  ptx ld.global={ld}  st.global={st}  vectorized_ld={vec_ld}")


def verify_transpose_outputs(input, outputs):
    ref = input.T
    all_ok = True
    for label, out in outputs.items():
        try:
            torch.testing.assert_close(ref, out, atol=0, rtol=0)
            print(f"  {label}: PASS")
        except AssertionError as e:
            all_ok = False
            max_diff = (ref - out).abs().max().item()
            print(f"  {label}: FAIL  max_diff={max_diff}")
    if not all_ok:
        raise AssertionError("transpose correctness check failed")
    return ref


def bench_transpose_pair(input, output, IBLOCK, JBLOCK, num_warps):
    out_gluon = torch.empty_like(output)
    out_gluon_index = torch.empty_like(output)
    out_gluon_cpasync = torch.empty_like(output)
    out_gluon_cpasync_pipelined = torch.empty_like(output)
    out_triton = torch.empty_like(output)
    out_triton_index = torch.empty_like(output)

    gluon_kernel = transpose(input, out_gluon, IBLOCK, JBLOCK, num_warps=num_warps)
    gluon_index_kernel = transpose_index(input, out_gluon_index, IBLOCK, JBLOCK, num_warps=num_warps)
    triton_kernel = triton_transpose(input, out_triton, IBLOCK, JBLOCK, num_warps=num_warps)
    triton_index_kernel = triton_transpose_index(input, out_triton_index, IBLOCK, JBLOCK, num_warps=num_warps)

    outputs = {
        "gluon (permute)": out_gluon,
        "gluon (index)": out_gluon_index,
        "triton (permute)": out_triton,
        "triton (index)": out_triton_index,
    }
    gluon_cpasync_kernel = None
    gluon_cpasync_pipelined_kernel = None
    if is_ampere_or_newer():
        gluon_cpasync_kernel = transpose_cpasync(input, out_gluon_cpasync, IBLOCK, JBLOCK, num_warps=num_warps)
        outputs["gluon (cpasync)"] = out_gluon_cpasync
        if JBLOCK % DEFAULT_JPIPE == 0:
            gluon_cpasync_pipelined_kernel = transpose_cpasync_pipelined(
                input, out_gluon_cpasync_pipelined, IBLOCK, JBLOCK, DEFAULT_JPIPE, DEFAULT_NUM_BUFFERS,
                num_warps=num_warps)
            outputs["gluon (cpasync pipelined)"] = out_gluon_cpasync_pipelined
    print("Correctness check (vs input.T):")
    verify_transpose_outputs(input, outputs)

    ms_gluon = triton.testing.do_bench(lambda: transpose(input, out_gluon, IBLOCK, JBLOCK, num_warps=num_warps),
                                       warmup=25, rep=200)
    ms_gluon_index = triton.testing.do_bench(
        lambda: transpose_index(input, out_gluon_index, IBLOCK, JBLOCK, num_warps=num_warps), warmup=25, rep=200)
    ms_triton = triton.testing.do_bench(
        lambda: triton_transpose(input, out_triton, IBLOCK, JBLOCK, num_warps=num_warps), warmup=25, rep=200)
    ms_triton_index = triton.testing.do_bench(
        lambda: triton_transpose_index(input, out_triton_index, IBLOCK, JBLOCK, num_warps=num_warps),
        warmup=25, rep=200)

    result = {
        "gluon_kernel": gluon_kernel,
        "gluon_index_kernel": gluon_index_kernel,
        "triton_kernel": triton_kernel,
        "triton_index_kernel": triton_index_kernel,
        "gluon_ms": ms_gluon,
        "gluon_index_ms": ms_gluon_index,
        "triton_ms": ms_triton,
        "triton_index_ms": ms_triton_index,
        "gluon_tbps": get_throughput(input, ms_gluon),
        "gluon_index_tbps": get_throughput(input, ms_gluon_index),
        "triton_tbps": get_throughput(input, ms_triton),
        "triton_index_tbps": get_throughput(input, ms_triton_index),
        "gluon_cpasync_kernel": gluon_cpasync_kernel,
        "gluon_cpasync_pipelined_kernel": gluon_cpasync_pipelined_kernel,
    }
    if gluon_cpasync_kernel is not None:
        ms_gluon_cpasync = triton.testing.do_bench(
            lambda: transpose_cpasync(input, out_gluon_cpasync, IBLOCK, JBLOCK, num_warps=num_warps),
            warmup=25, rep=200)
        result["gluon_cpasync_ms"] = ms_gluon_cpasync
        result["gluon_cpasync_tbps"] = get_throughput(input, ms_gluon_cpasync)
    if gluon_cpasync_pipelined_kernel is not None:
        ms_gluon_cpasync_pipelined = triton.testing.do_bench(
            lambda: transpose_cpasync_pipelined(input, out_gluon_cpasync_pipelined, IBLOCK, JBLOCK, DEFAULT_JPIPE,
                                                DEFAULT_NUM_BUFFERS, num_warps=num_warps),
            warmup=25, rep=200)
        result["gluon_cpasync_pipelined_ms"] = ms_gluon_cpasync_pipelined
        result["gluon_cpasync_pipelined_tbps"] = get_throughput(input, ms_gluon_cpasync_pipelined)
    return result


if __name__ == "__main__":
    print("Benchmarking transpose (equivalent config)")
    print("========================================")
    M, N = 32 * 1024, 32 * 1024
    input = torch.randn((M, N), device="cuda")
    output = torch.empty((N, M), device="cuda")

    IBLOCK, JBLOCK, num_warps = DEFAULT_IBLOCK, DEFAULT_JBLOCK, DEFAULT_NUM_WARPS
    print(f"IBLOCK={IBLOCK}  JBLOCK={JBLOCK}  num_warps={num_warps}")
    print(f"grid=({triton.cdiv(M, IBLOCK)}, {triton.cdiv(N, JBLOCK)})  tile={IBLOCK}x{JBLOCK}")

    result = bench_transpose_pair(input, output, IBLOCK, JBLOCK, num_warps)
    print()
    print_kernel_info("gluon (permute)", result["gluon_kernel"])
    print_kernel_info("gluon (index)", result["gluon_index_kernel"])
    if result["gluon_cpasync_kernel"] is not None:
        print_kernel_info("gluon (cpasync)", result["gluon_cpasync_kernel"])
    if result["gluon_cpasync_pipelined_kernel"] is not None:
        print_kernel_info("gluon (cpasync pipelined)", result["gluon_cpasync_pipelined_kernel"])
    print_kernel_info("triton (permute)", result["triton_kernel"])
    print_kernel_info("triton (index)", result["triton_index_kernel"])

    if os.environ.get("DUMP_IR"):
        dump_kernels = [
            ("gluon_permute", result["gluon_kernel"]),
            ("gluon_index", result["gluon_index_kernel"]),
            ("triton_permute", result["triton_kernel"]),
            ("triton_index", result["triton_index_kernel"]),
        ]
        if result["gluon_cpasync_kernel"] is not None:
            dump_kernels.insert(2, ("gluon_cpasync", result["gluon_cpasync_kernel"]))
        if result["gluon_cpasync_pipelined_kernel"] is not None:
            dump_kernels.insert(3, ("gluon_cpasync_pipelined", result["gluon_cpasync_pipelined_kernel"]))
        for label, kernel in dump_kernels:
            for ir_name in ("ttgir", "ptx"):
                ir = kernel.asm.get(ir_name, "")
                if ir:
                    path = f"/tmp/{label}_transpose.{ir_name}"
                    with open(path, "w") as f:
                        f.write(ir)
                    print(f"dumped {path}")

    print(f"\ngluon transpose (permute):  {result['gluon_tbps']:.3f} TB/s  ({result['gluon_ms']:.4f} ms)")
    print(f"gluon transpose (index):    {result['gluon_index_tbps']:.3f} TB/s  ({result['gluon_index_ms']:.4f} ms)")
    if result["gluon_cpasync_kernel"] is not None:
        print(
            f"gluon transpose (cpasync):  {result['gluon_cpasync_tbps']:.3f} TB/s  ({result['gluon_cpasync_ms']:.4f} ms)")
    if result["gluon_cpasync_pipelined_kernel"] is not None:
        print(f"gluon transpose (cpasync pipe): {result['gluon_cpasync_pipelined_tbps']:.3f} TB/s  "
              f"({result['gluon_cpasync_pipelined_ms']:.4f} ms)")
    print(f"triton transpose (permute): {result['triton_tbps']:.3f} TB/s  ({result['triton_ms']:.4f} ms)")
    print(f"triton transpose (index):   {result['triton_index_tbps']:.3f} TB/s  ({result['triton_index_ms']:.4f} ms)")

    ms = triton.testing.do_bench(lambda: output.copy_(input.T), warmup=25, rep=200)
    print(f"torch transpose:   {get_throughput(input, ms):.3f} TB/s  ({ms:.4f} ms)")

    print("""
Note on the comparison
----------------------
Tile config is identical: IBLOCK=JBLOCK=128, num_warps=4, grid=(256, 256).

(permute): load tile -> permute/trans -> convert_layout -> store.
(index):   load under layout_in (#blocked), convert_layout to layout_store
           (#blocked1), store with index-swapped offsets under layout_store.
           Must NOT convert to the same layout as load (that becomes a no-op).
(cpasync): cp.async load input tile to smem -> smem.load -> convert_layout
           -> index-swapped store (same as gluon index path, but async gmem load).
(cpasync pipelined): split tile along J into JPIPE strips; double-buffer smem and
           overlap cp.async load of strip k+1 with transpose+store of strip k.
           Requires Ampere+ and JBLOCK % JPIPE == 0.

Use DUMP_IR=1 to dump ttgir/ptx to /tmp for side-by-side inspection.
""")
