import pytest
import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

def _enabled(label):
    from sys import argv
    return len(argv) == 1 or label in argv[1].split(",")


@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK

    # The main difference between writing this kernel in Triton and Gluon is
    # we need to specify the layout of the 1D tensor. Layouts are propagated
    # forwards through type inference, so we only need to specify the layout for
    # the indices tensor.
    indices = gl.arange(0, XBLOCK, layout=layout)

    offsets = start + indices
    in_ptrs = in_ptr + offsets
    mask = offsets < xnumel

    value = gl.load(in_ptrs, mask=mask)
    out_ptrs = out_ptr + offsets
    gl.store(out_ptrs, value, mask=mask)
    
def memcpy_1d_impl(input, output, XBLOCK, layout, num_warps):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    compiled_kernel = memcpy_1d_kernel[grid](input, output, xnumel, XBLOCK, layout, num_warps=num_warps)
    return compiled_kernel


def get_throughput(input, ms):
    tbytes = (2 * input.numel() * input.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)


def bench_memcpy_impl(input, output, impl):
    compiled_kernel = impl(input, output)
    fn = lambda: impl(input, output)
    ms = triton.testing.do_bench(fn)
    return compiled_kernel, get_throughput(input, ms)


def bench_memcpy(impl):
    torch.manual_seed(0)
    xnumel = 2 << 30
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)

    return bench_memcpy_impl(input, output, impl)

@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize("XBLOCK", [128, 256])
@pytest.mark.parametrize("xnumel", [200, 1000])
@pytest.mark.parametrize("num_warps", [4])
def test_memcpy_1d(XBLOCK, xnumel, num_warps):
    torch.manual_seed(0)
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    layout = gl.BlockedLayout([1], [32], [num_warps], [0])
    memcpy_1d_impl(input, output, XBLOCK, layout, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)
    
    

# if __name__ == "__main__" and _enabled("R_vs_throughput"):
#     print("R vs. Throughput")
#     print("================")
#     XBLOCK = 2048
#     num_warps = 4
#     kernel = partial(memcpy_1d_impl, XBLOCK=XBLOCK, num_warps=num_warps)
#     compiled_kernels = []
#     for i in range(0, 5):
#         R = 2**i
#         layout = gl.BlockedLayout([R], [32], [num_warps], [0])
#         impl = partial(kernel, layout=layout)
#         compiled_kernel, throughput = bench_memcpy(impl)
#         compiled_kernels.append((R, compiled_kernel))
#         print(f"R={R:<3} {throughput:.3f} TB/s")
#     print()    
    
# if __name__ == "__main__" and _enabled("LDG_STG_instructions"):
#     print("LDG/STG instructions")
#     print("====================")
#     for R, compiled_kernel in compiled_kernels:
#         print(f"\nR={R}")
#         print("==========")
#         sass = compiled_kernel.asm["sass"]
#         for line in sass.split("\n"):
#             if "LDG.E" in line or "STG.E" in line:
#                 print(line)
#     print()
    
if __name__ == "__main__" and _enabled("XBLOCK_R_vs_throughput"):
    print("(XBLOCK, R) vs. Throughput")
    print("==========================")
    num_warps = 4

    print("XBLOCK   ", end=" ")
    for i in range(0, 5):
        print(f"R={2**i:<3}", end=" ")
    print()

    for j in range(10, 15):
        XBLOCK = 2**j
        print(f"{XBLOCK:<8}", end=" ")
        kernel = partial(memcpy_1d_impl, XBLOCK=XBLOCK, num_warps=num_warps)
        for i in range(0, 5):
            R = 2**i
            layout = gl.BlockedLayout([R], [32], [num_warps], [0])
            impl = partial(kernel, layout=layout)
            compiled_kernel, throughput = bench_memcpy(impl)
            print(f"{throughput:.3f}", end=" ")
        print()
    print()
    
    
'''
layout = gl.BlockedLayout(
    size_per_thread=[2, 4],
    threads_per_warp=[16, 2],
    warps_per_cta=[2, 2],
    order=[1, 0],
)



single warp layout:
[[ T0:0,  T0:1,  T0:2,  T0:3,  T1:0,  T1:1,  T1:2,  T1:3],
 [ T0:4,  T0:5,  T0:6,  T0:7,  T1:4,  T1:5,  T1:6,  T1:7],
 [ T2:0,  T2:1,  T2:2,  T2:3,  T3:0,  T3:1,  T3:2,  T3:3],
 [ T2:4,  T2:5,  T2:6,  T2:7,  T3:4,  T3:5,  T3:6,  T3:7],
 ...
 [T28:0, T28:1, T28:2, T28:3, T29:0, T29:1, T29:2, T29:3],
 [T28:4, T28:5, T28:6, T28:7, T29:4, T29:5, T29:6, T29:7],
 [T30:0, T30:1, T30:2, T30:3, T31:0, T31:1, T31:2, T31:3],
 [T30:4, T30:5, T30:6, T30:7, T31:4, T31:5, T31:6, T31:7]]


gl.SliceLayout(dim=1, parent=layout) 
slice layout
[  T0:0| T0:1| T0:2| T0:3| T1:0| T1:1| T1:2| T1:3,
   T0:4| T0:5| T0:6| T0:7| T1:4| T1:5| T1:6| T1:7,
   T2:0| T2:1| T2:2| T2:3| T3:0| T3:1| T3:2| T3:3,
   T2:4| T2:5| T2:6| T2:7| T3:4| T3:5| T3:6| T3:7,
 ...
  T28:0|T28:1|T28:2|T28:3|T29:0|T29:1|T29:2|T29:3,
  T28:4|T28:5|T28:6|T28:7|T29:4|T29:5|T29:6|T29:7,
  T30:0|T30:1|T30:2|T30:3|T31:0|T31:1|T31:2|T31:3,
  T30:4|T30:5|T30:6|T30:7|T31:4|T31:5|T31:6|T31:7]
  ↓
[  T0:0| T1:0,
   T0:1| T1:1,
   T2:0| T3:0,
   T2:1| T3:1,
 ...
  T28:0|T29:0,
  T28:1|T29:1,
  T30:0|T31:0,
  T30:1|T31:1]

''' 

@gluon.jit
def memcpy_2d_kernel(in_ptr, out_ptr,  #
                     xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                     layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK
    # For the 1D indices, use a SliceLayout along the dimensions we will expand.
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    # expand_dims along the slice dimension returns a tensor with the parent
    # layout, so this yields [XBLOCK, 1] and [1, YBLOCK] tensors with the same
    # layout which can be broadcasted together to [XBLOCK, YBLOCK].
    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    # Compute the mask the same way: select for indices along each dimension
    # that are in bounds and broadcast them together.
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask)
    gl.store(out_ptr + out_offsets, value, mask=mask)


def memcpy_2d_impl(input, output, XBLOCK, YBLOCK, layout, num_warps):
    xnumel, ynumel = input.shape
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    # Pass the strides of the input and output tensors into the kernel. The
    # compiler will specialize the kernel if any of the strides are 1, which is
    # common for the inner dimension of tensors.
    compiled_kernel = memcpy_2d_kernel[grid](  #
        input, output, xnumel, ynumel,  #
        *input.stride(), *output.stride(),  #
        layout, XBLOCK, YBLOCK, num_warps=num_warps)
    return compiled_kernel


@pytest.mark.parametrize("XBLOCK, YBLOCK", [(128, 256), (256, 128)])
@pytest.mark.parametrize("xnumel, ynumel", [(100, 2000), (1000, 200)])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("num_warps", [4])
def test_memcpy_2d(XBLOCK, YBLOCK, xnumel, ynumel, transposed, num_warps):
    torch.manual_seed(0)
    input = torch.randn((xnumel, ynumel), device="cuda")
    output = torch.empty_like(input)
    # Transposing the tensor makes it non-contiguous along the inner dimension.
    input = input.T if transposed else input
    output = output.T if transposed else output
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    memcpy_2d_impl(input, output, XBLOCK, YBLOCK, layout, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)
    
def bench_memcpy_2d(impl, transposed=False):
    # 8 GB tensor, but spread across 2 dimensions.
    xnumel = 32 * 1024
    ynumel = 64 * 1024
    input = torch.randn((xnumel, ynumel), device="cuda")
    output = torch.empty_like(input)
    input = input.T if transposed else input
    output = output.T if transposed else output
    return bench_memcpy_impl(input, output, impl)

if __name__ == "__main__" and _enabled("memcpy_2d_layout"):
    print("Benchmarking 2D memcpy")
    print("======================")
    XBLOCK = 1
    YBLOCK = 2048
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    impl = partial(memcpy_2d_impl, XBLOCK=XBLOCK, YBLOCK=YBLOCK, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_2d(impl)
    print(f"Throughput: {throughput:.3f} TB/s")
    _, throughput = bench_memcpy_2d(impl, transposed=True)
    print(f"Transposed throughput: {throughput:.3f} TB/s")
    
if __name__ == "__main__" and _enabled("memcpy_2d_contig"):
    print("Non-contiguous memcpy")
    print("=====================")
    # 8 GB tensor.
    xnumel = 32 * 1024
    ynumel = 64 * 1024
    input = torch.randn((xnumel, ynumel), device="cuda")
    # Take a view over every other row.
    input = input[::2]
    output = torch.empty_like(input)
    assert not input.is_contiguous() and output.is_contiguous()

    # Benchmark 2D memcpy.
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    impl = partial(memcpy_2d_impl, XBLOCK=1, YBLOCK=2048, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input, output, impl)
    print(f"2D memcpy: {throughput:.3f} TB/s")

    # Benchmark PyTorch contiguous.
    fn = lambda: input.contiguous()
    ms = triton.testing.do_bench(fn)
    throughput = get_throughput(input, ms)
    print(f"torch.Tensor.contiguous: {throughput:.3f} TB/s")

    # We can eke out even more performance by using the transposed "trick".
    layout = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    impl = partial(memcpy_2d_impl, XBLOCK=2048, YBLOCK=1, layout=layout, num_warps=4)
    # 64 * 1024, 32 * 1024
    # torch.Size([16384, 65536])  torch.Size([16384, 65536])
    _, throughput = bench_memcpy_impl(input.T, output.T, impl)
    print(f"2D memcpy (transposed): {throughput:.3f} TB/s")
    print()
    
if __name__ == "__main__" and _enabled("memcpy_2d_inout"):
    print("2D memcpy in/out layouts")
    print("=========================")

    # Input is contiguous along dim 1.
    input = torch.randn((32 * 1024, 32 * 1024), device="cuda")

    # Output is contiguous along dim 0.
    output = torch.empty((input.shape[1], input.shape[0]), device="cuda").T

    # order=[1, 0]
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    impl = partial(memcpy_2d_impl, XBLOCK=1, YBLOCK=2048, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input, output, impl)
    print(f"2D memcpy (order=[1, 0]): {throughput:.3f} TB/s")

    # order=[0, 1]
    layout = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    impl = partial(memcpy_2d_impl, XBLOCK=2048, YBLOCK=1, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input, output, impl)
    print(f"2D memcpy (order=[0, 1]): {throughput:.3f} TB/s")
    
def get_layout_for_gmem_access(tensor, num_warps):
    if len(tensor.shape) == 1:
        return gl.BlockedLayout([1], [32], [num_warps], [0])

    assert len(tensor.shape) == 2, "only 1D and 2D tensors are supported"
    assert 1 in tensor.stride(), "expected at least 1 contiguous dimension"
    if tensor.stride(1) == 1:
        return gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    else:
        return gl.BlockedLayout([1, 1], [32, 1], [num_warps, 1], [0, 1])
    
@gluon.jit
def get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride, ystride,  #
                         XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr):
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)
    offsets = xstride * indices_x[:, None] + ystride * indices_y[None, :]
    return mask, offsets


@gluon.jit
def memcpy_2d_inout_kernel(in_ptr, out_ptr,  #
                           xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                           layout_in: gl.constexpr, layout_out: gl.constexpr,  #
                           XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK

    # We need two sets of indices and masks for each layout. If the layouts
    # happen to be the same, the compiler will optimize away the extra code and
    # layout conversion.
    '''
    #blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
    #blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
    '''
    
    mask_in, in_offsets = get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride_in, ystride_in,  #
                                               XBLOCK, YBLOCK, layout_in)
    mask_out, out_offsets = get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride_out, ystride_out,  #
                                                 XBLOCK, YBLOCK, layout_out)

    value = gl.load(in_ptr + in_offsets, mask=mask_in)

    # Use `gl.convert_layout` to perform layout conversions.
    value = gl.convert_layout(value, layout_out)

    gl.store(out_ptr + out_offsets, value, mask=mask_out)


def memcpy_2d_inout(input, output, num_warps=4):
    assert input.shape == output.shape, "input and output must have the same shape"
    XBLOCK = 128
    YBLOCK = 128
    layout_in = get_layout_for_gmem_access(input, num_warps)
    layout_out = get_layout_for_gmem_access(output, num_warps)
    grid = (triton.cdiv(input.shape[0], XBLOCK), triton.cdiv(input.shape[1], YBLOCK))
    return memcpy_2d_inout_kernel[grid](  #
        input, output,  #
        input.shape[0], input.shape[1],  #
        *input.stride(), *output.stride(),  #
        layout_in, layout_out,  #
        XBLOCK, YBLOCK, num_warps=num_warps)


@pytest.mark.parametrize("xnumel, ynumel", [(300, 400)])
@pytest.mark.parametrize("transpose_in, transpose_out", [(True, False), (False, True)])
def test_memcpy_2d_inout(xnumel, ynumel, transpose_in, transpose_out):
    torch.manual_seed(0)
    if transpose_in:
        input = torch.randn((ynumel, xnumel), device="cuda").T
    else:
        input = torch.randn((xnumel, ynumel), device="cuda")
    if transpose_out:
        output = torch.empty((ynumel, xnumel), device="cuda").T
    else:
        output = torch.empty((xnumel, ynumel), device="cuda")
    memcpy_2d_inout(input, output)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


if __name__ == "__main__" and _enabled("memcpy_2d_inout"):
    _, throughput = bench_memcpy_impl(input, output, memcpy_2d_inout)
    print(f"2D memcpy (in/out layouts): {throughput:.3f} TB/s")