'''
该文件主要用于测试 Triton 的 Gluon 语法，包括：
基本使用方式和 triton.jit triton.language 一致
'''

import pytest
import triton
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import os


torch.manual_seed(0)

@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)
    
def copy_scalar(input, output):
    grid = (1,)
    copy_scalar_kernel[grid](input, output)

@pytest.mark.skip(reason="Not implemented")
def test_copy_scalar():
    input = torch.tensor([42.0], device="cuda")
    output = torch.empty_like(input)
    copy_scalar(input, output)
    assert torch.allclose(input, output)


@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, numl, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * BLOCK
    end = min(start + BLOCK, numl)
    for i in range(start, end):
        gl.store(out_ptr + i, gl.load(in_ptr + i))
        
def memcpy(input, output, BLOCK):
    numl = input.numel()
    grid = (triton.cdiv(numl, BLOCK),)
    memcpy_kernel[grid](input, output, numl, BLOCK, num_warps=1)
    

# @pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize("BLOCK", [64, ])
@pytest.mark.parametrize("numel", [40, 500])
def test_memcpy(BLOCK, numel):
    input = torch.randn(numel, device="cuda")
    output = torch.empty_like(input)
    memcpy(input, output, BLOCK)
    assert torch.allclose(input, output)
    
    
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 2 ** i}, num_warps=1) for i in range(8, 14)
    ],
    key=["numel"],
)
@gluon.jit
def memcpy_kernel_autotune(in_ptr, out_ptr, numl, BLOCK: gl.constexpr):
    memcpy_kernel(in_ptr, out_ptr, numl, BLOCK)
    
def memcpy_autotune(input, output, ):
    numl = input.numel()
    def grid(META): return (triton.cdiv(numl, META["BLOCK"]),)
    memcpy_kernel_autotune[grid](input, output, numl,)
    
@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize("BLOCK", [64, ])
@pytest.mark.parametrize("numel", [40, 500])
def test_memcpy_autotune(BLOCK, numel):
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
    input = torch.randn(numel, device="cuda")
    output = torch.empty_like(input)
    memcpy_autotune(input, output)
    assert torch.allclose(input, output)
    

import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(c_ptr + offsets, a_vals + b_vals, mask=mask)


def vector_add(a, b, c, BLOCK=256):
    numel = a.numel()
    assert b.numel() == numel and c.numel() == numel
    grid = (triton.cdiv(numel, BLOCK),)
    vector_add_kernel[grid](a, b, c, numel, BLOCK=BLOCK, num_warps=triton.cdiv(BLOCK, 32))

@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize("n", [1024])
def test_vector_add(n):
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.empty_like(a)
    vector_add(a, b, c)
    torch.testing.assert_close(a + b, c, atol=0.0, rtol=0.0)