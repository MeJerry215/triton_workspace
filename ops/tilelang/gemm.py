"""
TileLang GEMM (通用矩阵乘法) Op

提供两个版本:
  1. non_persistent — 经典 2D grid kernel，适合理解 tilelang 基本流程
  2. persistent    — persistent kernel，利用 persistent loop 自动均衡 SM 负载

用法:
    python ops/tilelang/gemm.py              # 正确性验证 + 性能对比
    python ops/tilelang/gemm.py --bench      # 只跑性能
    python ops/tilelang/gemm.py --M 2048 --N 2048 --K 2048  # 指定规模
"""

import argparse
import torch
import tilelang
import tilelang.language as T

# ===================================================================
# Version 1: Non-Persistent GEMM
# 经典风格：grid = (ceil(M/BM), ceil(N/BN))，每个 block 处理一个 tile
# 适合学习 tilelang 的基本 dataflow：copy → gemm → copy
# ===================================================================

@tilelang.jit
def gemm_non_persistent(
    A, B,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    num_stages: int,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    # 2D grid: 在 M, N 维度分块
    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.use_swizzle(10)  # L2 cache 优化

        T.clear(C_local)

        # Pipelined 循环：软件流水线隐藏访存延迟
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            # 从 global memory → shared memory
            T.copy(A[bx * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, by * block_N], B_shared)
            # tile-level GEMM: shared → register (调用 cute/hip 后端)
            T.gemm(A_shared, B_shared, C_local)

        # 直接从 register 写回 global memory
        T.copy(C_local, C[bx * block_M, by * block_N])

    return C


# ===================================================================
# Version 2: Persistent GEMM
# 使用 T.Persistent 原语在 SM 之间自动分配 tile，适合负载不均衡场景
# ===================================================================

@tilelang.jit
def gemm_persistent(
    A, B,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    num_stages: int,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    from tilelang.carver.arch import driver

    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    sm_num = driver.get_num_sms()                   # 查询 SM 数量
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)

    with T.Kernel(sm_num, threads=threads) as (block_id):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # Persistent loop: 自动在 sm_num 个 block 间均衡分配 tile
        for bx, by in T.Persistent(
            [m_blocks, n_blocks],
            sm_num,
            block_id,
        ):
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # 直接从 register 写回 global memory
            T.copy(C_local, C[bx * block_M, by * block_N])

    return C


# ===================================================================
# Reference
# ===================================================================

def ref_program(A, B):
    return A @ B


# ===================================================================
# Main
# ===================================================================

def main(M=4096, N=4096, K=4096, bench=False):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    threads = 256
    num_stages = 3

    total_flops = 2 * M * N * K
    print(f"GEMM: M={M}, N={N}, K={K}  ({total_flops/1e12:.2f} TFlops)")
    print(f"Block: M={BLOCK_M}, N={BLOCK_N}, K={BLOCK_K}, threads={threads}, stages={num_stages}")
    print(dash := "-" * 60)

    # ---- 编译两个 kernel ----
    print("Compiling kernels ...")
    np_kernel = gemm_non_persistent.compile(
        M=M, N=N, K=K,
        block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
        threads=threads, num_stages=num_stages,
    )
    p_kernel = gemm_persistent.compile(
        M=M, N=N, K=K,
        block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
        threads=threads, num_stages=num_stages,
    )
    print("  Done")

    # 也可以用 get_profiler 获取自动生成的 input tensors
    np_profiler = np_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    p_profiler = p_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

    # ---- 正确性验证 ----
    if not bench:
        print("Validating correctness (non-persistent)...")
        np_profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
        print("  [PASS]")

        print("Validating correctness (persistent)...")
        p_profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
        print("  [PASS]")

    # ---- 性能对比 ----
    print("Benchmarking ...")
    np_ms = np_profiler.do_bench(warmup=500)
    p_ms = p_profiler.do_bench(warmup=500)

    print(dash)
    print(f"  Non-Persistent GEMM  |  {np_ms:.4f} ms  |  {total_flops / np_ms * 1e-9:.2f} TFlops")
    print(f"  Persistent    GEMM  |  {p_ms:.4f} ms  |  {total_flops / p_ms * 1e-9:.2f} TFlops")
    print(f"  Speedup (P/NP): {np_ms / p_ms:.2f}x")
    print(dash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TileLang GEMM Op")
    parser.add_argument("--M", type=int, default=4096, help="M dimension")
    parser.add_argument("--N", type=int, default=4096, help="N dimension")
    parser.add_argument("--K", type=int, default=4096, help="K dimension")
    parser.add_argument("--bench", action="store_true", help="只跑性能，跳过正确性验证")
    args = parser.parse_args()
    main(M=args.M, N=args.N, K=args.K, bench=args.bench)
