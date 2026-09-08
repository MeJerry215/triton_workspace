# TileLang 语法与 API 参考

> **目标读者**: 有 Python 基础的 GPU 编程开发者，希望学习 TileLang 语法并编写高性能 GPU Kernel 的人员。
>
> **范围**: 覆盖 TileLang DSL 的核心语法、数据结构、控制流、内置函数以及编译执行 API。
>
> **版本参考**: TileLang `main` 分支 (2025-2026)

---

## 1. TileLang 语法速览

TileLang 是一个基于 **Apache TVM TIR/TIRX** 构建的 Python DSL（领域特定语言），用于编写 GPU Kernel。它提供了一组 Python 级别的语言构造，能够表达 **tile 级别的矩阵运算、数据搬运和并行计算**。

### 一个最小的 TileLang Kernel

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def matmul(A, B, block_M: int, block_N: int, block_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C
```

### 编写 TileLang Kernel 的基本规则

| 规则 | 说明 |
|------|------|
| **装饰器** | 使用 `@tilelang.jit` 或 `@T.prim_func` 标记 |
| **张量参数** | 使用 `T.Tensor((shape), dtype)` 声明输入张量 |
| **输出张量** | 使用 `T.empty((shape), dtype)` 声明或直接在 `@tilelang.jit(out_idx=[...])` 声明输出索引 |
| **符号形状** | 使用 `T.const("M, N, K")` 声明符号变量 |
| **Kernel 上下文** | 使用 `with T.Kernel(grid, threads=...) as block_id` 进入 kernel 作用域 |
| **Buffer 分配** | 使用 `T.alloc_shared` / `T.alloc_fragment` / `T.alloc_local` 分配各级内存 |
| **数据搬运** | 使用 `T.copy` 在不同内存层级之间搬运数据 |
| **Tile 计算** | 使用 `T.gemm` 执行矩阵乘，使用 `T.reduce_*` 执行归约 |
| **并行循环** | 使用 `T.Parallel` 标记可并行化的循环 |
| **流水线** | 使用 `T.Pipelined` 标记需要软件流水线的循环 |

---

## 2. 核心语法元素

### 2.1 `@tilelang.jit` vs `@T.prim_func`

这两个装饰器是 TileLang 的入口，但角色不同：

| | `@tilelang.jit` | `@T.prim_func` |
|---|---|---|
| **角色** | 完整的 JIT 编译器（编译 + 缓存 + 执行） | AST-to-TIR 转换器 |
| **输出** | `JITKernel`（可执行的 kernel 适配器） | `tirx.PrimFunc`（TIR 中间表示） |
| **是否编译** | 是 — 自动调用 `lower()` + nvcc | 否 — 仅生成 TIR |
| **是否可执行** | 是 — `kernel(a, b)` 直接运行 | 否 — 需传给 `tilelang.compile()` |
| **执行模式** | lazy（返回 `JITKernel`）和 eager（立即执行） | 无 |

**典型用法**：

```python
# ─── @tilelang.jit (eager 模式) ───
@tilelang.jit
def gemm(A, B, C, block_M=64):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], fp16]
    B: T.Tensor[[K, N], fp16]
    C: T.Tensor[[M, N], fp16]
    with T.Kernel(M, N, thread=128) as (bx, by):
        ...
gemm(a, b, c)         # 编译 + 立即执行


# ─── @tilelang.jit (lazy 模式) ───
@tilelang.jit
def matmul(M, N, K, block_M=128, block_N=128, block_K=32):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), fp16),
               B: T.Tensor((K, N), fp16),
               C: T.Tensor((M, N), fp16)):
        ...
    return kernel

k = matmul(1024, 1024, 1024)  # 编译
k(a, b)                       # 执行


# ─── 纯 @T.prim_func (手动编译) ───
@T.prim_func
def my_kernel(A: T.Tensor((M, K), fp16), ...):
    ...

artifact = tilelang.lower(my_kernel, target="cuda")
k = tilelang.compile(my_kernel, target="cuda")
```

### 2.2 符号变量 —— `T.const`

`T.const` 用于声明在编译期未知的符号变量（如矩阵维度）。编译时由传入的实际 tensor shape 推断。

```python
M, N, K = T.const("M, N, K")       # 多个符号
M = T.const("M")                     # 单个符号
BM = T.const("BM", default=128)      # 带默认值的符号
```

### 2.3 数据类型

TileLang 的数据类型通过 `T` 命名空间访问：

| 类型 | 说明 |
|------|------|
| `T.float16` / `T.fp16` | 16 位半精度浮点 |
| `T.bfloat16` / `T.bf16` | 16 位 Brain 浮点 |
| `T.float32` / `T.float` | 32 位单精度浮点 |
| `T.float64` | 64 位双精度浮点 |
| `T.int8`, `T.int16`, `T.int32`, `T.int64` | 有符号整数 |
| `T.uint8`, `T.uint16`, `T.uint32`, `T.uint64` | 无符号整数 |
| `T.bool` | 布尔类型 |

**字符串形式的 dtype**：很多 API 也接受字符串形式（如 `"float16"`, `"int32"`）。

### 2.4 Tensor 声明

```python
# 输入张量（使用 T.Tensor）
A: T.Tensor((M, K), T.float16)
B: T.Tensor[[M, N], T.float32]      # 也支持方括号语法

# 输出张量（使用 T.empty）
C = T.empty((M, N), T.float16)

# 在 @tilelang.jit 的 out_idx 参数声明输出
@tilelang.jit(out_idx=[-3, -2, -1])   # 最后三个参数是输出
def my_kernel(X, ..., Y, Mean, Rstd):
    ...
```

---

## 3. 内存分配 API

TileLang 提供多种内存作用域的分配函数，对应 GPU 存储层级：

```python
# Register / fragment — 寄存器级存储，用于计算
T.alloc_fragment((block_M, block_N), T.float32)

# Shared memory — 线程块内共享，用于数据复用和线程间通信
T.alloc_shared((block_M, block_K), T.float16)

# Local memory — 线程私有本地存储（可能溢出到显存）
T.alloc_local((TILE_K,), T.float16)

# Single-element variable — 单变量寄存器分配
T.alloc_var(T.float32, init=0.0)

# Global memory workspace — 全局显存工作区
T.alloc_global((N,), T.float32)
```

| 分配 API | 内存层级 | CUDA 对应 | 用途 |
|----------|---------|-----------|------|
| `T.alloc_fragment` | 寄存器 (local.fragment) | Register | 核心计算（gemm 操作数、归约中间值） |
| `T.alloc_shared` | 共享内存 (shared.dyn) | `__shared__` | 线程块内共享数据 |
| `T.alloc_local` | 本地内存 (local) | `__local__` | 线程私有存储 |
| `T.alloc_var` | 寄存器变量 (local.var) | Register | 标量变量 |
| `T.alloc_global` | 全局内存 (global) | `__device__` | kernel 工作区 |

### 3.1 `T.alloc_reducer` —— 专用的归约 Buffer

用于需要跨线程归约的场景，会自动生成归约相关的代码：

```python
o_reducer = T.alloc_reducer(block_M, accum_dtype, replication="all")
T.clear(o_reducer)
for ...:
    o_reducer[i] += ...
T.finalize_reducer(o_reducer)
T.copy(o_reducer, output)
```

### 3.2 `T.alloc_barrier` / `T.alloc_cluster_barrier` —— 屏障分配

```python
mbar = T.alloc_barrier(1)             # 单个 mbarrier
mbars = T.alloc_barrier([128] * n)     # 多个 mbarrier
cluster_bar = T.alloc_cluster_barrier()  # 集群屏障
```

### 3.3 `T.alloc_tmem` —— Tensor Memory (SM100+)

用于 Blackwell 架构的 Tensor Memory：

```python
tmem = T.alloc_tmem((block_M, block_N), T.float16)
```

### 3.4 `T.alloc_descriptor` —— 描述符分配

```python
desc = T.alloc_descriptor(1)              # WGMMA shared memory descriptor
tcgen05_desc = T.alloc_tcgen05_smem_desc()  # TCGen05 shared memory descriptor
```

---

## 4. Kernel 启动与线程绑定

### 4.1 `T.Kernel` —— Kernel 体（CUDA `__global__` 函数的等价物）

**`T.Kernel` 是一个上下文管理器，它的 `with` 代码块体就是 GPU kernel 的实际实现。** 这与 `@T.prim_func` 完全不同——`@T.prim_func` 是函数声明级别的装饰器，而 `T.Kernel` 是函数体内部的上下文。

简单说：
- `@T.prim_func`（或 `@tilelang.jit` 的 eager 模式）→ 负责将 Python 代码转换为 TIR 中间表示
- `T.Kernel` → 负责声明 **kernel 启动配置**（grid/block），它的 `with` 体就是 **kernel 的代码本体**（对应 CUDA 的 `__global__` 函数体）

**关系示意图**：

```text
@tilelang.jit         ← 完整的 JIT 编译器
  └─ @T.prim_func    ← 将函数声明转为 TIR (可选，lazy 模式需显式写出)
       └─ T.Kernel   ← Kernel 体（CUDA __global__ 函数体）
              ├─ alloc_*         分配 buffer
              ├─ T.copy          数据搬运
              ├─ T.Parallel / T.Pipelined  循环
              ├─ T.gemm / T.reduce_*       计算
              └─ T.copy          写回结果
```

**在 `with T.Kernel(...)` 体内部编写的内容**包括：
```
  内存分配  →  T.alloc_shared / T.alloc_fragment / T.alloc_local ...
  数据搬运  →  T.copy(src, dst)
  计算      →  T.gemm / T.reduce_* / T.fill / T.clear
  循环      →  for ... T.Parallel / T.Pipelined / T.serial
  写回结果  →  T.copy(local_or_shared, global_tensor)
```

**`T.Kernel` 的完整用法示例**：

```python
@tilelang.jit                # ← ① JIT 编译器入口
def gemm(...):
    A: T.Tensor(...)         # ← ② Tensor 声明 (eager 模式直接在 jit 函数内声明)
    C = T.empty(...)
    
    with T.Kernel(...) as (bx, by):    # ← ③ Kernel 体：启动配置 + 实现
        A_shared = T.alloc_shared(...)  #     ├─ 内存分配
        C_local  = T.alloc_fragment(...)#     ├─
        T.clear(C_local)                #     ├─ 初始化
        for k in T.Pipelined(...):      #     ├─ 循环
            T.copy(A[bx, k], A_shared)  #     ├─ 数据搬运
            T.gemm(A_shared, B, C_local)#     ├─ 计算
        T.copy(C_local, C[bx, by])      #     ├─ 写回
        # ← ④ Kernel 体结束
    return C
```

> **关键区别总结**：
> | | `@T.prim_func` | `T.Kernel` |
> |---|---|---|
> | **种类** | 函数装饰器（decorator） | 上下文管理器（context manager） |
> | **作用** | 将函数声明转为 TIR | 声明启动配置 + 定义 kernel 实现体 |
> | **位置** | 在函数定义上方 | 在函数体内部 |
> | **产出** | `tirx.PrimFunc`（TIR 中间表示） | 无独立产出，其 `with` 体被编译为 CUDA `__global__` 函数 |
> | **类比 CUDA** | 无直接等价物（类似函数签名声明） | 相当于 `__global__ void kernel(...)` 的花括号体 |

**`T.Kernel` 的参数**：

```python
# 1D grid  — 对应 CUDA 的 blockIdx.x
with T.Kernel(grid_x, threads=128) as bx:
    ...   # bx = blockIdx.x

# 2D grid  — 对应 blockIdx.x, blockIdx.y
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...   # bx = blockIdx.x, by = blockIdx.y

# 3D grid
with T.Kernel(grid_x, grid_y, grid_z, threads=128) as (bx, by, bz):
    ...

# Persistent kernel (block 数等于 SM 数)
sm_num = driver.get_num_sms()
with T.Kernel(sm_num, threads=128) as block_id:
    ...   # block_id 在 0..sm_num-1 之间，通过 T.Persistent 分配 tile
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `*grid_dims` | int / PrimExpr | 各维度的 grid 大小（对应 `gridDim.x/y/z`） |
| `threads` | int / tuple | 线程块大小。可为单个 int（`threads=128`）或 tuple（`threads=(32, 4)` 表示 `blockDim.x=32, blockDim.y=4`） |
| `block_id` / `(bx, by, bz)` | 上下文变量 | 当前 block 索引（对应 `blockIdx.x/y/z`） |

### 4.2 线程绑定 —— `T.get_thread_binding` / `T.thread_binding`

```python
tn = T.get_thread_binding(0)          # threadIdx.x
tk = T.get_thread_binding(1)          # threadIdx.y

# 多维度线程绑定
tn, tk = T.get_thread_bindings()       # 获取所有线程索引
```

### 4.3 `T.ClusterKernel` —— 集群 Kernel (SM90+)

用于 Hopper 架构的集群启动：

```python
with T.ClusterKernel(grid_x, grid_y, cluster_x, cluster_y, threads=128) as (bx, by):
    ...
```

### 4.4 `T.CUDASourceCodeKernel` —— 直接嵌入 CUDA C++

```python
with T.CUDASourceCodeKernel("""
    // 直接编写 CUDA C++ 代码
    int tid = threadIdx.x;
    output[tid] = input[tid] * 2.0f;
""") as kernel:
    ...
```

### 4.5 `T.ws` / `T.WarpSpecialize` —— Warp Specialization

将不同的 warp 组分配给 producer（数据搬运）和 consumer（计算）角色：

```python
# Producer warp group (tid 0-127)
with T.ws(0):
    # 执行数据搬运 (TMA load, cp.async...)

# Consumer warp group (tid 128-255)
with T.ws(1):
    # 执行计算 (WGMMA...)
```

---

## 5. 循环结构

### 5.1 `T.Parallel` —— 可并行化的元素级循环

`T.Parallel` 标记的循环是多线程并行执行的，通常用于元素级运算：

```python
# 1D 并行循环
for i in T.Parallel(N):
    output[i] = input[i] * 2

# 2D 嵌套并行循环
for i, j in T.Parallel(M, N):
    C_local[i, j] = A_local[i, j] + B_local[i, j]

# 带 coalesced_width 的并行循环
for i, j in T.Parallel(M, N, coalesced_width=8):
    ...
```

**重要规则**：`T.Parallel` 会由编译器的 `LayoutInference` pass 自动推断并优化线程映射。也可通过 `loop_layout` 手动指定 fragment layout。

### 5.2 `T.Pipelined` —— 软件流水线循环

`T.Pipelined` 用于需要隐藏访存延迟的循环。编译器会自动插入软件流水线（prologue + body + epilogue）：

```python
# 3 级流水线
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[bx * block_M, k * block_K], A_shared)  # 加载
    T.copy(B[k * block_K, by * block_N], B_shared)  # 加载
    T.gemm(A_shared, B_shared, C_local)              # 计算
```

| 参数 | 说明 |
|------|------|
| `num_stages` | 流水线级数（2-5，取决于共享内存大小） |
| `order` | 迭代顺序（默认顺序执行） |

### 5.3 `T.Persistent` —— Persistent 循环

`T.Persistent` 在多个 SM 之间动态分配 tile，适合负载不均衡的场景：

```python
with T.Kernel(sm_num, threads=128) as block_id:
    # Persistent loop: 自动在 sm_num 个 block 间均分 tile
    for bx, by in T.Persistent([m_blocks, n_blocks], sm_num, block_id):
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[bx * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, by * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[bx * block_M, by * block_N])
```

### 5.4 `T.serial` —— 串行循环

```python
for k in T.serial(N):
    ...
```

### 5.5 `T.unroll` —— 完全展开循环

```python
# 循环完全展开
for k in T.unroll(4):
    ...
```

### 5.6 `T.vectorized` —— 向量化循环

```python
# 编译器会将其向量化为向量加载指令
for k in T.vectorized(TILE_K):
    A_local[k] = A[...]
```

---

## 6. 数据搬运 API

### 6.1 `T.copy` —— 通用数据复制

`T.copy` 是 TileLang 中最核心的数据搬运 API，能根据源和目标的存储层级自动选择合适的指令：

```python
# Global → Shared (通过 cp.async 或普通加载)
T.copy(A[bx * block_M, ko * block_K], A_shared)

# Shared → Fragment (寄存器)
T.copy(A_shared, A_frag)

# Fragment → Global (直接写回显存)
T.copy(C_local, C[bx * block_M, by * block_N])

# Global → Fragment (直接加载)
T.copy(X[bx * blk_m, 0], X_local)
```

`T.copy` 会根据 **源和目标 buffer 的 scope** 自动推断搬运用途和指令：

| 源 → 目标 | 使用的指令 |
|-----------|-----------|
| global → shared | `cp.async` (SM80+) 或普通 `ldg`+`sts` |
| shared → fragment | `lds` 加载到寄存器 |
| shared → global | `stg` 写回显存 |
| fragment → global | 直接写回（bypass shared memory） |
| global → fragment | 直接加载（bypass shared memory） |

### 6.2 `T.async_copy` —— 异步复制

```python
T.async_copy(A[bx * block_M, ko * block_K], A_shared)
```

### 6.3 `T.tma_copy` —— TMA 复制 (SM90+)

Hopper 架构张量内存加速器：

```python
T.tma_copy(A[bx * block_M, ko * block_K], A_shared)
```

### 6.4 `T.transpose` / `T.im2col` —— 特殊变换

```python
T.transpose(A_shared, A_T_shared)      # 矩阵转置
T.im2col(input, col_buf, kernel_size)  # im2col 变换
```

---

## 7. Shared Memory Swizzle（共享内存排布控制）

> **本节与 `T.use_swizzle()` 是两回事**：`T.use_swizzle()` 控制 threadblock **调度顺序**（详见第 15 章），而本节控制数据在 **shared memory 中的排布方式**。

### 7.1 为什么需要 Shared Memory Swizzle？

GPU 的 shared memory 被划分为 32 个 bank。当同一个 warp 中的多个线程访问**同一个 bank 中不同地址**时，会发生 **bank conflict**，导致访存串行化。Swizzle 通过 XOR 地址重映射，将连续的数据行交错分布到不同 bank，从而避免冲突。

### 7.2 `make_swizzled_layout()` + `T.annotate_layout()` —— 核心 API

控制 shared memory 排布的标准方式是通过 **`tilelang.layout.make_swizzled_layout()` 生成一个 swizzle layout，再通过 `T.annotate_layout()` 绑定到 shared buffer**。

```python
from tilelang.layout import make_swizzled_layout

# 生成 swizzle layout 并与 shared buffer 绑定
A_shared = T.alloc_shared((block_M, block_K), T.float16)
B_shared = T.alloc_shared((block_K, block_N), T.float16)

T.annotate_layout({
    A_shared: make_swizzled_layout(A_shared),
    B_shared: make_swizzled_layout(B_shared),
})
```

`make_swizzled_layout` 会自动根据 buffer 的 shape 和 dtype 选择合适的 swizzle 模式。对于 fp16 的 GEMM shared tile，它默认采用 128B 全 bank swizzle。

### 7.3 `make_swizzled_layout` 的变体

| API | 用途 |
|-----|------|
| `make_swizzled_layout(buffer)` | **通用 swizzle layout**（自动选择模式），适用于标准 GEMM/copy |
| `make_wgmma_swizzled_layout(buffer)` | **WGMMA 专用**（SM90+ Hopper），用于 Hopper 的 warp group MMA |
| `make_tcgen05mma_swizzled_layout(buffer)` | **TCGen5 MMA 专用**（SM100+ Blackwell） |
| `make_volta_swizzled_layout(buffer, is_a)` | **Volta 专用**（SM70），用于第一代 Tensor Core |

### 7.4 按 bank 粒度的 Swizzle 模式

TileLang 提供了三种不同粒度的 bank swizzle，以 `make_*_bank_swizzled_layout` 命名：

| API | 粒度 | XOR 位数 | 128B 内可容纳 (fp16) | 适用场景 |
|-----|------|----------|---------------------|---------|
| `make_quarter_bank_swizzled_layout(buffer)` | **32B** | 1-bit | 8×16 | 轻量级 swizzle，较少占用地址空间 |
| `make_half_bank_swizzled_layout(buffer)` | **64B** | 2-bit | 8×32 | 中等 swizzle，平衡 bank conflict 和地址位消耗 |
| `make_full_bank_swizzled_layout(buffer)` | **128B** | 3-bit | 8×64 | **最强 swizzle**，128B 内完全消除 bank conflict |

它们也可以直接调用（传入 `rows`, `cols`, `element_size_bits`）：

```python
from tilelang.layout import (
    make_quarter_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_full_bank_swizzled_layout,
)

# 32B swizzle (1-bit XOR, 8×16 for fp16)
layout_32b = make_quarter_bank_swizzled_layout(8, 16, element_size_bits=16)

# 64B swizzle (2-bit XOR, 8×32 for fp16)
layout_64b = make_half_bank_swizzled_layout(8, 32, element_size_bits=16)

# 128B swizzle (3-bit XOR, 8×64 for fp16)
layout_128b = make_full_bank_swizzled_layout(8, 64, element_size_bits=16)
```

**Swizzle 模式的数学原理**：通过对行地址与列索引做 XOR 运算，将原本连续的列地址散布到不同 bank。

```
Without swizzle:              With 128B swizzle:
   bank: 0 1 2 ... 31           bank: 0 1 2 ... 31
   row 0: a b c ...             row 0: a b c ...
   row 1: A B C ...             row 1: A B C ...  (XOR 变换)
   row 2: a b c ...             row 2: c a b ...  (列循环移位)
```

### 7.5 `SwizzleMode` 枚举

`SwizzleMode` 枚举定义了四种 swizzle 模式，用于描述符配置：

```python
from tilelang.layout import SwizzleMode

# 支持的模式
SwizzleMode.NONE        # 无 swizzle
SwizzleMode.SWIZZLE_32B  # 32B swizzle
SwizzleMode.SWIZZLE_64B  # 64B swizzle
SwizzleMode.SWIZZLE_128B # 128B swizzle

# 查询属性
mode.swizzle_byte_size()    # 返回字节数: 1 (NONE), 32, 64, 128
mode.smem_alignment()       # 返回所需对齐字节: 128, 256, 512, 1024
```

### 7.6 `make_linear_layout` —— 线性排布

当不需要 swizzle 时，可以用线性 layout 让数据按行优先连续排列：

```python
from tilelang.layout import make_linear_layout

T.annotate_layout({
    buf: make_linear_layout(buf),
})
```

### 7.7 完整示例：在 GEMM 中使用 Shared Memory Swizzle

```python
from tilelang.layout import make_swizzled_layout

@tilelang.jit
def gemm_swizzled(A, B, block_M=128, block_N=128, block_K=32, threads=256, num_stages=3):
    M, N, K = T.const("M N K")
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float16)
        B_shared = T.alloc_shared((block_K, block_N), T.float16)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)

        # ★ 核心：为 shared buffer 绑定 swizzle layout
        T.annotate_layout({
            A_shared: make_swizzled_layout(A_shared),
            B_shared: make_swizzled_layout(B_shared),
        })

        T.use_swizzle(10)          # ← 这是 threadblock 调度 swizzle（L2 优化）
        T.clear(C_local)           #   与 shared memory swizzle 是两回事

        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[bx * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, by * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[bx * block_M, by * block_N])

    return C
```

### 7.8 vs `T.use_swizzle()` —— 澄清两者的区别

| | `T.use_swizzle(panel_size)` | `make_swizzled_layout()` + `T.annotate_layout()` |
|---|---|---|
| **控制对象** | Threadblock **调度顺序**（哪个 block 处理哪个 tile） | 数据在 shared memory 中的**字节排布** |
| **优化目标** | **L2 cache 局部性**（减少 L2 miss） | **Shared memory bank conflict**（减少 bank conflict） |
| **作用域** | Grid 级别的 tile 分配 | Shared memory 内的行/列交错 |
| **类比** | CUTLASS 的 `threadblock_swizzle` | CUTLASS 的 `SmemStaging` 或 CuTe 的 `Swizzle` |
| **写法** | `T.use_swizzle(10)` | `T.annotate_layout({buf: make_swizzled_layout(buf)})` |

> **经验法则**：只要发现 shared memory 的访问模式有规律的行列访问（如 GEMM 的 A/B tile 加载），就应该用 `make_swizzled_layout` 做 bank conflict 优化。而 `T.use_swizzle` 是 grid 级别的补充优化。

---

## 8. Tile 计算 API

### 7.1 `T.gemm` —— Tile 级矩阵乘

`T.gemm` 是 TileLang 最高层的计算 API，将 tile 级别的矩阵乘运算分派到底层 Tensor Core：

```python
T.gemm(A_shared, B_shared, C_local)

# 转置版本
T.gemm(A_shared, B_shared, C_local, transpose_A=True, transpose_B=False)

# 控制 warp 策略
T.gemm(A_shared, B_shared, C_local, policy=GemmWarpPolicy.Square)

# 累加器清零
T.gemm(A_shared, B_shared, C_local, clear_accum=True)

# k_pack 参数（在累加维度上打包）
T.gemm(A_shared, B_shared, C_local, k_pack=2)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `transpose_A` | bool | False | A 矩阵是否转置 |
| `transpose_B` | bool | False | B 矩阵是否转置 |
| `policy` | GemmWarpPolicy | Square | Warp 映射策略（Square, FullRow, FullCol） |
| `clear_accum` | bool | False | 是否在 GEMM 前清零累加器 |
| `k_pack` | int | 1 | K 维度打包因子 |

### 7.2 `T.wgmma_gemm` —— WGMMA GEMM (SM90+)

Hopper 架构的 warp group 级矩阵乘：

```python
T.wgmma_gemm(A_shared, B_shared, C_local)
```

### 7.3 `T.tcgen05_gemm` —— TC Gen5 GEMM (SM100+)

Blackwell 架构的第五代 Tensor Core：

```python
T.tcgen05_gemm(A_shared, B_shared, C_local)
```

---

## 8. 归约与扫描操作

### 8.1 归约操作

```python
# 沿维度 1 求和
T.reduce_sum(X_local, sum_row, dim=1)

# 沿维度 1 求最大值
T.reduce_max(X_frag, max_val, dim=1)

# 沿维度 1 求最小值
T.reduce_min(X_frag, min_val, dim=1)

# 沿维度 1 求绝对值和
T.reduce_abssum(X_frag, result, dim=1)

# 沿维度 1 求绝对值最大值
T.reduce_absmax(X_frag, result, dim=1)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | Buffer | 源 fragment buffer |
| `dst` | Buffer | 目标 buffer（通常维度少于源） |
| `dim` | int | 归约的维度 |
| `clear` | bool | 归约前是否清零目标 buffer |

**详细归约 API**：

| API | 运算 |
|-----|------|
| `T.reduce_sum(src, dst, dim)` | 求和 |
| `T.reduce_max(src, dst, dim)` | 最大值 |
| `T.reduce_min(src, dst, dim)` | 最小值 |
| `T.reduce_abssum(src, dst, dim)` | 绝对值求和 |
| `T.reduce_absmax(src, dst, dim)` | 绝对值最大值 |
| `T.reduce_bitand(src, dst, dim)` | 按位与 |
| `T.reduce_bitor(src, dst, dim)` | 按位或 |
| `T.reduce_bitxor(src, dst, dim)` | 按位异或 |

### 8.2 Warp 级别归约

```python
T.warp_reduce_sum(value)     # Warp 内求和归约
T.warp_reduce_max(value)     # Warp 内最大值归约
T.warp_reduce_min(value)     # Warp 内最小值归约
T.warp_reduce_bitand(value)  # Warp 内按位与归约
T.warp_reduce_bitor(value)   # Warp 内按位或归约
```

### 8.3 高级归约 —— `T.alloc_reducer` + `T.finalize_reducer`

```python
o_reducer = T.alloc_reducer(block_M, accum_dtype, replication="all")
T.clear(o_reducer)
for i1_m, i1_n in T.Parallel(block_M, block_N):
    o_reducer[i1_m] += a_frag[i1_m, i1_n] * x_frag[i1_n]
T.finalize_reducer(o_reducer)
T.copy(o_reducer, output)
```

### 8.4 扫描操作

```python
T.cumsum(src, dst, dim=0)   # 前缀和
T.cummax(src, dst, dim=0)   # 前缀最大值
```

---

## 9. 填充与初始化

### 9.1 `T.fill` —— 填充

```python
T.fill(C_local, 0)                       # 填充为标量
T.fill(C_local, T.float32(0))            # 显式类型填充
T.fill(expand_max_idx, -1)               # 填充为 -1
T.fill(lse, -T.infinity(T.float32))      # 填充为负无穷
```

### 9.2 `T.clear` —— 清零

```python
T.clear(C_local)          # 清零（等价于 T.fill(buf, 0)）
T.clear(C_accum)
```

---

## 10. 数学与类型转换

### 10.1 类型转换

```python
# 使用 T.Cast
val_fp32 = T.Cast(T.float32, val_fp16)
val_fp16 = T.Cast("float16", val_fp32)

# 使用 .astype 方法（同 TVM TIR）
val_fp32 = val_fp16.astype(T.float32)
```

### 10.2 数学函数

| API | 说明 | 对应 CUDA |
|-----|------|-----------|
| `T.exp(x)` | 指数 | `expf` |
| `T.exp2(x)` | 2 的指数 | `exp2f` |
| `T.log(x)` | 自然对数 | `logf` |
| `T.log2(x)` | 以 2 为底对数 | `log2f` |
| `T.tanh(x)` | 双曲正切 | `tanhf` |
| `T.rsqrt(x)` | 平方根倒数 | `rsqrtf` |
| `T.sqrt(x)` | 平方根 | `sqrtf` |
| `T.sin(x)` | 正弦 | `sinf` |
| `T.cos(x)` | 余弦 | `cosf` |
| `T.floor(x)` | 向下取整 | `floorf` |
| `T.ceil(x)` | 向上取整 | `ceilf` |
| `T.abs(x)` | 绝对值 | `fabsf` |
| `T.max(x, y)` | 最大值 | `fmaxf` |
| `T.min(x, y)` | 最小值 | `fminf` |
| `T.clamp(val, lo, hi)` | 截断 | `fminf(fmaxf)` |
| `T.if_then_else(cond, t, f)` | 三目运算 | `cond ? t : f` |

### 10.3 辅助函数

```python
T.ceildiv(a, b)          # 向上取整除法 (a + b - 1) // b
T.infinity(T.float32)     # 正无穷
T.float32(1.0)            # 创建 float32 常量
```

---

## 11. 原子操作

```python
# 原子加
T.atomic_add(C[by * block_M + i, bx * block_N + j], C_local[i, j])
T.atomic_add(C[by * block_M + i, bx * block_N + j], C_local[i, j], memory_order="relaxed")

# 原子减
T.atomic_min(ptr, val)
T.atomic_max(ptr, val)

# 原子按位或
T.atomic_or(ptr, val)

# 原子加载/存储
T.atomic_load(ptr)
T.atomic_store(ptr, val)

# 64 位 / 128 位原子加
T.atomic_addx2(ptr, val)   # atomicAdd on 64-bit
T.atomic_addx4(ptr, val)   # atomicAdd on 128-bit
```

---

## 12. Warp 级别原语

### 12.1 Shuffle 指令

```python
# XOR shuffle（跨 lane 数据交换）
T.shfl_xor(value, delta, width=32)

# Down shuffle（向下移动）
T.shfl_down(value, delta, width=32)

# Up shuffle（向上移动）
T.shfl_up(value, delta, width=32)

# Broadcast（从指定 lane 广播）
T.shfl_sync(value, srcLane, width=32)
```

### 12.2 Warp Vote / Ballot 指令

```python
T.any_sync(predicate)      # 任意线程满足条件？
T.all_sync(predicate)      # 所有线程满足条件？
T.ballot_sync(predicate)   # 返回满足条件的线程的 bitmask
T.ballot(predicate)        # 简写（全 warp mask）
T.activemask()             # 当前激活的线程 mask
```

### 12.3 Warp Match 指令 (SM70+)

```python
T.match_any_sync(value)    # 返回与当前线程值相同的 lane mask
T.match_all_sync(value)    # 如果所有线程值相同则返回 mask
```

### 12.4 Lane / Warp ID

```python
T.get_lane_idx()               # 当前线程的 lane 索引 (0-31)
T.get_warp_idx()               # 当前线程的 warp 索引
T.get_warp_idx_sync()          # 同步后的 warp 索引
T.get_warp_group_idx()         # 当前线程的 warp group 索引
T.shuffle_elect(thread_extent) # 在指定范围内选举一个线程
```

---

## 13. 同步与屏障

### 13.1 线程块同步

```python
T.sync_threads()                          # __syncthreads()
T.sync_threads(barrier_id, arrive_count)  # 命名屏障同步
```

### 13.2 Named Barrier（命名屏障）

```python
# Producer: 到达但不等待
T.named_barrier_arrive(ready_barrier, total_threads)

# Consumer: 同步直到所有线程到达
T.sync_threads(ready_barrier, total_threads)
```

### 13.3 Mbarrier (SM90+)

```python
mbar = T.alloc_barrier(1)

T.mbarrier_expect_tx(mbar, tx_count)       # 设置预期事务数
T.mbarrier_arrive(mbar)                     # 到达屏障
T.mbarrier_wait_parity(mbar, parity)        # 等待屏障奇偶状态
```

### 13.4 集群同步 (SM90+)

```python
T.cluster_arrive()                     # 集群到达
T.cluster_wait()                       # 集群等待
T.cluster_sync()                       # 集群同步
T.block_rank_in_cluster()              # 当前 block 在集群中的排名
```

### 13.5 全局同步

```python
T.sync_global()      # grid 级同步（所有 block 同步）
T.sync_grid()        # grid 级同步（sm100+ 支持）
```

### 13.6 Warp 同步

```python
T.sync_warp()        # __syncwarp()
```

---

## 14. 逻辑与比较操作

```python
T.any_of(predicate)   # 任意线程满足条件（block级别）
T.all_of(predicate)   # 所有线程满足条件（block级别）

# 同步版 predicate
T.syncthreads_count(predicate)   # 返回满足条件的线程数
T.syncthreads_and(predicate)     # 所有线程都满足？
T.syncthreads_or(predicate)      # 任意线程满足？
```

---

## 15. Annotation / 编译提示

### 15.1 `T.use_swizzle` —— Threadblock 调度 Swizzle（L2 Cache 优化）

> **⚠️ 注意**：本节控制的是 threadblock 在 grid 上的**调度顺序**（rasterization），与 shared memory 内部的数据排布 **无关**。如果你要找的是 shared memory swizzle（bank conflict 优化），请参见 **[第 7 章](#7-shared-memory-swizzle共享内存排布控制)**。

`T.use_swizzle` 通过改变 threadblock 遍历 tile 的顺序来提升 L2 cache 局部性。默认的 row-major 遍历会导致相邻 threadblock 在 L2 中争抢相同的 cache line，而 swizzle 将 tile 按 panel 分组交错分配。

```python
# 启用 threadblock swizzle（改变 block 处理 tile 的顺序）
T.use_swizzle(panel_size=10)
T.use_swizzle(panel_size=10, order="row")    # 行优先 rasterization
T.use_swizzle(panel_size=10, order="column") # 列优先 rasterization
T.use_swizzle(10, enable=False)              # 显式禁用
```

### 15.2 `T.annotate_layout` —— 手动指定 Layout

> **用于 shared memory swizzle**：将 `make_swizzled_layout()` 与 `T.annotate_layout()` 搭配使用可以控制 shared memory 中数据的排布，详见 **[第 7 章](#7-shared-memory-swizzle共享内存排布控制)**。

```python
from tilelang.layout import Fragment, Layout

# Fragment layout（寄存器级排布，影响 Tensor Core 指令布局）
T.annotate_layout({
    A_frag: Fragment("row-major"),
    B_frag: Fragment("col-major"),
})

# Shared memory layout（通过 make_swizzled_layout 生成）
from tilelang.layout import make_swizzled_layout
T.annotate_layout({
    A_shared: make_swizzled_layout(A_shared),
})

# 自定义 Layout 函数
T.annotate_layout({
    buf: Layout(buf.shape, lambda i, j: (i, j)),  # 恒等映射
})
```

### 15.3 `T.annotate_safe_value` —— 安全值标注

```python
T.annotate_safe_value({
    output_buf: 0.0,   # 表示 output_buf 在读取前已被初始化为 0
})
```

### 15.4 `T.annotate_l2_hit_ratio` —— L2 驻留提示

```python
T.annotate_l2_hit_ratio({
    global_buf: 0.5,   # 期望 50% 的 L2 命中率
})
```

### 15.5 `T.annotate_restrict_buffers` —— 非 restrict 标注

当多个 buffer 可能别名时使用：

```python
T.annotate_restrict_buffers(x, y)
```

### 15.6 `T.annotate_min_blocks_per_sm` —— 驻留块数

```python
T.annotate_min_blocks_per_sm(2)   # 每个 SM 至少 2 个 block
```

---

## 16. 调试与打印

### 16.1 `T.print` —— 设备端打印

```python
T.print("value = ", val)                # 类似 printf
T.print(val0, val1, format="%f %d")     # 格式控制
```

### 16.2 `T.device_assert` —— 断言

```python
T.device_assert(condition, "error message")
```

---

## 17. 随机数生成

```python
state = T.rng_init(seed)                # 初始化随机状态
val = T.rng_rand(state)                 # 生成随机 uint32
val_f = T.rng_rand_float(state)         # 生成随机浮点数
```

---

## 18. PDL（可编程延迟隐藏）

```python
T.pdl_trigger(...)      # PDL 触发
T.pdl_sync(...)         # PDL 同步
```

---

## 19. 运算符重载

TileLang 中的 Buffer 支持基本的算术运算符重载：

```python
C_local[i, j] = A_local[i, j] + B_local[i, j]   # 加法
C_local[i, j] = A_local[i, j] * 2.0              # 乘法
C_local[i, j] = A_local[i, j] - B_local[i, j]   # 减法
C_local[i, j] = A_local[i, j] / B_local[i, j]   # 除法
x[i] += y[i]                                      # 自加
```

---

## 20. 编译与执行 API

### 20.1 `tilelang.compile`

```python
import tilelang

# 编译 PrimFunc
kernel = tilelang.compile(prim_func, target="cuda")

# 获取生成的 CUDA 源码
print(kernel.get_kernel_source())

# 执行
kernel(a, b, c)
```

### 20.2 `JITKernel` —— 编译后的执行单元

```python
# 通过 @tilelang.jit 的 compile 方法获得
k = my_kernel.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)

# 执行
k(tensor_a, tensor_b)

# 查看生成的 CUDA 代码
source = k.get_kernel_source()

# 获取 profiler（自动生成测试数据）
profiler = k.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

# 验证正确性
profiler.assert_allclose(ref_func, rtol=1e-2, atol=1e-2)

# 性能测试
latency = profiler.do_bench(warmup=100, rep=500)
```

### 20.3 `tilelang.lower`

```python
# 获取编译后的 artifact（包含 host_mod, device_mod 等）
artifact = tilelang.lower(prim_func, target="cuda")
```

### 20.4 `tilelang.profiler.do_bench`

```python
from tilelang.profiler import do_bench

ms = do_bench(lambda: kernel(a, b), warmup=25, rep=100)
ms_cupti = do_bench(lambda: kernel(a, b), backend="cupti")
ms_event = do_bench(lambda: kernel(a, b), backend="event")
```

### 20.5 Pass Configs

```python
@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def my_kernel(...):
    ...
```

---

## 21. Autotune 自动调优

```python
from tilelang.autotuner import autotune
import itertools

def get_configs():
    iter_params = dict(
        block_M=[64, 128, 256],
        block_N=[64, 128, 256],
        num_stages=[0, 1, 2, 3],
        threads=[128, 256],
    )
    return [dict(zip(iter_params, values))
            for values in itertools.product(*iter_params.values())]

@autotune(configs=get_configs(), warmup=3, rep=20)
@tilelang.jit
def gemm_autotuned(A, B, block_M=128, block_N=128, num_stages=2, threads=256):
    ...

# 自动编译 + 调优
best = gemm_autotuned.compile(M=1024, N=1024, K=1024)
```

---

## 22. 端到端示例汇总

### 22.1 Vector Add（最简入门）

```python
@tilelang.jit
def vector_add(n: T.const, BLOCK_SIZE: T.const = 1024):
    @T.prim_func
    def kernel(x: T.Tensor((n,), fp16),
               y: T.Tensor((n,), fp16),
               z: T.Tensor((n,), fp16)):
        with T.Kernel(T.ceildiv(n, BLOCK_SIZE), threads=BLOCK_SIZE) as pid:
            offsets = pid * BLOCK_SIZE + T.get_thread_binding(0)
            if offsets < n:
                z[offsets] = x[offsets] + y[offsets]
    return kernel
```

### 22.2 GEMM（标准矩阵乘）

```python
@tilelang.jit
def gemm(A, B, block_M=128, block_N=128, block_K=32, threads=256, num_stages=3):
    M, N, K = T.const("M N K")
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float16)
        B_shared = T.alloc_shared((block_K, block_N), T.float16)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)

        T.use_swizzle(10)
        T.clear(C_local)

        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[bx * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, by * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[bx * block_M, by * block_N])

    return C
```

### 22.3 Online Softmax

```python
@tilelang.jit
def softmax_kernel(X, BLOCK_N=8192, dtype: T.dtype = T.float16):
    X: T.Tensor([M, N], dtype)
    Y = T.empty([M, N], dtype)
    accum_dtype = T.float32
    scale = 1.44269504

    with T.Kernel(T.ceildiv(M, 1), threads=128) as i_m:
        x = T.alloc_fragment([1, BLOCK_N], dtype)
        lse = T.alloc_fragment([1], accum_dtype)
        max_x = T.alloc_fragment([1], dtype)
        exp_x = T.alloc_fragment([1, BLOCK_N], accum_dtype)
        sum_exp_x = T.alloc_fragment([1], accum_dtype)

        T.fill(lse, -T.infinity(accum_dtype))

        for i_n in T.Pipelined(T.ceildiv(N, BLOCK_N)):
            T.copy(X[i_m, i_n * BLOCK_N], x)
            T.reduce_max(x, max_x, dim=1, clear=True)
            for i, j in T.Parallel(1, BLOCK_N):
                exp_x[i, j] = T.exp2(x[i, j] * scale - max_x[i] * scale)
            T.reduce_sum(exp_x, sum_exp_x, dim=1, clear=True)
            for i in T.Parallel(1):
                lse[i] = max_x[i] * scale + T.log2(
                    T.exp2(lse[i] - max_x[i] * scale) + sum_exp_x[i]
                )

        for i_n in T.Pipelined(T.ceildiv(N, BLOCK_N)):
            T.copy(X[i_m, i_n * BLOCK_N], x)
            for i, j in T.Parallel(1, BLOCK_N):
                Y[i_m, i_n * BLOCK_N + j] = T.exp2(x[i, j] * scale - lse[i])

    return Y
```

### 22.4 LayerNorm (TileLang 风格)

```python
@tilelang.jit(out_idx=[-3, -2, -1])
def layernorm_fwd(N, D, eps=1e-5, blk_m=1, threads=256, in_dtype="bfloat16", out_dtype="bfloat16"):
    accum_dtype = "float"

    @T.prim_func
    def main(X: T.Tensor((N, D), in_dtype),
             gamma: T.Tensor((D,), in_dtype),
             beta: T.Tensor((D,), in_dtype),
             Y: T.Tensor((N, D), out_dtype),
             Mean: T.Tensor((N,), accum_dtype),
             Rstd: T.Tensor((N,), accum_dtype)):
        with T.Kernel(T.ceildiv(N, blk_m), threads=threads) as bx:
            X_smem = T.alloc_shared((blk_m, D), in_dtype)
            G_smem = T.alloc_shared((D,), in_dtype)
            B_smem = T.alloc_shared((D,), in_dtype)
            X_local = T.alloc_fragment((blk_m, D), accum_dtype)
            X_sq_local = T.alloc_fragment((blk_m, D), accum_dtype)
            sum_row = T.alloc_fragment((blk_m,), accum_dtype)
            sumsq_row = T.alloc_fragment((blk_m,), accum_dtype)
            mean_row = T.alloc_fragment((blk_m,), accum_dtype)
            rstd_row = T.alloc_fragment((blk_m,), accum_dtype)

            T.copy(X[bx * blk_m, 0], X_smem)
            T.copy(gamma, G_smem)
            T.copy(beta, B_smem)

            for i, j in T.Parallel(blk_m, D):
                X_local[i, j] = T.Cast(accum_dtype, X_smem[i, j])
            for i, j in T.Parallel(blk_m, D):
                X_sq_local[i, j] = X_local[i, j] * X_local[i, j]

            T.reduce_sum(X_local, sum_row, dim=1)
            T.reduce_sum(X_sq_local, sumsq_row, dim=1)

            inv_D = T.float32(1.0) / T.Cast(accum_dtype, D)
            for i in T.Parallel(blk_m):
                mean_row[i] = sum_row[i] * inv_D
                rstd_row[i] = T.rsqrt(sumsq_row[i] * inv_D - mean_row[i] * mean_row[i] + T.Cast(accum_dtype, eps))
                Mean[bx * blk_m + i] = mean_row[i]
                Rstd[bx * blk_m + i] = rstd_row[i]

            for i, j in T.Parallel(blk_m, D):
                norm = (X_local[i, j] - mean_row[i]) * rstd_row[i]
                X_smem[i, j] = T.Cast(out_dtype,
                    norm * T.Cast(accum_dtype, G_smem[j]) + T.Cast(accum_dtype, B_smem[j]))

            T.copy(X_smem, Y[bx * blk_m, 0])

    return main
```

---

## 23. 常见问题 FAQ

**Q: `T.Kernel` 中的 `threads` 参数是否等价于 `blockDim`？**

A: 是的。`threads=128` 等价于 CUDA 的 `blockDim.x = 128`。也可以传入 tuple，如 `threads=(32, 4)` 对应 `blockDim.x=32, blockDim.y=4`。

**Q: `T.alloc_fragment` 和 `T.alloc_shared` 的区别？**

A: `alloc_fragment` 分配的是寄存器级存储（`local.fragment` 作用域），用于计算操作数，直接参与 Tensor Core 指令。`alloc_shared` 分配的是共享内存（`shared.dyn` 作用域），用于 block 内线程间共享数据和缓存全局内存加载。

**Q: `T.copy` 在 global→shared 时使用了什么机制？**

A: 在 SM80+（Ampere）上，`T.copy` 的 global→shared 路径会被编译器自动降级为 `cp.async` 异步拷贝指令，允许与计算重叠。在较早架构上则使用普通的 `ldg` + `sts`。

**Q: `T.Parallel` 和 `T.serial` 的关键区别？**

A: `T.Parallel` 标记的循环会被编译器的 `LayoutInference` pass 自动分析并分配线程映射，实现多线程并行执行。`T.serial` 则是串行执行（单线程），常用于循环依赖的场景（如累加）。

**Q: `T.Pipelined` 是如何工作的？**

A: `T.Pipelined` 的循环体被编译器自动拆分为三个阶段：prologue（预加载）、body（流水线核心）、epilogue（排空）。多个迭代的加载和计算可以重叠，从而隐藏访存延迟。`num_stages` 参数控制流水线深度。

**Q: 如何查看生成的 CUDA C++ 代码？**

A: `kernel.get_kernel_source()` 可以获取最终生成的 CUDA C++ 源码。也可以通过 `TILELANG_PASS_DIFF=1` 环境变量打开 pass 级别的 IR diff 输出。

**Q: TileLang 支持哪些 GPU 后端？**

A: NVIDIA CUDA（默认）、AMD ROCm HIP、Apple Metal、CPU（LLVM/C 代码生成）、WebGPU。通过注册不同的 `DeviceCodegen` 和 `PassPipeline` 实现后端切换。

**Q: 如何将 TileLang 与 PyTorch 集成？**

A: TileLang 的 `JITKernel` 直接接受 `torch.Tensor` 参数（通过 DLPack 协议转换）。可以编写 `torch.autograd.Function` 封装，实现前向/反向 kernel 的自动梯度计算。

---

## 24. API 速查表

### 核心上下文

| API | 用途 |
|-----|------|
| `T.Kernel(*grid, threads=N)` | Kernel 启动配置 |
| `T.ClusterKernel(*grid, cluster_x, cluster_y, threads=N)` | 集群 Kernel (SM90+) |
| `T.ws(warp_group_id)` | Warp Specialization |
| `T.CUDASourceCodeKernel(src)` | 嵌入 CUDA C++ |

### 内存分配

| API | 作用域 |
|-----|--------|
| `T.alloc_fragment(shape, dtype)` | Register (local.fragment) |
| `T.alloc_shared(shape, dtype)` | Shared Memory (shared.dyn) |
| `T.alloc_local(shape, dtype)` | Local Memory (local) |
| `T.alloc_var(dtype)` | Register var (local.var) |
| `T.alloc_global(shape, dtype)` | Global Memory (global) |
| `T.alloc_reducer(shape, dtype)` | Reducer (local.fragment) |
| `T.alloc_barrier(n)` | Mbarrier |
| `T.empty(shape, dtype)` | Output tensor 声明 |

### 循环构造

| API | 说明 |
|-----|------|
| `T.Parallel(*extents)` | 并行循环（元素级） |
| `T.Pipelined(extent, num_stages=N)` | 软件流水线循环 |
| `T.Persistent(tiles, sm_count, block_id)` | Persistent 循环 |
| `T.serial(extent)` | 串行循环 |
| `T.unroll(extent)` | 展开循环 |
| `T.vectorized(extent)` | 向量化循环 |

### 数据搬运

| API | 说明 |
|-----|------|
| `T.copy(src, dst)` | 通用数据搬运 |
| `T.async_copy(src, dst)` | 异步复制 |
| `T.tma_copy(src, dst)` | TMA 复制 (SM90+) |
| `T.transpose(src, dst)` | 转置 |

### Tile 计算

| API | 说明 |
|-----|------|
| `T.gemm(A, B, C)` | 通用 tile GEMM |
| `T.wgmma_gemm(A, B, C)` | WGMMA GEMM (SM90+) |
| `T.tcgen05_gemm(A, B, C)` | TCGen5 GEMM (SM100+) |

### 归约

| API | 说明 |
|-----|------|
| `T.reduce_sum(src, dst, dim)` | 求和归约 |
| `T.reduce_max(src, dst, dim)` | 最大值归约 |
| `T.reduce_min(src, dst, dim)` | 最小值归约 |
| `T.reduce_abssum(src, dst, dim)` | 绝对值求和 |
| `T.reduce_absmax(src, dst, dim)` | 绝对值最大值 |
| `T.finalize_reducer(reducer)` | 归约器收尾 |
| `T.warp_reduce_sum(val)` | Warp 求和 |
| `T.cumsum(src, dst, dim)` | 前缀和 |

### 原子操作

| API | 说明 |
|-----|------|
| `T.atomic_add(ptr, val)` | 原子加 |
| `T.atomic_max(ptr, val)` | 原子最大值 |
| `T.atomic_min(ptr, val)` | 原子最小值 |
| `T.atomic_or(ptr, val)` | 原子按位或 |
| `T.atomic_addx2(ptr, val)` | 64 位原子加 |
| `T.atomic_addx4(ptr, val)` | 128 位原子加 |

### 同步

| API | 说明 |
|-----|------|
| `T.sync_threads()` | `__syncthreads()` |
| `T.sync_warp()` | `__syncwarp()` |
| `T.sync_global()` / `T.sync_grid()` | Grid 级同步 |
| `T.mbarrier_arrive(mbar)` | Mbarrier 到达 |
| `T.mbarrier_wait_parity(mbar, p)` | Mbarrier 等待 |
| `T.cluster_sync()` | 集群同步 |

### Annotation

| API | 说明 |
|-----|------|
| `T.use_swizzle(panel_size)` | Threadblock 调度 swizzle（L2 优化，见 §15.1） |
| `T.annotate_layout(layout_map)` | 手动指定 Layout（含 shared memory swizzle，见 §7） |
| `T.annotate_safe_value(map)` | 安全值标注 |
| `T.annotate_l2_hit_ratio(map)` | L2 驻留比例 |
| `T.annotate_restrict_buffers(*bufs)` | 非 restrict 标注 |
| `T.annotate_min_blocks_per_sm(n)` | 最小驻留 block |

### Shared Memory Swizzle

| API | 说明 |
|-----|------|
| `make_swizzled_layout(buffer)` | 通用 shared memory swizzle layout |
| `make_wgmma_swizzled_layout(buffer)` | WGMMA 专用 swizzle (SM90+) |
| `make_tcgen05mma_swizzled_layout(buffer)` | TCGen5 专用 swizzle (SM100+) |
| `make_full_bank_swizzled_layout(buffer)` | 128B 全 bank swizzle |
| `make_half_bank_swizzled_layout(buffer)` | 64B 半 bank swizzle |
| `make_quarter_bank_swizzled_layout(buffer)` | 32B 四分之一 bank swizzle |
| `make_linear_layout(buffer)` | 线性排布（无 swizzle） |
| `SwizzleMode` | Swizzle 模式枚举（NONE/32B/64B/128B） |

---

*本文件由 AI 辅助生成，基于 TileLang 主分支源代码分析。*
