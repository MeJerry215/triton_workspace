# 500 Gluon 语法与学习指南

面向已有 Triton 基础的开发者，系统梳理 **Gluon 语法、使用方式、学习路径**。内容基于官方教程与仓库源码（`triton/python/tutorials/gluon/`、`triton/python/triton/experimental/gluon/`）。

**官方入口**：[Gluon 简介](https://triton-lang.cn/main/getting-started/tutorials/gluon/intro.html#example-intro)

---

## 500.1 Gluon 是什么

| 维度 | Triton | Gluon |
|------|--------|-------|
| 定位 | 高层 tile SPMD，编译器管布局/搬运 | 同栈更低层，用户显式管布局、共享内存、异步 |
| 装饰器 | `@triton.jit` | `@gluon.jit` |
| 语言模块 | `triton.language as tl` | `triton.experimental.gluon.language as gl` |
| 张量类型 | `tl` block，无 layout 概念 | `distributed_type`，**必须带 layout** |
| 性能天花板 | 编译器自动优化 | 手调可超 Triton，代价是硬件知识 |

**核心心智模型**：Gluon program = GPU 上一个 CTA（thread block）；tile = 带 layout 的 N 维寄存器张量；性能瓶颈往往在 **layout 选错** 或 **未用异步路径**。

---

## 500.2 环境与导入

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
# 标准库辅助（与 tl 类似）
from triton.experimental.gluon.language._standard import cdiv, zeros, zeros_like
```

**要求**：

- Triton 带 `experimental.gluon`（本仓库 `triton/python/triton/experimental/gluon/`）
- CUDA GPU；部分特性按架构门控（见 500.12）

**本地教程路径**（建议按序跑）：

```
triton/python/tutorials/gluon/
  01-intro.py
  02-layouts.py
  03-async-copy.py
  04-tma.py
  05-wgmma.py
  06-tcgen05.py
  07-persistence.py
  08-warp-specialization.py
```

```bash
cd triton/python/tutorials/gluon
pytest 01-intro.py -q
TRITON_PRINT_AUTOTUNING=1 python 01-intro.py
```

---

## 500.3 与 Triton 的语法对照

### 500.3.1 相同部分

- **JIT**：`@gluon.jit` 装饰内核，调用 `kernel[grid](*args, num_warps=..., num_stages=...)`
- **Grid**：`grid = (triton.cdiv(N, BLOCK),)` 或 `def grid(META): return (...)`
- **constexpr**：`XBLOCK: gl.constexpr`，编译期常量
- **autotune**：`@triton.autotune(configs=[...], key=[...])` 可直接套在 `@gluon.jit` 上
- **主机端**：PyTorch CUDA tensor 自动变全局指针；`program_id` / `num_programs` 语义一致
- **大量算子**：`load`/`store`/`where`/`reduce`/`atomic_*`/`reshape`/`broadcast` 等从 Triton 转发，用法与 `tl` 相近

### 500.3.2 Gluon 独有部分

| 能力 | API | 说明 |
|------|-----|------|
| 显式 layout | `BlockedLayout`, `CoalescedLayout`, `AutoLayout` 等 | 张量创建/转换时必须考虑 |
| 布局转换 | `gl.convert_layout(x, layout)` | 可能触发寄存器重排 |
| 共享内存描述符 | `gl.allocate_shared_memory` + `.load`/`.store` | 非直接指针 |
| 异步 G→S | `nvidia.ampere.async_copy` | Ampere+ |
| TMA | `nvidia.hopper.tma` + `TensorDescriptor` | Hopper+ |
| WGMMA | `nvidia.hopper.warpgroup_mma*` | Hopper |
| Warp 专化 | `gl.warp_specialize(...)` | Hopper+ |
| 诊断 | `gl.bank_conflicts`, `gl.num_warps`, `gl.num_ctas` | 调 layout / SMEM |

---

## 500.4 最小内核：标量拷贝

```python
@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)

def copy_scalar(input, output):
    grid = (1,)
    copy_scalar_kernel[grid](input, output, num_warps=1)
```

**要点**：单元素时无需 layout；一旦用 tile 就必须指定 layout。

---

## 500.5 1D memcpy：标量循环版

```python
@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    end = min(start + XBLOCK, xnumel)
    for i in range(start, end):
        value = gl.load(in_ptr + i)
        gl.store(out_ptr + i, value)

def memcpy(input, output, XBLOCK):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK),)
    memcpy_kernel[grid](input, output, xnumel, XBLOCK, num_warps=1)
```

**性能陷阱**：每个 CTA 一次只搬 1 个元素，带宽极差（教程中 ~666 GB/s vs 峰值 TB/s）。下一步必须用 **tile + layout**。

---

## 500.6 autotune

```python
@triton.autotune(
    configs=[triton.Config({"XBLOCK": 2**i}, num_warps=1) for i in range(8, 14)],
    key=["xnumel"],
)
@gluon.jit
def memcpy_kernel_autotune(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK)

def memcpy_autotune(input, output):
    xnumel = input.numel()
    def grid(META):
        return (triton.cdiv(xnumel, META["XBLOCK"]),)
    memcpy_kernel_autotune[grid](input, output, xnumel)
```

**调试**：`TRITON_PRINT_AUTOTUNING=1 python 01-intro.py` 查看选中 config。

**与 Triton 差异**：Gluon autotune 往往还要搜 `layout`、`num_warps`、pipeline stage，config 字段更多。

---

## 500.7 Layout 基础（BlockedLayout）

Gluon 张量在寄存器中的分布由 **layout** 定义，层次：**CTA → warp → lane → register**。

```python
layout = gl.BlockedLayout(
    size_per_thread=[2, 4],   # 每线程拥有的子块元素数
    threads_per_warp=[16, 2], # 每 warp 如何铺线程（NVIDIA warp=32）
    warps_per_cta=[2, 2],     # 每 CTA 如何铺 warp
    order=[1, 0],             # 先铺哪个维： [1,0]=行优先
)
```

**块形状（block shape）**：

```
block_shape[d] = size_per_thread[d] * threads_per_warp[d] * warps_per_cta[d]
```

上例：`[2*16*2, 4*2*2] = [64, 16]`。

**order 含义**：

- `order=[1, 0]`：先沿 dim1 铺，再 dim0 → 行主序子块
- `order=[0, 1]`：列主序子块

**张量大于 block**：按 order 在 block 上 **tile**，每线程寄存器数 = 每 block 寄存器 × tile 数。

**张量小于 block**：**broadcast** 到 block，物理寄存器可能远大于逻辑元素数（浪费寄存器）。

### 500.7.1 1D tile memcpy（相对 Triton 的核心差异）

```python
@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    indices = gl.arange(0, XBLOCK, layout=layout)  # 必须给 layout
    offsets = start + indices
    mask = offsets < xnumel
    value = gl.load(in_ptr + offsets, mask=mask)
    gl.store(out_ptr + offsets, value, mask=mask)
```

**类型推断**：`indices` 的 layout 会向前传播；后续 `load`/`store` 的 tile 继承该 layout。

**1D + num_warps=4 时常用 layout**：

```python
gl.BlockedLayout(
    size_per_thread=[R],      # R 为 2 的幂
    threads_per_warp=[32],
    warps_per_cta=[4],
    order=[0],
)
```

教程实测（GB200）：`XBLOCK=2048, R=1` 可达 ~6.5 TB/s；`R` 影响 LDG/STG 向量化与 stall，需 profile。

### 500.7.2 其他 Distributed Layout

| Layout | 用途 |
|--------|------|
| `AutoLayout()` | 编译器推断；可用 `set_auto_layout` 固定 |
| `CoalescedLayout()` | 全局内存合并访问推断 |
| `SliceLayout(dim, parent)` | 从 parent 切某一维，如 2D→1D 索引 |
| `DotOperandLayout(idx, parent, k_width)` | MMA 操作数布局 |
| `NVMMADistributedLayout` | Tensor Core MMA |
| `DistributedLinearLayout` | 显式 reg/lane/warp/block 基向量 |

```python
# 2D kernel 中取列偏移示例
layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
```

### 500.7.3 layout 工具 API

```python
y = gl.convert_layout(x, target_layout)
y = gl.set_auto_layout(x, concrete_layout)
ll = gl.to_linear_layout(layout, shape)
n = gl.bank_conflicts(distr_ty, shared_ty)  # 共享内存 bank 冲突估计
```

---

## 500.8 共享内存（Shared Memory）

### 500.8.1 分配与访问

```python
smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
    vec=1, per_phase=1, max_phase=1, order=[0]
)
smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)
reg_tile = smem.load(reg_layout)   # GMEM layout → 寄存器 layout
smem.store(reg_tile)
```

**描述符方法**：`slice` / `index` / `permute` / `reshape` / `_reinterpret`

### 500.8.2 Shared Layout 类型

| 类型 | 场景 |
|------|------|
| `SwizzledSharedLayout` | 通用，减 bank conflict |
| `NVMMASharedLayout` | MMA/WGMMA 配套 |
| `PaddedSharedLayout` | 带 padding |
| `SharedLinearLayout` | 线性基表示 |

---

## 500.9 异步拷贝（03-async-copy）

**路径**：Global → Shared（`cp.async`），再 `smem.load` → 寄存器 → `store` 回 Global。

```python
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

@gluon.jit
def memcpy_1d_cpasync_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel

    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)

    cp.async_copy_global_to_shared(smem, in_ptr + offsets, mask=mask)
    cp.commit_group()
    cp.wait_group(0)

    value = smem.load(layout)
    gl.store(out_ptr + offsets, value, mask=mask)
```

**流水线模式**：`issue copy → 做计算 → wait_group` 重叠访存与算术。

**硬件**：`torch.cuda.get_device_capability()[0] >= 8`（Ampere+）。

---

## 500.10 TMA（04-tma，Hopper+）

**动机**：普通 `LDG/STG` 地址/掩码/结果占寄存器；TMA 用 **tensor descriptor** 压缩地址，走 async proxy。

**主机端**：`TensorDescriptor` 包装 shape/stride/block_shape/layout。

```python
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier

@gluon.jit
def memcpy_1d_tma_kernel(in_desc, out_desc, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    smem_layout: gl.constexpr = in_desc.layout
    smem = gl.allocate_shared_memory(in_desc.dtype, [XBLOCK], smem_layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(in_desc, [pid * XBLOCK], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    tma.async_copy_shared_to_global(out_desc, [pid * XBLOCK], smem)
    tma.store_wait(pendings=0)
```

**同步要点**：

- **mbarrier**：phase 0/1 交替；`init(count)` / `expect(bytes)` / `wait(phase)` / `invalidate`
- TMA store 完成用 `store_wait`，与 `cp.async` 的 commit group **分开**

**硬件**：SM ≥ 90（Hopper+）。

---

## 500.11 WGMMA（05-wgmma，Hopper）

```python
from triton.experimental.gluon.language.nvidia.hopper import (
    warpgroup_mma_init, warpgroup_mma, warpgroup_mma_wait,
)
```

**语义**：`d = a @ b + c`；`b` 必须经 shared；`c` 在寄存器；`a` 可寄存器或 shared。

**形状约束**（摘要）：

- 指令形状 `[m, n, k]`：`m=16`，`k = 256 / dtype_bits`，`n` 为 8 的倍数（有上限表）
- 至少 **4 warps** 组成一个 warp group；`warps_per_cta` 最小原子 `[4, 1]`
- `DotOperandLayout` + `BlockedLayout` 必须与 MMA 父 layout 一致

**dot_fma**（非 WGMMA 的通用 MMA）：

```python
acc = gl.dot_fma(a, b, acc)  # a/b 为 DotOperandLayout，acc 为 BlockedLayout
```

---

## 500.12 架构与特性矩阵

| 特性 | 最低 NVIDIA | 模块路径 |
|------|-------------|----------|
| 基础 Gluon | 通用 CUDA | `gluon.language` |
| `cp.async` | Ampere (sm80) | `language.nvidia.ampere.async_copy` |
| TMA / mbarrier | Hopper (sm90) | `language.nvidia.hopper` |
| WGMMA | Hopper | `language.nvidia.hopper` |
| TCGen05 / TMEM | Blackwell (sm100) | `language.nvidia.blackwell` |
| Warp 专化 | Hopper+ | `gl.warp_specialize` |
| AMD GFX1250 | CDNA | `language.amd.gfx1250` |

检测模板：

```python
def is_hopper_or_newer():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
```

---

## 500.13 Warp Specialization（08）

```python
results = gl.warp_specialize(
    [
        (default_fn, arg0, arg1),      # 默认分区：沿用父 num_warps
        (loader_fn, desc, bar),        # worker 分区 1
        (store_fn, desc, smem),        # worker 分区 2
    ],
    worker_num_warps=[1, 1],
    worker_num_regs=[24, 24],
)
```

**概念**：

- **partition**：不同 warp 组做不同任务（如专职 TMA load / MMA / store）
- 默认分区可返回 tensor；worker 分区在父 kernel 闲置时执行
- **代价**：同步、SMEM 通信、寄存器压力；**不支持递归** `warp_specialize`

---

## 500.14 语言 API 速查

### 500.14.1 张量创建

```python
gl.arange(start, end, layout=None)
gl.full(shape, value, dtype, layout=None)
gl.zeros(shape, dtype, layout=None)          # _standard
gl.zeros_like(x, layout=None)
```

### 500.14.2 内存与同步

```python
gl.load(ptr, mask=None, other=None)
gl.store(ptr, val, mask=None)
gl.allocate_shared_memory(dtype, shape, layout, value=None)
gl.thread_barrier()
gl.num_warps()
gl.num_ctas()
```

### 500.14.3 控制与元编程

```python
gl.program_id(axis=0)
gl.num_programs(axis=0)
gl.static_assert(cond, msg="")
gl.static_print(...)
gl.static_range(start, end)   # 编译期展开
pid: gl.constexpr = ...
layout: gl.constexpr = gl.BlockedLayout(...)
```

### 500.14.4 从 Triton 转发的主要算子

`add`, `sub`, `mul`, `where`, `reduce`, `broadcast`, `reshape`, `expand_dims`, `join`, `split`, `permute`, `gather`, `cast`, `atomic_*`, `maximum`, `minimum`, `associative_scan`, `inline_asm_elementwise`, `device_print`, `device_assert`, `assume`, `multiple_of`, `max_contiguous`, `max_constancy`

### 500.14.5 数学（`language._math`）

`exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt`, `sin`, `cos`, `abs`, `fma`, `erf`, `floor`, `ceil`, `umulhi`, ...

### 500.14.6 标准库（`language._standard`）

`cdiv`, `sum`, `max`, `min`, `ravel`, `reduce_or`, `xor_sum`

---

## 500.15 编写 Gluon 内核的推荐流程

```
1. 先用 Triton 写正确版本（逻辑参考）
2. 定 tile 形状 (BLOCK_M, BLOCK_N, ...)
3. 选 BlockedLayout / CoalescedLayout（1D memcpy 先扫 R）
4. 写 kernel：.arange / load / store + mask
5. 若带宽不够 → SwizzledSharedLayout + cp.async 流水线
6. 若 Hopper+ → TensorDescriptor + TMA + mbarrier
7. GEMM → DotOperandLayout + WGMMA/TMA 多层 pipeline
8. autotune BLOCK / num_warps / num_stages / layout 参数
9. 看 SASS：compiled.asm["sass"] 检查 LDG/STG 宽度
10. nsight / triton.testing.do_bench 验证
```

---

## 500.16 常见错误与排查

| 现象 | 可能原因 |
|------|----------|
| `tensor shape and layout rank mismatch` | `shape` 维数与 `BlockedLayout` 维数不一致 |
| `Did you forget to add @triton.gluon.jit` | 在 JIT 外调用了 builtin |
| 正确但极慢 | 标量循环或未指定 layout；R 过大导致 stall |
| SMEM bank conflict 高 | `SwizzledSharedLayout` 参数不当；用 `bank_conflicts` 预估 |
| TMA 死锁 | mbarrier phase 与 `wait` 不一致；忘记 `invalidate` |
| WGMMA 报错 | warps < 4；`DotOperandLayout` 与 acc layout 不匹配 |

**调试开关**：

```bash
TRITON_PRINT_AUTOTUNING=1
# 查看生成代码
compiled = kernel[grid](...)
print(compiled.asm["sass"])
```

---

## 500.17 与 Triton 互操作

- **Triton → Gluon 翻译**：`triton/python/triton/tools/triton_to_gluon_translater/`
- **同一 JIT 基础设施**：`constexpr`、`triton.Config`、`do_bench` 通用
- **项目实践**：工作区 `tune_a.py` 展示 `@gluon.jit` + `@triton.autotune` 的 attention 类内核

---

## 500.18 教程学习路线（建议 2–4 周）

| 周 | 教程 | 掌握目标 |
|----|------|----------|
| 1 | 01-intro, 02-layouts | jit、grid、constexpr、BlockedLayout、tile load/store、autotune |
| 2 | 03-async-copy | `allocate_shared_memory`、`cp.async`、软件流水线 |
| 3 | 04-tma, 05-wgmma | TensorDescriptor、mbarrier、TMA GEMM 雏形 |
| 4 | 06–08, 官方后续章节 | TCGen05、持久化 kernel、warp 专化、Multi-CTA |

**每章验收**：pytest 全绿 + `__main__` benchmark 达到教程量级带宽/TFLOPS。

---

## 500.19 官方文档索引（中文站）

| 编号 | 主题 | URL 路径（相对 triton-lang.cn） |
|------|------|--------------------------------|
| 01 | Gluon 简介 | `.../gluon/intro.html` |
| 02 | 张量布局 | `.../gluon/layouts.html` |
| 03 | 异步复制 | `.../gluon/async-copy.html` |
| 04 | TMA | `.../gluon/tma.html` |
| 05 | Warp-Group MMA | `.../gluon/wgmma.html` |
| 06 | 第五代 TensorCore | `.../gluon/tcgen05.html` |
| 07 | 持久化内核 | `.../gluon/persistence.html` |
| 08 | Warp 专门化 | `.../gluon/warp-specialization.html` |

---

## 500.20 速查：最小可运行模板

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def kernel(ptr, n, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    pid = gl.program_id(0)
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n
    x = gl.load(ptr + offs, mask=mask, other=0.0)
    gl.store(ptr + offs, x, mask=mask)

def launch(x, BLOCK=1024):
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](x, n, BLOCK, num_warps=4)

if __name__ == "__main__":
    x = torch.randn(1 << 20, device="cuda")
    launch(x)
    torch.cuda.synchronize()
```

---

## 500.21 本仓库相关文件

| 路径 | 说明 |
|------|------|
| `triton/python/tutorials/gluon/*.py` | 官方教程源码 |
| `triton/python/triton/experimental/gluon/language/_core.py` | 核心 builtin |
| `triton/python/triton/experimental/gluon/language/_layouts.py` | 全部 layout 定义 |
| `triton/python/triton/experimental/gluon/language/__init__.py` | 导出 API 列表 |
| `triton/python/test/gluon/` | 单元测试 |
| `tune_a.py` | Gluon + autotune 实战示例 |

---

*文档编号 500 起，与 `markdowns/100_llvm_basic.md`、`markdowns/00_jit_overview.md` 等系列并列，便于按编号检索。*
