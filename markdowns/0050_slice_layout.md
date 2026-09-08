# 050 SliceEncodingAttr 详解：Slice、Expand 与任意 Parent Layout

> **本文目标**：从**使用场景**出发，回答四个问题——**Slice 是怎么用的**、**任意 parent layout 下 slice 的语义是什么**、**反操作 expand_dims 如何理解**、**expand_dims 在不同维度上 LinearLayout 的 basis 怎么变**。
>
> 前置阅读：`010_layout.md` §6.3（Slice 全景简介）、`020_linear_layout.md` §5.7（Slice `toLinearLayout` 源码与手算）。本文是前两者的 **展开与深化**，用完整例子展示 bases 的变化。

---

## 0. 阅读路线

```
§1  一句话定义             → Slice 是什么、为什么需要它
§2  使用场景总览           → Triton 中何时自动/手动产生 Slice（含 Gluon 手动场景）
§3  IR 层面 Slice 的语义    → #ttg.slice 的 4 步算法
§4  Blocked Parent 下的完整算例  → 1D→1D、2D→1D、2D→2D
§5  Swizzle Shared Parent 下算例 → dim 不同位置的影响 + order 的影响
§6  反操作：expand_dims      → expand 在不同维度的 LinearLayout 语义
§7  Expand_dims 计算举例    → Blocked 和 Swizzle 在 dim=0/dim=1 的具体 basis 变化
§8  总结                    → 关键对照表
§9  更多 Parent Layout 类型  → Linear / NvidiaMma / Nested Slice / DotOperand 的转换
```

---

## §1 一句话定义

**`SliceEncodingAttr`（`#ttg.slice`）是一个 wrapper layout**——它不自己定义任何数据分布，而是对 **parent layout** 做 **dimension removal**（降维）：

```
语义：先在 dim 处插入一个 size=1 的轴 → 用 parent layout 分配 → 再 squeeze 掉该轴
```

```
用法上的直觉：
  parent = 2D BlockedLayout [M, N]
  Slice(dim=0)  → 保留 N 维的分布，去掉 M 维 → 得到沿 N 的 1D layout
  Slice(dim=1)  → 保留 M 维的分布，去掉 N 维 → 得到沿 M 的 1D layout
```

**为什么需要？**

| 场景 | 问题 | Slice 的作用 |
|------|------|-------------|
| 归约操作（`reduce`/`sum`/`argmin` 等）的结果 | 降维后需要保持 parent 在其他维的分布信息 | Slice 记录"被 squeeze 的 axis"和 parent layout |
| `expand_dims` 的反向推断 | expand 需要知道输入是从什么 layout 切出来的 | 编译器检查 slice dim 是否匹配 axis，匹配则直接解开 |
| rank-reducing load/store | load 结果少一个维但 thread 分布不变 | 用 Slice encoding 承载低 rank 的分布 |
| FP8 scale / TMA 索引 | 临时构造高维 layout 再 squeeze 回低维以满足硬件约束 | 用 Slice 连接高维 parent 和低维实际类型 |

**与 `020` §5.7 的分工**：

| 文档 | 侧重 |
|------|------|
| `020_linear_layout.md` §5.7 | 源码算法 + 手算例子，**编译器实现** |
| **本文** | **使用场景 + 任意 parent 的行为 + Expand 反操作 + 编译器自动产生的时机** |

---

## §2 使用场景总览：什么时候会产生 SliceEncodingAttr

SliceEncodingAttr 在 Triton 中**绝大多数情况是编译器自动产生的**，用户代码无需手动构造。以下按使用场景分类说明。

### 2.1 场景一：`tl.reduce` / `tl.sum` / `tl.argmin` 等沿维归约的结果

**这是最核心、最频繁的产生场景。**

```python
# Triton Python 代码
sum_val = tl.sum(tensor_2d, axis=0)  # shape [M, K] → [K]
```

归约操作去掉了一个维（axis=0），编译器需要为结果 `tensor<K>` 推导 layout。它把输入 `tensor_2d` 的 2D layout **包一层 `SliceEncodingAttr(dim=0, parent=input_layout)`**，表示"结果的 1D layout 是 parent 沿 dim0 切掉后剩下的分布"。

```
输入  tensor_2d:  tensor<MxK, #blocked<...>>
                    │
                    ▼ reduce sum(axis=0)
输出  sum_val:     tensor<K, #slice<dim=0, parent=#blocked<...>>>
```

同理适用于 `tl.mean`、`tl.var`、`tl.argmin`、`tl.argmax`、`tl.min`、`tl.max` ——**所有沿维归约的操作**。

**代码对应**：`lib/Dialect/TritonGPU/Transforms/Utility.cpp` 中 `inferDstEncoding` 对 `ReduceOp` 的处理：

```cpp
// Utility.cpp:315–318
return triton::gpu::SliceEncodingAttr::get(
    op->getContext(), op.getAxis(), srcEncoding);
```

**`SliceLayout` 构造等价**：在 Gluon 前端中，可以用 `SliceLayout` 显式构造等价的 layout 对象：

```python
from triton.experimental.gluon import language as gl

# 输入 tensor [M, K] 的 2D blocked layout
blocked_2d = gl.BlockedLayout(
    size_per_thread=[1, 4],
    threads_per_warp=[4, 8],
    warps_per_cta=[4, 1],
    order=[1, 0],
)

# reduce sum(axis=0) 的结果 layout = 去掉 dim0 后的 slice
# 等价于编译器自动推导的 #slice<dim=0, parent=#blocked2d>
reduce_result_layout = gl.SliceLayout(dim=0, parent=blocked_2d)
# 此时 layout.rank == 1，表示保留 dim1（K 维）的分布

# reduce sum(axis=1) 的结果 layout = 去掉 dim1 后的 slice
reduce_result_layout_axis1 = gl.SliceLayout(dim=1, parent=blocked_2d)
# 此时 layout.rank == 1，表示保留 dim0（M 维）的分布
```

### 2.2 场景二：`OptimizeThreadLocality` pass——减少跨线程通信

这个 pass **不是 loop unroll**，而是专门为 **`scf.for` 循环体中的归约操作**做的数据排布优化，目的是**最小化归约时的跨线程通信**。

**做什么**：当 `scf.for` 循环体内有 `tl.reduce`，且累加器初始化为常数（如 `0.0`）时，该 pass 把归约数据从 2D reshape 为 3D，使得**沿归约维的元素全部集中到同一个 thread 内**（thread-local），从而消除循环体中的跨线程通信——只在循环结束后做一次 post-loop reduction 来合并各 thread 的结果。

```
优化前（循环体里跨线程通信）：         优化后（循环体内 thread-local）：
for i in range(N):                    for i in range(N):
    acc = reduce_sum_2d(data)              acc_tmp = reduce_local_3d(data)  ← 只在一个 thread 内
                                          # 循环结束后
    result = acc + ...                     result = reduce_cross_thread(acc_tmp)  ← 只做一次
```

**Slice 出现在哪里**：pass 构造了一个 3D `BlockedEncodingAttr`（`blocked3d`），然后在第 321 行：

```cpp
auto slice2d = triton::gpu::SliceEncodingAttr::get(
    mod.getContext(), rank, blocked3d);
```

这里的 `rank` = 原始 rank（如 2），`blocked3d` = 3D reshaped layout。**`SliceEncodingAttr(rank=2, parent=blocked3d)` 的含义是"从 3D layout 中切掉 dim=2，得到一个 2D 结果"**，用这个 slice encoding 来承载循环累加器的类型。

因此 dump TTGIR 时会在循环累加器上看到 `#ttg.slice`——**它不是 loop unroll 的产物，而是"reshape → reduce → slice"这个优化模式的中间类型**。

#### 直观的 IR 变换示例

假设原始 2D 归约：tensor 32×32，沿 axis=1 求和。使用常见的 coalescing 友好 layout：

```mlir
// sizePerThread = [1, 4]   → 每 thread 在 dim1 上有 4 个元素
// threadsPerWarp = [4, 8]  → 8 个 thread 分布在 dim1 上 → 跨线程通信
// warpsPerCTA   = [4, 1]
// order = [1, 0]           → dim1 最快
// tile = [16, 32]，32×32 的 tensor: dim0 wrap 2×, dim1 正好铺满
#blocked2d = #ttg.blocked<{sizePerThread = [1, 4],
    threadsPerWarp = [4, 8], warpsPerCTA = [4, 1],
    order = [1, 0]}>
```

原始循环体（每轮迭代都要 shuffle）：

```mlir
// 归约结果：沿 axis=1 去掉 dim1 → 1D，用 slice encoding 记录 parent
%init = arith.constant dense<0.0> : tensor<32xf32, #slice<dim=1, parent=#blocked2d>>
%result = scf.for %i = %c0 to %cN step %c1 iter_args(%acc = %init) -> tensor<32xf32, #...> {
  %data = tt.load %ptr[%i] : tensor<32x32xf32, #blocked2d>
  // dim1 上 8 个 thread × 每 thread 4 个元素 = 32 个元素
  // 8 个 thread 需要 shuffle 才能合并 → 每轮都做
  %sum = tt.reduce %data axis=1 -> tensor<32xf32, #slice<dim=1, parent=#blocked2d>>
    attrs = {op = "add"}
  %new_acc = arith.addf %acc, %sum : tensor<32xf32, #...>
  scf.yield %new_acc : tensor<32xf32, #...>
}
```

**OptimizeThreadLocality pass 的变换（基于源码 `getThreadLocalityOptimizedShape` / `getThreadLocalityOptimizedEncoding`）**：

```
原始 2D:      shape=[32, 32],  elemsPerThread[1]=4
reshape 3D:  shape=[32, 8, 4]   (dim1 = 32÷4 = 8, dim2 = 4)
效果: dim2 的 4 个元素全部在同一个 thread 内 → reduce 无需通信
```

```mlir
// pass 按以下逻辑构造 3D blocked encoding:
//   sizePerThread   = insert([1,4],  2, 4) → [1,1,4]
//   threadsPerWarp  = insert([4,8],  2, 1) → [4,8,1]
//   warpsPerCTA     = insert([4,1],  2, 1) → [4,1,1]
//   order           = insert([1,0],  0, 2) → [2,1,0]  (dim2 最快)
// tile3d = [16, 8, 4], tensor [32,8,4]: dim0 wrap 2×, dim1/2 正好铺满
#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 4],
    threadsPerWarp = [4, 8, 1], warpsPerCTA = [4, 1, 1],
    order = [2, 1, 0]}>

// slice encoding: "从 3D layout 切掉 dim=2，得到一个 2D [32,8] 的布局"
// 用于承载循环累加器的类型（源码 321 行）
#slice2d = #ttg.slice<dim = 2, parent = #blocked3d>

// 新的循环 — 累加器 shape=[32,8], encoding=#slice2d
%new_init = arith.constant dense<0.0> : tensor<32x8xf32, #slice2d>
%new_loop = scf.for %i = %c0 to %cN step %c1 iter_args(%acc3d = %new_init)
    -> tensor<32x8xf32, #slice2d> {
  %data = tt.load %ptr[%i] : tensor<32x32xf32, #blocked2d>

  // tt.view + allow_reorder: blocked2d → blocked3d，零成本（不产生指令）
  %data3d = tt.view %data {allow_reorder = true}
      : tensor<32x32xf32, #blocked2d> -> tensor<32x8x4xf32, #blocked3d>

  // 沿 axis=2 归约 → 4 个元素全在同一个 thread 内 → 无需 shuffle
  // 结果被 slice 掉 dim2 → 回到 2D shape=[32,8]
  %local_sum = tt.reduce %data3d axis=2
      -> tensor<32x8xf32, #slice2d>
    attrs = {op = "add"}
  %new_acc3d = arith.addf %acc3d, %local_sum : tensor<32x8xf32, #slice2d>
  scf.yield %new_acc3d : tensor<32x8xf32, #slice2d>
}

// 循环结束后：做一次跨线程归约合并 8 个 thread 的 dim1 结果
%final_2d = tt.reduce %new_loop axis=1
    -> tensor<32xf32, #slice<dim=1, parent=#blocked2d>>
  attrs = {op = "add"}

// 转回原始 layout
%final = tt.convert_layout %final_2d : tensor<32xf32, #...>
```

**关键变化**：

| 项目 | 优化前 | 优化后 |
|------|--------|--------|
| 循环内通信 | 每轮 shuffle（8 thread 沿 dim1） | **无通信**（axis=2 归约全在 thread 内） |
| 通信开销 | N 轮 × 每次 warp shuffle | 只在循环结束后做一次 post-loop reduce |
| 累加器 encoding | `#slice<dim=1, parent=#blocked2d>` | `#slice<dim=2, parent=#blocked3d>` |
| reshape view | 无 | `tt.view` + `allow_reorder`（不产生指令） |
| 代价 | — | 循环内布局与最终布局不同，多一次 `convert_layout` |

**`SliceLayout` 构造等价**：

```python
from triton.experimental.gluon import language as gl

# 原始 2D blocked layout
blocked_2d = gl.BlockedLayout(
    size_per_thread=[1, 4],
    threads_per_warp=[4, 8],
    warps_per_cta=[4, 1],
    order=[1, 0],
)

# pass 构造的 3D reshaped layout：dim1 拆分 [8, 4]，dim2 的 4 个元素归到同一 thread
blocked_3d = gl.BlockedLayout(
    size_per_thread=[1, 1, 4],
    threads_per_warp=[4, 8, 1],
    warps_per_cta=[4, 1, 1],
    order=[2, 1, 0],   # dim2 最快
)

# 累加器 encoding：从 3D layout 切掉 dim=2
loop_acc_layout = gl.SliceLayout(dim=2, parent=blocked_3d)
# loop_acc_layout.rank == 2

# post-loop reduce
post_loop_layout = gl.SliceLayout(dim=1, parent=blocked_2d)
```

### 2.3 场景三：`tl.expand_dims` 的反向推断

当编译器遇到 `expand_dims(x, axis)` 时，它会检查输入 `x` 是否带有 `SliceEncodingAttr`。如果是且 `axis == slice_dim`，就把 slice 解开、**直接返回 parent layout**。

```
tensor<K, #slice<dim=0, parent=#blocked<...>>>
    │
    ▼ expand_dims(axis=0)
tensor<1xK, #blocked<...>>   ← slice 被消解，恢复 parent
```

**编译器强制要求**（`Dialect.cpp:2705-2713`）：
> `ExpandDimsOp` 的输入 **必须** 带有 `SliceEncodingAttr`，且 axis 必须与 slice 的 dim 一致，否则报错。

**`SliceLayout` 构造等价**：

```python
from triton.experimental.gluon import language as gl

# 假设 reduce(axis=0) 的结果 layout
blocked_2d = gl.BlockedLayout(
    size_per_thread=[1, 4],
    threads_per_warp=[4, 8],
    warps_per_cta=[4, 1],
    order=[1, 0],
)
reduce_layout = gl.SliceLayout(dim=0, parent=blocked_2d)  # rank=1

# expand_dims(axis=0) 把 slice 消解掉，恢复 parent layout
#   tensor<1xK, blocked_2d>  ← 等价于编译器自动推断
# 在 Gluon 前端中无需手动写 expand_dims，编译器自动处理
# 但如果你手动构造，gl.SliceLayout(dim=0, parent=...) 这个关系
# 会自然地被 expand_dims 推断为 parent
```

**轴匹配规则的完整推导**：下面以 `tl.reduce` + `[:, None]` / `[None, :]` 的实际场景来验证。

```python
import triton.language as tl

# 2D tensor [M, N], layout = #blocked2d
tensor_2d = tl.load(...)

# ── 场景 A：reduce(axis=1) ──
sum_val = tl.sum(tensor_2d, axis=1)   # [M], #slice<dim=1, parent=#blocked2d>
result = sum_val[:, None]              # [M, 1]  expand axis=1 == slice dim=1 → ✅ 通过
# result = sum_val[None, :]            # [1, M]  expand axis=0 ≠ slice dim=1 → ❌ 拒绝

# ── 场景 B：reduce(axis=0) ──
sum_val = tl.sum(tensor_2d, axis=0)   # [N], #slice<dim=0, parent=#blocked2d>
result = sum_val[None, :]              # [1, N]  expand axis=0 == slice dim=0 → ✅ 通过
# result = sum_val[:, None]            # [N, 1]  expand axis=1 ≠ slice dim=0 → ❌ 拒绝
```

关键规则总结：

> **只能在被 squeeze 的维上 expand 回去。**

用 `[:, None]` vs `[None, :]` 来理解：

| `tl.reduce` 的 axis | slice dim | `[:, None]` (expand axis=1) | `[None, :]` (expand axis=0) |
|-------------------|-----------|-----------------------------|-----------------------------|
| `axis=0` | `dim=0` | ❌ axis 1 ≠ dim 0 | ✅ axis 0 == dim 0 |
| `axis=1` | `dim=1` | ✅ axis 1 == dim 1 | ❌ axis 0 ≠ dim 1 |

**测试验证**：四个组合均已通过实际编译测试：

```
  ✅ reduce(axis=1) + [:, None]   (dim=1 == axis=1): 编译 + 运行成功
  ✅ reduce(axis=0) + [None, :]   (dim=0 == axis=0): 编译 + 运行成功
  ❌ reduce(axis=1) + [None, :]   (dim=1 ≠ axis=0): 编译失败（layout 推断拒绝）
  ❌ reduce(axis=0) + [:, None]   (dim=0 ≠ axis=1): 编译失败（layout 推断拒绝）
```

### 2.4 场景四：FP8 scale tensor 升维（`DecomposeScaledBlocked` pass）

MXFP8/FP8 量化 matmul 中，scale factor 是一个 1D tensor。为了把它 broadcast 到 2D 与矩阵乘操作数对齐，编译器自动构造一个 2D parent layout，然后包上 `SliceEncodingAttr`，再配合 `expand_dims` 完成升维。

用户代码无感知，但 dump IR 可以看到：

```mlir
// scale tensor 在 pass 中
tensor<K, #slice<dim=1, parent=#blocked<{sizePerThread=[1,1], ...}>>>
```

**`SliceLayout` 构造等价**：编译器在 pass 内部会构造如下 layout：

```python
from triton.experimental.gluon import language as gl

# 构造 2D parent layout（M 为前向广播维，K 为 scale 维）
# sizePerThread=[1, 1] 表示每维每 thread 只持 1 个元素
blocked_2d = gl.BlockedLayout(
    size_per_thread=[1, 1],
    threads_per_warp=[4, 8],
    warps_per_cta=[4, 1],
    order=[1, 0],
)

# 包一层 slice，去掉前向广播维（dim=0），保留 K 维的分布
# 等价于 dump IR 中的 #slice<dim=1, parent=#blocked<...>>
# 注意：dim=1 表示去掉 parent 的 dim1（列维），保留 dim0（行维）
# 此处 dim=1 是因为 ir dump 用的是 parent 坐标系
scale_layout = gl.SliceLayout(dim=1, parent=blocked_2d)
# scale_layout.rank == 1（1D scale tensor）

# 后续 expand_dims 时 slice 被消解，恢复 2D parent
```

### 2.5 场景五：TMA gather4/scatter4 索引张量（`TritonGPUConversion`）

TMA（Tensor Memory Accelerator）的 `gather4`/`scatter4` 指令要求每 thread 持有 **4 个连续索引元素**。编译器先构造一个 2D layout（把 4 元素分组放到内维），然后用 `SliceEncodingAttr(dim=0)` squeeze 回 1D，既满足硬件约束又保持类型正确。

```cpp
// TritonGPUConversion.cpp:150
auto newEncoding = SliceEncodingAttr::get(ctx, /*dim=*/0, parentEncoding);
```

**`SliceLayout` 构造等价**：

```python
from triton.experimental.gluon import language as gl

# 构造 2D parent layout：dim1 上分配 4 个连续元素（gather4 粒度）
# [N, 4] 其中 N = 索引总数 / 4
parent_2d = gl.BlockedLayout(
    size_per_thread=[1, 4],     # 每 thread 在 dim1 上有 4 个元素
    threads_per_warp=[8, 1],
    warps_per_cta=[4, 1],
    order=[1, 0],               # dim1 连续
)

# 用 SliceEncodingAttr(dim=0) squeeze 回 1D
# 等价于源码 newEncoding = SliceEncodingAttr::get(ctx, 0, parentEncoding)
tma_idx_layout = gl.SliceLayout(dim=0, parent=parent_2d)
# tma_idx_layout.rank == 1，但每个 thread 仍持有 4 个连续索引元素
```

### 2.6 场景六：`tl.squeeze` 的结果

当用户手动 squeeze 一个 size=1 的维时，编译器也为结果推导出 SliceEncodingAttr：

```python
# x.shape = [M, 1, K]
x_sq = tl.squeeze(x, axis=1)  # [M, K]
# x_sq 的 encoding = #slice<dim=1, parent=x.layout>
```

**`SliceLayout` 构造等价**：

```python
from triton.experimental.gluon import language as gl

# 假设 x 的 3D blocked layout
blocked_3d = gl.BlockedLayout(
    size_per_thread=[1, 1, 4],
    threads_per_warp=[4, 1, 8],
    warps_per_cta=[1, 1, 4],
    order=[2, 0, 1],
)

# squeeze(axis=1) → 去掉中间维，保留 dim0/dim2 的分布
squeezed_layout = gl.SliceLayout(dim=1, parent=blocked_3d)
# squeezed_layout.rank == 2（原 rank - 1）
# 第 1 维被 squeeze 掉后，dim0 和 dim2 重新编号为 dim0/dim1
```

### 2.7 场景七（用户手动写）：Gluon 2D kernel 中的 1D 索引

> 这是**唯一需要用户手动构造 `SliceLayout`** 的场景，仅出现在 Gluon 编程模型中。

在 Triton 的原生 `tl` 接口中，用户不需要手动创建 SliceEncodingAttr。但在 **Gluon**（`triton.language.extra.gluon` / `gl`）中，2D kernel 需要两个 1D 的 `arange` 来分别索引行和列，而 layout 是 2D 的，所以必须用 `SliceLayout(dim, parent)` 从 2D layout 中**取出一维的投影**。

```python
@gluon.jit
def memcpy_2d_kernel(in_ptr, out_ptr, xnumel, ynumel,
                     XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,
                     layout: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)
    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK

    # 从 2D layout 取出行的 1D 投影
    indices_x = start_x + gl.arange(0, XBLOCK,
                    layout=gl.SliceLayout(dim=1, parent=layout))

    # 从 2D layout 取出列的 1D 投影
    indices_y = start_y + gl.arange(0, YBLOCK,
                    layout=gl.SliceLayout(dim=0, parent=layout))

    # broadcast 回 2D — 零成本
    in_offsets = (indices_x[:, None] * stride_in_x +
                  indices_y[None, :] * stride_in_y)
    value = gl.load(in_ptr + in_offsets)
    gl.store(out_ptr + out_offsets, value, mask=mask)
```

**等价的手动构造方式**：

```python
from triton.experimental.gluon import language as gl

# 定义一个 2D blocked layout（例如 XBLOCK=128, YBLOCK=64）
layout_2d = gl.BlockedLayout(
    size_per_thread=[1, 4],
    threads_per_warp=[4, 8],
    warps_per_cta=[4, 1],
    order=[1, 0],
)

# 从 2D layout 取"行方向"的 1D 投影：
#   dim=1: 去掉 parent 的列维（dim1），保留行维（dim0）的分布
row_slice = gl.SliceLayout(dim=1, parent=layout_2d)
# row_slice.rank == 1，对应行索引 indices_x

# 从 2D layout 取"列方向"的 1D 投影：
#   dim=0: 去掉 parent 的行维（dim0），保留列维（dim1）的分布
col_slice = gl.SliceLayout(dim=0, parent=layout_2d)
# col_slice.rank == 1，对应列索引 indices_y

# 在 gl.arange 中传入这些 layout：
indices_x = gl.arange(0, 128, layout=row_slice)   # 1D 行索引
indices_y = gl.arange(0, 64,  layout=col_slice)    # 1D 列索引
```

**关键理解**：

| `SliceLayout` 写法 | 含义 |
|--------------------|------|
| `SliceLayout(dim=1, parent=2D_layout)` | 去掉 parent 的 dim1（列维），保留 dim0（行维）的分布 |
| `SliceLayout(dim=0, parent=2D_layout)` | 去掉 parent 的 dim0（行维），保留 dim1（列维）的分布 |

**为什么不用搞 Slice？** 因为 `gl.arange(0, XBLOCK, ...)` 产生的是 **1D tensor**，但 parent layout 描述的是一个 2D `[XBLOCK, YBLOCK]` tile 的 thread 分工。没有 SliceLayout 的话，编译器不知道这个 1D arange 应该按照 parent 的哪个维度来分配线程。SliceLayout 的作用就是**在 2D parent 中精确指定「取哪个维的投影」**。

**如何零成本回到 2D？** 两个 1D 索引通过 `[:, None]` / `[None, :]` broadcast 回 2D 时，编译器自动做 `expand_dims` 的逆操作——把 SliceLayout 消解掉、恢复 parent 2D layout。整个过程**不产生额外指令**。

### 2.8 使用场景总览表

| 场景 | 触发方式 | 产生位置 | 用户是否手动写 | slice 的 dim 指向谁 |
|------|----------|----------|--------------|-------------------|
| `tl.reduce` / `tl.sum` 等归约 | **自动** | Layout inference | ❌ | 被归约的 axis |
| 循环内归约优化 | **自动** | `OptimizeThreadLocality` | ❌ | 被优化提升的维 |
| `tl.expand_dims` 反推 | **自动** | Layout inference | ❌ | 被 expand 的 axis（消解用） |
| FP8 scale 升维 | **自动** | `DecomposeScaledBlocked` | ❌ | 被广播的维 |
| TMA gather/scatter 索引 | **自动** | `TritonGPUConversion` | ❌ | 额外的分组维 |
| `tl.squeeze` 结果 | **自动** | Layout inference | ❌ | 被 squeeze 的 axis |
| Gluon 2D kernel 的 1D arange | **手动** | 用户代码（`gl.arange` 时传参） | ✅ `SliceLayout(dim, parent)` | 要保留的维 |

> 核心结论：在原生 Triton（`tl` 接口）编码中，**用户几乎从不需要手动创建 SliceEncodingAttr**。编译器会在 reduce、expand_dims、squeeze 等操作上自动处理好。**唯一的例外是 Gluon**——2D kernel 中生成 1D 行/列索引时，需显式传 `SliceLayout(dim=..., parent=...)` 给 `gl.arange`。

---

## §3 IR 层面 Slice 的语义

### 3.1 四步算法

`SliceEncodingAttr::toLinearLayout(shape)` 的 4 步（`020` §5.7）：

```cpp
// Step ①: 在 child shape 的 dim 处插入 size=1 → 得到 parent shape
SmallVector<int64_t> parentShape(shape);
parentShape.insert(parentShape.begin() + getDim(), 1);

// Step ②: 用 parent shape 算 parent 的 LinearLayout
LinearLayout parentLL = toLinearLayout(parentShape, getParent());

// Step ③: 从 parent LL 中去掉 getDim() 这个输出维 → 降 rank
auto sliceLL = removeStandardDim(parentLL, getDim());

// Step ④: 清理 register 维中全零的 basis（被 squeeze 的维废掉的 bit）
auto bases = sliceLL.getBases();
for (const auto &basis : bases["register"]) {
    if (any_of(basis, [](int b) { return b != 0; }))
        newRegBases.push_back(basis);
}
bases["register"] = newRegBases;
return LinearLayout(bases, sliceLL.getOutDimNames());
```

### 3.2 关键辅助函数 `removeStandardDim`

```cpp
LinearLayout removeStandardDim(const LinearLayout &layout, int dim) {
    // 去掉 dim 对应的输出维名
    auto dims = layout.getOutDimNames();
    dims.erase(dims.begin() + dim);
    // sublayout：只保留 dims 里的输出维（去掉 dim 那一列）
    auto newLayout = layout.sublayout(layout.getInDimNames(), dims);
    // 重命名剩下的输出维为 dim0, dim1, ...
    renamedDims = [0, 1, ..., rank-1];
    return LinearLayout(newLayout.getBases(), renamedDims, false);
}
```

**核心操作**：`removeStandardDim` 做的事情是——遍历 basis 中每一行（每个输入 bit 对应的输出偏移向量），**删掉第 `dim` 列**，然后把剩下的列重编号。

### 3.3 整体理解

```
Child shape [4, 8], dim=0
                    │ Step ① parentShape = [1, 4, 8]
                    │ Step ② parentLL (输出维 [dim0, dim1, dim2])
                    │         register bits → [a0, a1, a2]
                    │         lane bits     → [b0, b1, b2]
                    │         warp bits     → [c0, c1, c2]
                    │
                    ▼ Step ③ removeStandardDim(parentLL, 0)
            register bits → [a1, a2]  ← 删掉第 0 列 a0
            lane bits     → [b1, b2]  ← 删掉第 0 列 b0
            warp bits     → [c1, c2]  ← 删掉第 0 列 c0
                    │
                    ▼ Step ④ 清理全零 register bit
            Child LL (输出维 [dim0, dim1] = 原来 [dim1, dim2])
```

---

## §4 Blocked Parent 下的完整算例

### 4.1 2D→1D：parent 为 blocked `[M, N]`，Slice `dim=0`

**参数**：

| 项目 | 值 |
|------|-----|
| child shape | `[16]` |
| `dim` | 0 |
| parent | `BlockedLayout{sizePerThread=[1,2], threadsPerWarp=[1,4], warpsPerCTA=[1,1], order=[1,0]}` |

**Step ①**：parentShape = `[1, 16]`

**Step ②**：parent LL（输出维 `[dim0, dim1]`）

`identityStandardND` 按 `order=[1,0]` 分配 bit，dim1（size=16）先得：

| 输入维 | bit | basis `[dim0, dim1]` | 含义 |
|--------|-----|---------------------|------|
| register | 0 | `[0, 1]` | dim1 += 1（`sizePerThread[1]=2` 的贡献） |
| register | 1 | `[0, 8]` | dim1 wrap bit（`ensureLayoutNotSmallerThan` 补的） |
| lane | 0 | `[0, 2]` | dim1 += 2 |
| lane | 1 | `[0, 4]` | dim1 += 4 |

dim0 列全为 0（因为 dim0 size=1，没有 basis 分配给它）。

**Step ③**：`removeStandardDim(parentLL, 0)` → 删 dim0 列，dim1 重命名为 dim0

```
register bit 0 → [1]    // 从 [0,1] 删掉 0
register bit 1 → [8]    // 从 [0,8] 删掉 0
lane bit 0     → [2]    // 从 [0,2] 删掉 0
lane bit 1     → [4]    // 从 [0,4] 删掉 0
```

**Step ④**：无全零 register basis，不变。

**最终**：一个 1D Layout，等价于直接构造 `BlockedLayout{sizePerThread=[2], threadsPerWarp=[4], order=[0]}` 在 `shape=[16]` 的结果。

### 4.2 2D→2D：parent 为 blocked `[M, N, K]`，Slice `dim=1`

**参数**：

| 项目 | 值 |
|------|-----|
| child shape | `[4, 8]` |
| `dim` | 1 |
| parent | `BlockedLayout{sizePerThread=[1,1,2], threadsPerWarp=[1,32,1], warpsPerCTA=[1,1,1], order=[2,1,0]}` |

**Step ①**：parentShape = `[4, 1, 8]`

**Step ②**：parent LL（输出维 `[dim0, dim1, dim2]`）

`identityStandardND` 按 `order=[2,1,0]` 从最快维开始分配 bit：

| 输入维 | bit | basis `[dim0, dim1, dim2]` | 来历 |
|--------|-----|---------------------------|------|
| register | 0 | `[0, 0, 1]` | `sizePerThread[2]=2` → dim2 最快要 1 bit |
| register | 1 | `[0, 0, 2]` | `ensureLayoutNotSmallerThan` 补的：dim2 size=8 需要 3 bit（已有 1） |
| register | 2 | `[0, 0, 4]` | 同上，dim2 第 3 bit |
| register | 3 | `[1, 0, 0]` | `ensureLayoutNotSmallerThan` 补的：dim0 size=4 需要 2 bit |
| register | 4 | `[2, 0, 0]` | 同上，dim0 第 2 bit |
| lane | 0 | `[0, 1, 0]` | `threadsPerWarp[1]=32` → 按 order 下一个是 dim1，5 bit |
| lane | 1 | `[0, 2, 0]` | 同上，dim1 第 2 bit |
| lane | 2 | `[0, 4, 0]` | 同上，dim1 第 3 bit |
| lane | 3 | `[0, 8, 0]` | 同上，dim1 第 4 bit |
| lane | 4 | `[0, 16, 0]` | 同上，dim1 第 5 bit |

> ⚠️ **注意**：`identityStandardND` 严格按照 `order` 分配 bit，lane 的 5 bit 分配到 **dim1** 而非 dim0！但 parent shape[1]=1，dim1 的任何 offset 都 ≡ 0（mod 1），所以这些 lane bit 虽然存在但没有实际区分度——32 个线程在 dim1 方向只能映射到 1 个位置。所以 **dim1 的 basis 列全是 0**。

**Step ③**：`removeStandardDim(parentLL, 1)` → 删 dim1（中间列）

```
register bit 0 → [0, 1]    // 从 [0,0,1] 删掉中间 0
register bit 1 → [0, 2]    // 从 [0,0,2] 删掉中间 0
register bit 2 → [0, 4]    // 从 [0,0,4] 删掉中间 0
register bit 3 → [1, 0]    // 从 [1,0,0] 删掉中间 0
register bit 4 → [2, 0]    // 从 [2,0,0] 删掉中间 0
lane bit 0     → [0, 0]    // 从 [0,1,0] 删掉中间 1 → 全零
lane bit 1..4  → [0,0]     // 全部全零
```

重命名：dim0→dim0, dim2→dim1

**Step ④**：清理全零 register basis——lane 的 5 个 bit 全是 `[0,0]`（因为本来就在 dim1 上，dim1 被删掉后列没了，值全 0），应全部删除。register 5 个 bit 全部非零，保留。

**最终**：

```
register: [[0, 1], [0, 2], [0, 4], [1, 0], [2, 0]]
lane:      []   ← 全部被清理，因为 dim1 列的 bit 被删后只剩全零
```

可见 Slice dim=1 恰好切掉了 `threadsPerWarp` 唯一的活跃维（dim1），导致 lane 级别的所有 bit 成为 dead code。

#### 这个 layout 合理吗？——不合理，这是反模式

`BlockedLayout{sizePerThread=[1,1,2], threadsPerWarp=[1,32,1], warpsPerCTA=[1,1,1], order=[2,1,0]}` 配 parent shape `[4,1,8]` 在**数学上有效**（能算出 LinearLayout basis），但实践中**不合理**：

| 问题 | 说明 |
|------|------|
| `threadsPerWarp[1]=32` 但 shape[1]=1 | 5 个 lane bit 全部浪费——32 线程在 size=1 的维上没有区分度 |
| `warpsPerCTA=[1,1,1]` | 仅 1 warp（32 线程），GPU 并行度极低 |
| `sizePerThread[0]=1` 但 shape[0]=4 | 需要 `ensureLayoutNotSmallerThan` 额外补 2 个 register bit 来覆盖 |
| `sizePerThread[2]=2` 但 shape[2]=8 | 也需要补 2 个 register bit |

**合理的 layout 应让参数与 shape 匹配**，例如 parent shape `[4, 8, 16]` 且各维 size>1 时：

```python
layout = gl.BlockedLayout(
    size_per_thread=[1, 1, 4],    # dim2 每线程 4 元素
    threads_per_warp=[4, 2, 4],   # 线程分布与各维匹配
    warps_per_cta=[1, 4, 1],      # 4 warp 沿 dim1
    order=[2, 1, 0],
)
```

> 文档中故意选了这个退化 layout 来演示：**即使 parent layout 参数配得不好，Slice 的 4 步算法仍然可以正确计算**。这展示了算法的鲁棒性，但**不建议在实际 kernel 中使用这种参数配比**。

---

## §5 Swizzle Shared Parent 下算例

### 5.1 2D→2D：swizzle shared，Slice `dim=0`

**参数**：

| 项目 | 值 |
|------|-----|
| child shape | `[128, 64]` |
| `dim` | 0 |
| parent | `SwizzledSharedEncodingAttr{vec=4, perPhase=4, maxPhase=8, order=[1,0]}` |

**Step ①**：parentShape = `[1, 128, 64]`

**Step ②**：parent LL（输出维 `[dim0, dim1, dim2]`）

Swizzle shared 的 LL 是手写构造的，结果类似：

| 输入维 | bit | basis `[dim0, dim1, dim2]` |
|--------|-----|---------------------------|
| offset | 0 | `[0, 0, 1]` |
| offset | 1 | `[0, 0, 2]` |
| offset | 2 | `[0, 0, 4]` |
| offset | 3 | `[0, 0, 8]` |
| offset | 4 | `[0, 0, 16]` |
| offset | 5 | `[0, 0, 32]` |
| offset | 6 | `[0, 1, 0]` ← swizzle：翻 offset bit 6 同时改变 dim1（行）和 dim2（列） |
| offset | 7 | `[0, 2, 0]` |
| offset | 8 | `[0, 4, 0]` |
| offset | 9 | `[0, 8, 0]` |
| offset | 10 | `[0, 16, 0]` |
| offset | 11 | `[0, 32, 0]` |
| offset | 12 | `[0, 64, 0]` |

dim0（插入的 size=1 维）列全为 0。

**Step ③**：`removeStandardDim(parentLL, 0)` → 删 dim0 列

```
offset bit 0  → [0, 1]     // 保留 dim1(=0), dim2(=1)
offset bit 1  → [0, 2]
...（dim2 部分不变）
offset bit 6  → [1, 0]     // swizzle 行贡献保留
offset bit 7  → [2, 0]
...（dim1 部分不变）
offset bit 12 → [64, 0]
```

重命名：dim1→dim0, dim2→dim1

**Step ④**：无 register 维（shared 没有 register 输入维），跳过。

**最终**：一个 `[128, 64]` 的普通 swizzle shared LL，**完全不受 slice 影响**。

### 5.2 何时受 slice 影响？

**核心规律**：当且仅当 parent 的 basis 中 **某列对应被 slice 的维有非零值**，slice 才会改变剩余 basis。否则只是删除一列零。

| parent 中被 slice 的维大小 | dim 对应列是否可能非零 | 对剩余 basis 的影响 |
|---------------------------|----------------------|-------------------|
| =1（插入的哑维） | 全为 0（或 `ensureLayoutNotLargerThan` 已清零） | 几乎无影响 |
| >1（parent 正常维） | 有非零值 | 删除后剩余 basis 的数值不变但排列改变 |

**何时 register 会被清理（Step ④）**：

```
parent 中 dim 维的 size=1
    但 order 导致该维先分到了 register bit（如 order=[0,1] 且 dim=0）
    → register bit 映射到被 slice 的维
    → ensureLayoutNotLargerThan 清零
    → removeStandardDim 后该 basis 全零
    → Step ④ 删除 → register 占用减少
```

**示例**：child `shape=[16]`，`dim=0`，parent 的 `sizePerThread=[2,2], order=[0,1]`：

```
order=[0,1] → dim0(fast) 先分到 register bit0 → [1, 0]
parent dim0 size=1 → ensureLayoutNotLargerThan 置为 [0, 0]
removeStandardDim(parentLL, 0) 后该 basis → [0]（全零）
Step ④ 删除 → 最终 register 少了一个 bit
```

### 5.3 Order 如何影响 Slice 结果（深入）

> **一句话**：order 不改变 Slice 的 4 步算法，但改变 Step ② 中 parent LL 的 basis 分布 → 进而影响哪些 bit 的 basis 列被删掉、哪些 register bit 需要清理。

#### 5.3.1 Order 的角色回顾：谁拿到哪个 bit

`identityStandardND(name, sizes, order)` 按 `order` 指定的顺序把 bit 分配给各输出维：

| order | 含义 | bit 分配 |
|-------|------|----------|
| `[0, 1]` | dim0 最快（minor） | bit0→dim0, bit1→dim0, … 直到 dim0 占满，然后 bit 给 dim1 |
| `[1, 0]` | dim1 最快（minor） | bit0→dim1, bit1→dim1, … 直到 dim1 占满，然后 bit 给 dim0 |

order 不改变「每个维分到几个 bit」（由 `sizes[d]` 决定），但改变 **bit 的数值层**：先分到的维拿低位（小数值），后分到的维拿高位（大数值）。

#### 5.3.2 场景一：被 slice 的维是 size=1（插入的哑维）

这是 Slice 算法 Step ① 插入的那个维。它的 basis 列最终会被 `ensureLayoutNotLargerThan` 清零。**Order 决定这个清零过程是否导致 register 清理。**

**前提**：parent 的 `sizePerThread` 中有一个 >1 的值对应了被插入的维。

**例 1**：child `[16]`, `dim=0`, `parent = Blocked<sizePerThread=[2,2], threadsPerWarp=[1,4], order=[1,0]>`

| 层级 | dim0 有几个 bit？ | dim1 有几个 bit？ | parentShape = [1, 16] |
|------|-----------------|-----------------|----------------------|
| register | sizePerThread[0]=1 → **0 bit** | sizePerThread[1]=2 → **1 bit** | `[0,1]` |
| lane | threadsPerWarp[0]=1 → **0 bit** | threadsPerWarp[1]=4 → **2 bits** | `[0,2]`, `[0,4]` |

`order=[1,0]` → dim1 先分到所有 bit。dim0 分到 0 个 bit。所以没有任何 bit 需要清零，Step ④ 无事可做。

**例 2**：child `[16]`, `dim=0`, `parent = Blocked<sizePerThread=[2,1], threadsPerWarp=[1,4], order=[0,1]>`

| 层级 | dim0 有几个 bit？ | dim1 有几个 bit？ | parentShape = [1, 16] |
|------|-----------------|-----------------|----------------------|
| register | sizePerThread[0]=2 → **1 bit** | sizePerThread[1]=1 → **0 bit** | `[1,0]` → size=1 清零 → `[0,0]` |
| lane | threadsPerWarp[0]=1 → **0 bit** | threadsPerWarp[1]=4 → **2 bits** | `[0,2]`, `[0,4]` |

`order=[0,1]` → dim0 先分到 register 的 1 个 bit（`[1,0]`），但 dim0 实际 size=1 → `ensureLayoutNotLargerThan` 清零 → `[0,0]`。removeStandardDim 后该 basis 全零 → Step ④ 删除 → **register 少了一个 bit**。

**对比结果**：

| order | parent LL register bit0 | slice 后 register | 是否有清理 |
|-------|------------------------|-------------------|-----------|
| `[1,0]` | `[0, 1]`（映射到 dim1） | 保留 → `[1]` | ❌ 无 |
| `[0,1]` | `[1, 0]`（映射到 dim0）→ 清零 → `[0,0]` | 被删除 | ✅ 少一个 bit |

**规律**：当被 slice 的插入维在 order 中位置越靠前（越快），它越容易在 `sizePerThread` 中分到 register bit，越容易触发 Step ④ 的清理。反之，如果它在 order 中靠后（越慢），`sizePerThread` 的 bit 优先给了其他维，它分不到 bit，就不需要清理。

#### 5.3.3 场景二：被 slice 的维是 parent 的正常维（非插入哑维）

这时被删的 dim 在 parent 中 size>1，它的 basis 列有非零值。**Order 决定这些非零值原来分布在哪几个输入 bit 上，但不改变删除后的数值。**

**例 3**：parent shape = `[4, 16]`, slice `dim=0`，parent blocked `sizePerThread=[1,2], threadsPerWarp=[2,2], warpsPerCTA=[1,1]`

**order = `[1,0]`**（dim1 最快）：

```
identityStandardND 分配：
  register: dim0=1(0bit), dim1=2(1bit) → bit0 → [0, 1]
  lane:     dim0=2(1bit), dim1=2(1bit) → order=[1,0]: bit0→dim1([0,1]), bit1→dim0([1,0])

parent LL（输出 [dim0, dim1]）：
  register bit0 → [0, 1]
  lane bit0     → [0, 1]     ← dim1 贡献
  lane bit1     → [1, 0]     ← dim0 贡献
  ensureLayoutNotSmallerThan 补 wrap：
  register bit1 → [0, 8]     ← dim1 wrap 到 16
```

removeStandardDim(parentLL, 0)（删 dim0 列）：
```
register bit0 → [1]
register bit1 → [8]
lane bit0     → [1]      ← 非零，保留
lane bit1     → [0]      ← 全零！原来只映射 dim0
```
Step ④：删除 lane bit1（`[0]`）→ lane 只剩 1 个 bit。

**order = `[0,1]`**（dim0 最快）：

```
identityStandardND 分配：
  register: dim0=1(0bit), dim1=2(1bit) → bit0 → [0, 1]（同 order=[1,0]）
  lane:     dim0=2(1bit), dim1=2(1bit) → order=[0,1]: bit0→dim0([1,0]), bit1→dim1([0,1])

parent LL：
  register bit0 → [0, 1]
  lane bit0     → [1, 0]     ← dim0 贡献（和 order=[1,0] 不同！）
  lane bit1     → [0, 1]     ← dim1 贡献
  register bit1 → [0, 8]     ← dim1 wrap
```

removeStandardDim(parentLL, 0)：
```
register bit0 → [1]
register bit1 → [8]
lane bit0     → [0]     ← 全零！dim0 列被删
lane bit1     → [1]     ← 保留
```
Step ④：删除 lane bit0 → 同样是 lane 只剩 1 个 bit，但保留的和 order=[1,0] **不是同一个 bit**。

**关键发现**：

| order | 被清理的 lane bit | 保留的 lane basis | 最终 register+basis |
|-------|------------------|------------------|-------------------|
| `[1,0]` | bit1（`[1,0]`→`[0]`） | bit0 = `[1]` | reg=`[1],[8]`, lane=`[1]` |
| `[0,1]` | bit0（`[1,0]`→`[0]`） | bit1 = `[1]` | reg=`[1],[8]`, lane=`[1]` |

**register/lane bit 的数值相同，但占用的 bit 位置不同**。在实际硬件中，lane 的 0-31 是固定的，bit 位置决定哪个 thread 被 broadcast。所以 order 改变了"哪些 thread 在这次 slice 后变成 broadcast"。

#### 5.3.4 直观总结

```
parent = 2D blocked [M, N]

order=[1,0] (N fastest)       order=[0,1] (M fastest)
  N 拿低 bit, M 拿高 bit        M 拿低 bit, N 拿高 bit
                                
slice dim=0 (去掉 M)           slice dim=0 (去掉 M)
  原来映射到 M 高 bit 的那些     原来映射到 M 低 bit 的那些
  → 被清理（变成 broadcast）      → 被清理（变成 broadcast）

结果数值一样，但哪几个 thread          结果数值一样，但哪几个 thread  
被 broadcast 不同                     被 broadcast 不同
```

| 结论 | 说明 |
|------|------|
| **order 不影响 slice 后 basis 的数值** | removeStandardDim 只删列，不改数值 |
| **order 影响哪些输入 bit 被清理** | 决定哪个 register/lane bit 原来映射到被删的维 |
| **order 影响 register 是否减少** | 当插入哑维时，若 order 使它先分到 register bit → 可能触发清理 |
| **order 不影响 slice + expand 的互逆性** | expand 把 0 列插回同一位置，basis 数值不变 |

---

## §6 反操作：Expand 的语义

### 6.1 什么是 Expand？

**Expand**（`expand_dims`）是 Slice 的逆操作：

- **Slice**：在某个维 squeeze size=1 的轴（去掉一列 basis）
- **Expand**：在某个维 **插入** size=1 的轴（增加一列 basis）

```
               Slice (dim=d)
    [M, N]  ────────────────→  [N] （降维）
      ↑                           │
      │  Expand (dim=d)           │
      └───────────────────────────┘
           恢复 [M, N] 分布
```

### 6.2 LinearLayout 层面的 Expand 语义

对一个 LinearLayout `LL: (reg, lane, warp, block) → (dim0, dim1, ..., dim_{k-1})`，在 `dim=d` 处做 expand_dims：

```
输入：LL 的输出维数 = k, basis 每行 = [b0, b1, ..., b_{k-1}]
输出：新 LL 的输出维数 = k+1
      每行 basis = [b0, ..., b_{d-1}, 0, b_d, ..., b_{k-1}]
      即在 dim 位置插入一列 0
```

**关键**：expand 插入的是一列 **0**。这意味着：

- 翻任何一个输入 bit **不会改变** 这个新插入的维的坐标
- 新维的 size=1，只有一个值 0，且所有硬件位置都映到 0
- 这就是 **broadcast** 的数学表示：所有 thread 都持有这个 size=1 维的唯一坐标

### 6.3 Expand 在不同维度的语义差异

相同 parent layout，expand 在不同 `dim` 位置插入的效果完全不同：

| Expand dim | 新维在输出坐标中的位置 | 效果 |
|-----------|----------------------|------|
| `dim=0`（最前面） | `[new, old0, old1, ...]` | 在 **最外层** 加一维，parent 所有维右移 |
| `dim=k`（最后面） | `[old0, old1, ..., new]` | 在 **最内层** 加一维，parent 维不变 |
| `dim=d`（中间） | `[old0.., new, ..old_{k-1}]` | 在 **指定位置** 插入，前后维各保留其含义 |

**与 parent layout 的 order 无关**：expand 是在 **输出空间**（逻辑坐标）插入一个维，而 order 只影响 parent 构造时 bit 如何分配到各输出维。expand 插入的 0 列不依赖 order。

### 6.4 Expand 的零成本性质

在 LinearLayout 层面，expand_dims 是 **纯元数据操作**：

```
expand_dims(LL, dim=d) = LL 的 basis 中每行在 dim 位置插入一个 0
```

- 不增加 register bits
- 不增加 lane/warp 输入
- 不改变任何硬件坐标的映射方式
- 唯一变化：输出维数 +1，basis 向量变长 1

所以 `Slice(dim=d, parent=Layout)` 做 expand_dims(d) 恢复 parent 时 **零成本**——只是删掉的 0 列又插回去了。

---

## §7 Expand 在不同维度上的计算举例

### 7.1 Blocked LinearLayout 在 dim=0 和 dim=1 展开

**基础 Layout**：1D Blocked，`shape=[16]`，等效 `BlockedLayout{sizePerThread=[2], threadsPerWarp=[4]}`

```
LL bases（输出维 [dim0]）：
  register bit 0 → [1]
  register bit 1 → [8]    （wrap bit）
  lane bit 0     → [2]
  lane bit 1     → [4]
```

#### Case A：Expand at `dim=0`（最前面插入新维）

```
expand_dims(LL, dim=0)：
  输出维 [new_dim, dim0]

  register bit 0 → [0, 1]     ← 新维不动（0），旧 dim0 右移
  register bit 1 → [0, 8]
  lane bit 0     → [0, 2]
  lane bit 1     → [0, 4]
```

含义：新维 size=1（广播），所有 thread 在新维上的坐标都是 0。相当于把 1D 分布塞到「2D 张量的第 1 列」：

```
新张量 shape = [1, 16]
等价 2D BlockedLayout:
  sizePerThread=[1, 2], threadsPerWarp=[1, 4], order=[1, 0]
  → dim0(size=1) 不得 bit，dim1(size=16) 占所有 bit
```

#### Case B：Expand at `dim=1`（最后面插入新维）

```
expand_dims(LL, dim=1)：
  输出维 [dim0, new_dim]

  register bit 0 → [1, 0]     ← 旧 dim0 在左边，新维在右边
  register bit 1 → [8, 0]
  lane bit 0     → [2, 0]
  lane bit 1     → [4, 0]
```

含义：相当于把 1D 分布塞到「2D 张量的第 1 行」：

```
新张量 shape = [16, 1]
等价 2D BlockedLayout:
  sizePerThread=[2, 1], threadsPerWarp=[4, 1], order=[0, 1]
  → dim0(size=16) 占所有 bit，dim1(size=1) 不得 bit
```

### 7.2 Swizzle Shared LinearLayout 在 dim=0 和 dim=1 展开

**基础 Layout**：Swizzle Shared，`shape=[8, 16]`，`vec=2, perPhase=1, maxPhase=4`

假设其 LL 的 bases 如下（输出维 [dim0=行, dim1=列]）：

```
offset bit 0 → [0, 1]      // 列 +1
offset bit 1 → [0, 2]      // 列 +2
offset bit 2 → [0, 4]      // 列 +4
offset bit 3 → [0, 8]      // 列 +8
offset bit 4 → [1, 0]      // 行 +1
offset bit 5 → [2, 0]      // 行 +2
offset bit 6 → [4, 0]      // 行 +4
offset bit 7 → [0, 16]     // swizzle XOR: 列 +16（来自 phase 混叠）
offset bit 8 → [8, 0]      // 行 wrap
```

#### Case A：Expand at `dim=0`（行前插入新维）

```
expand_dims(LL, dim=0)：
  输出维 [new_dim, dim0(行), dim1(列)]

  offset bit 0 → [0, 0, 1]
  offset bit 1 → [0, 0, 2]
  offset bit 2 → [0, 0, 4]
  offset bit 3 → [0, 0, 8]
  offset bit 4 → [0, 1, 0]
  offset bit 5 → [0, 2, 0]
  offset bit 6 → [0, 4, 0]
  offset bit 7 → [0, 0, 16]
  offset bit 8 → [0, 8, 0]
```

新维 size=1，basis=0（全广播）。这等价于在 `parentShape=[1, 8, 16]` 上直接构造的 swizzle。

#### Case B：Expand at `dim=1`（行列之间插入新维）

```
expand_dims(LL, dim=1)：
  输出维 [dim0(行), new_dim, dim1(列)]

  offset bit 0 → [0, 0, 1]
  offset bit 1 → [0, 0, 2]
  offset bit 2 → [0, 0, 4]
  offset bit 3 → [0, 0, 8]
  offset bit 4 → [1, 0, 0]
  offset bit 5 → [2, 0, 0]
  offset bit 6 → [4, 0, 0]
  offset bit 7 → [0, 0, 16]
  offset bit 8 → [8, 0, 0]
```

新维在行和列之间插入，不影响行/列原有的 bit 映射。

#### Case C：Expand at `dim=2`（最后面插入新维）

```
expand_dims(LL, dim=2)：
  输出维 [dim0(行), dim1(列), new_dim]

  offset bit 0 → [0, 1, 0]
  offset bit 1 → [0, 2, 0]
  offset bit 2 → [0, 4, 0]
  offset bit 3 → [0, 8, 0]
  offset bit 4 → [1, 0, 0]
  offset bit 5 → [2, 0, 0]
  offset bit 6 → [4, 0, 0]
  offset bit 7 → [0, 16, 0]
  offset bit 8 → [8, 0, 0]
```

新维在最内层，同样全 0 广播。

### 7.3 计算结果对比表

| 操作 | 输出维 | register/lane/offset 的 basis 变化 | 物理含义 |
|------|--------|-----------------------------------|----------|
| 原始 1D blocked | `[dim0]` | `[1], [8], [2], [4]` | 16 个元素在 1D 排布 |
| expand dim=0 | `[new, dim0]` | `[0, 1], [0, 8], [0, 2], [0, 4]` | size=1 新维在最外层广播 |
| expand dim=1 | `[dim0, new]` | `[1, 0], [8, 0], [2, 0], [4, 0]` | size=1 新维在最内层广播 |
| expand dim=0 后 slice dim=0 | `[dim0]` | `[1], [8], [2], [4]` | 回到原始 → **slice 与 expand 互逆** |

| 操作 | 输出维 | offset 的 basis 变化 | 含义 |
|------|--------|---------------------|------|
| 原始 2D swizzle | `[row, col]` | `[0,1], [0,2], ..., [1,0], [2,0], ...` | 8×16 的 swizzle 排布 |
| expand dim=0 | `[new, row, col]` | `[0,0,1], [0,0,2], ..., [0,1,0], ..., [0,8,0]` | 外层广播 |
| expand dim=1 | `[row, new, col]` | `[0,0,1], [0,0,2], ..., [1,0,0], ..., [8,0,0]` | 行-列之间广播 |
| expand dim=2 | `[row, col, new]` | `[0,1,0], [0,2,0], ..., [1,0,0], ..., [8,0,0]` | 最内层广播 |

---

## §8 总结

### 8.1 关键对照表

| 概念 | 说明 |
|------|------|
| **Slice 的本质** | parent layout 的 dimension removal adapter |
| **Slice 的算法** | 插入 size=1 → parent LL → 删一列 → 清废 register bit |
| **Expand 的本质** | 在 basis 向量中插入一列 0 |
| **与 parent order 的关系** | 无直接关系；order 只影响 Step ② 的 parent LL 构造 |
| **Slice 何时清理 register** | 当 size=1 的插入维在 parent 中分到了 register bit 且被清零后 |
| **expand 何时零成本** | 始终零成本（纯元数据操作） |
| **Slice + expand 是否互逆** | 是，前提是 slice 和 expand 的 dim 相同 |

### 8.2 直觉总结

```
Slice 是「砍掉一个维度的 basis 列」
Expand 是「插入一个全零的 basis 列」
两者互逆，且都不改变剩余的 bit 分工。
```

**三维铁律**：

1. **Slice 砍掉的是什么**：parent 去掉的那个维的 **所有 basis 列**（即使全是 0）
2. **Expand 插的是什么**：一个全 0 的 basis 列（= 所有硬件位置都在这个新维上坐标 = 0）
3. **Slice 的 register 减少**：不是必须的；只有被砍维在 parent 中真正占用了 register bit 且被清零后才会发生

### 8.3 快速诊断指南

当面对一个 `#ttg.slice` encoding 时，快速理解其效果：

```python
# 伪代码框架
def understand_slice(child_shape, dim, parent):
    # 1. parent 的什么维被 squeeze 了？
    parent_shape = insert_1_at(child_shape, dim)

    # 2. parent 中其他维的 order/分布是否受影响？
    #    答：不影响。removeStandardDim 只是删一列 basis。

    # 3. 删掉的是什么？
    if parent's dim_d_size == 1:
        # 插入的哑维 → basis 列已全零 → 无实质影响
        result = "几乎等价于 parent 在其它维的原始分布"
    elif register bits 映射到该维:
        # 可能有 register 清理
        result = "register 占用减少，其他不变"
    else:
        result = "直接丢掉该维的分布信息"

    return result
```

---

## §9 更多 Parent Layout 类型的 Slice → LinearLayout 转换

> §4 和 §5 分别讲解了 Blocked 和 Swizzle Shared 两种 parent 下的 slice 手算。本节补充 Linear、NvidiaMma、Nested Slice、DotOperand 四种常见 parent，形成完整的 parent 类型图谱。
>
> 所有 parent 类型都走同一套 4 步算法（§3.1），差异**仅在于 Step ② 的 parent LL 结构不同**。理解各种 parent 的 LL 特征，就能预判 slice 后的结果。

### 9.1 LinearEncodingAttr parent（直接 LinearLayout）

**特点**：`#ttg.linear` 存储的就是编制器内部的 LinearLayout，`toLinearLayout` 几乎直出，没有额外的 encoding 语义。

```mlir
#ttg.linear<{register = [[0,1], [0,2]],
             lane     = [[1,0], [2,0]],
             warp     = [[4,0], [8,0]],
             block    = []}>
```

**参数**：child shape `[16]`，`dim=0`，parent 即上述 `#ttg.linear`

| 步骤 | 操作 | basis 变化 |
|------|------|-----------|
| Step ① | parentShape = `[1, 16]` | — |
| Step ② | parent LL（[dim0, dim1]） | `reg→[0,1],[0,8]` `lane→[0,2],[0,4]` `warp→[0,0]` |
| Step ③ | `removeStandardDim(LL, 0)` 删 dim0 列 | `reg→[1],[8]` `lane→[2],[4]` |
| Step ④ | 检查全零 register | 无，不变 |

**结果**：与同等参数 blocked parent 完全一致。

**关键区别**：`#ttg.linear` 的 basis 是**显式写在 IR 里**的，不像 blocked 需要从 `sizePerThread` 等参数推算。所以 Step ② 没有 "identityStandardND → ensureLayoutNotSmallerThan" 那套流程，直接取存好的 bases 即可。

---

### 9.2 NvidiaMmaEncodingAttr parent（Tensor Core 输出）

**特点**：MMA 输出 C 矩阵的 layout 由 `instrShape`（如 `[16,8]`）固定，**不是**标准的 identityStandardND 铺排。其 basis 中各 bit 的分工遵循 Tensor Core 硬件规定（见 `020_linear_layout.md` §5.4.1）。

以 MMAv2 `instrShape=[16,8], warpsPerCTA=[1,1]` 为例：

```
LL bases（输出 [dim0=M, dim1=N]）：
   register bit 0 → [0, 1]     // N 方向 +1
   register bit 1 → [0, 2]     // N 方向 +2
   register bit 2 → [0, 4]     // N 方向 +4（wrap 到 8）
   lane bit 0     → [1, 0]     // M 方向 +1（每个 lane 差一行）
   lane bit 1     → [2, 0]     // M 方向 +2
   lane bit 2     → [4, 0]     // M 方向 +4
   lane bit 3     → [8, 0]     // M 方向 +8
   warp           → []         // 单 warp，无贡献
```

#### Case A：`dim=0`（去掉 M 维）

| 步骤 | 操作 | basis 变化 |
|------|------|-----------|
| Step ① | parentShape 插入 size=1 → `[1, 16, 8]` | — |
| Step ② | parent LL（[dim0, dim1, dim2]）= MMA LL 扩展 | `reg→[0,0,1],[0,0,2],[0,0,4]` `lane→[0,1,0],[0,2,0],[0,4,0],[0,8,0]` |
| Step ③ | 删 dim0 列 | `reg→[0,1],[0,2],[0,4]` `lane→[1,0],[2,0],[4,0],[8,0]` |
| Step ④ | 检查全零 register | 无 |
| **结果** | 输出维 [dim1, dim2] → 等价于 N 方向上每 thread 持 4 个元素，M 被广播掉 |

#### Case B：`dim=1`（去掉 N 维）

| 步骤 | 操作 | basis 变化 |
|------|------|-----------|
| Step ① | parentShape → `[16, 1, 8]` | — |
| Step ② | parent LL（[dim0, dim1, dim2]） | 同上，但 dim1 size=1 |
| Step ③ | 删 dim1 列（中间列） | `reg→[0,0]→[0,1],[0,2],[0,4]` 中的中间 0 被删 → `reg→[0,1],[0,2],[0,4]`；但注意这里 dim1 列全零（因为 size=1），所以 `reg` 最终变成 `[1],[2],[4]` |
| Step ④ | 检查全零 register | 无 |
| **结果** | 输出维 [dim0, dim2] → 等价于 M=16, N=8 各保留，但 register 的 basis 中 N 的贡献被保留 |

> **理解**：MMA 的 register 只映射到 N 维（列方向），lane 只映射到 M 维（行方向）。所以 slice(dim=0) 后 lane 的 M 贡献被删除 → 结果变成所有 lane 共享同一 M 坐标（broadcast）；slice(dim=1) 后 register 的 N 贡献保留 → 每 thread 仍持多个 N 元素。

---

### 9.3 Nested Slice（slice of slice）

**特点**：当连续两次归约时（如先 reduce axis=1 再 reduce axis=0），结果 encoding 是 `Slice(dim=0, parent=Slice(dim=1, parent=...))`。`toLinearLayout` 递归处理。

```
tensor<MxNxK, #blocked<...>>
    │ reduce(axis=2)
    ▼
tensor<MxN, #slice<dim=2, parent=#blocked<...>>>
    │ reduce(axis=1)
    ▼
tensor<M, #slice<dim=1, parent=#slice<dim=2, parent=#blocked<...>>>>
```

**递归展开**：最外层 `toLinearLayout` 的 Step ② 调用内层 `SliceEncodingAttr::toLinearLayout(parentShape)` → 内层再走一遍自己的 4 步 → 返回一个降了一维的普通 LL → 最外层再做 removeStandardDim。

**参数**：child shape `[16]`，`dim=0`，parent = `Slice<dim=1, parent=BlockedLayout<...>>`

| 步骤 | 操作 | basis 变化 |
|------|------|-----------|
| Step ① | parentShape = `[1, 16]` | — |
| Step ② | parent = `Slice(dim=1, parent=2D_Blocked)` → 递归调 toLinearLayout(`[1,16]`) | 内层 Slice 先做：parentShape=[1,1,16] → parent LL 2D → removeStandardDim(dim=1) → 返回 1D LL |
| Step ③ | 最外层 removeStandardDim(LL, 0) | 又删一列 |
| Step ④ | 检查全零 register | 可能两轮各清理一次 |

**结果**：多层 slice 嵌套等价于一次 slice 掉多个维，但中间可能有 register 清理被叠加。

**关键**：`toLinearLayout` 的递归调用是透明的——`SliceEncodingAttr::toLinearLayout` 在 Step ② 调用的就是通用的 `toLinearLayout(parentShape, getParent())`，如果 parent 也是 Slice，就再次进入同样的函数。最终得到一个普通 LL。

---

### 9.4 DotOperandEncodingAttr parent

**特点**：`#ttg.dot_op` 本身不定义数据分布，它的 `toLinearLayout` 根据 `parent` 类型派发到不同函数（blocked → `fmaDotToLinearLayout`、nvidia_mma → `nvidiaDotToLinearLayout`、amd_mfma → `mfmaDotToLinearLayout`）。

当 slice 作用在 dot_op 上时，Step ② 先调 dot_op 的 `toLinearLayout`，得到的是一个描述 A/B 操作数排布的 LL，然后再 removeStandardDim。

```
Slice(dim=d, parent=DotOperand<opIdx=0, parent=MMA C>)
```

#### 以 FMA dot + slice dim=0 为例

parent = `dot_op<opIdx=0, parent=BlockedLayout<[M,N,K]>>`，A 矩阵 shape = `[M, K]`。

dot_op 的 FMA 路径中 A 的 LL 特征：
- K 维由 register 的低 bit 覆盖（`sizePerThread` 沿 K）
- M 维由 lane/warp 覆盖
- 额外 K 的 broadcast（多个 thread 持同一 K 切片）

```
parent LL（输出 [dim0=M, dim1=K]）：
  register bit 0 → [0, 1]     // K += 1
  register bit 1 → [0, 2]     // K += 2
  register bit 2 → [0, 4]     // K += 4
  lane bit 0     → [1, 0]     // M += 1
  lane bit 1     → [2, 0]     // M += 2
  ... (更多 M bits)
  register 可能还有 K wrap bits
```

切片 dim=0 后，M 列被删除 → 结果变成只保留 K 维分布的 1D LL：

```
removeStandardDim(LL, 0)：
  register bit 0 → [1]
  register bit 1 → [2]
  register bit 2 → [4]
  lane bit 0     → [0]     ← 全零！M 的列被删后，lane 只映射 M，变成 [0]
  lane bit 1     → [0]
```

**Step ④ 清理**：lane 的 basis 变成了全零 → 被删除 → lane 的 bit "失效"，多个 lane 指向同一 K 坐标 → **broadcast**。这正好反映 dot_op 的语义：当 M 维被 squeeze 后，所有原先沿 M 分工的 lane 都变成持有同一个 K 切片。

---

### 9.5 六种 Parent Layout 的 Slice 行为对比

| Parent 类型 | LL 的输出输入维 | slice dim 对应列 | register 清理 | 最终效果 |
|------------|----------------|-----------------|-------------|---------|
| **Blocked** | `(reg,lane,warp)→[M,N,…]` | 非零（正常维）或全零（哑维） | 仅当 size=1 的哑维分到 register bit | 丢掉该维的分布信息，或等价于 parent 剩余维分布 |
| **Linear** | 同 blocked（basis 显式存储） | 同上 | 同上 | 同上，但不经过 identityStandardND |
| **Swizzle Shared** | `(offset)→[…]` | 非零或全零 | 无 register 维（shared） | 丢一列 offset basis |
| **NvidiaMma** | `(reg,lane)→[M,N]` reg→N, lane→M | 删 M 列则 lane→广播；删 N 列则 reg→广播 | 无（MMA register 无废 bit） | 被删的维变 broadcast，另一个维保留 |
| **DotOperand (FMA)** | `(reg,lane)→[M,K]` 类似 blocked 但多 K broadcast | 删 M 列则 lane→全零 | ✅ lane 全零 basis 被清理 → 所有 lane 同坐标 | K 维保留，M 维 broadcast |
| **Nested Slice** | 递归 → 中间结果是一个降了维的普通 LL | 每次删一列 | 可能多轮叠加 | 连续降维，中间清理累加 |

### 9.6 统一理解框架

无论 parent 是什么类型，Slice 的 toLinearLayout 都可以简化为三个字：

```
删一列
```

具体而言：

```
parentLL 的 basis 矩阵（每行 = 一个输入 bit，每列 = 一个输出维）
    │  选择第 dim 列
    ▼
删除该列 → 重编号剩余列 → 清理全零行
```

不同 parent 的差异仅在于 **Step ② 构造出来的 basis 矩阵长什么样**，但 Step ③-④ 对所有人都一样。这就是 LinearLayout 作为统一中间表示的价值——不管高层是 MMA、shared 还是 blocked，到了 LL 层面，slice 就只是一个"删列"操作。
