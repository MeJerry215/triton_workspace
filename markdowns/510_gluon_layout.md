# 510 Gluon 张量布局（Layout）阅读笔记

基于 [官方教程 layouts](https://triton-lang.cn/main/getting-started/tutorials/gluon/layouts.html) 与源码 `_layouts.py`、`02-layouts.py`，说明 Gluon 中 **layout 是什么、怎么选、为何影响性能**。TTGIR 底层语义见 `010_layout.md`。

---

## 0. 阅读路线

```
§1 为何 Gluon 要显式 layout     → 与 Triton 的核心差异
§2 BlockedLayout 机制           → 块形状、order、四层铺砖
§3 张量 vs 块：tile / broadcast → 寄存器预算
§4 1D memcpy 与 R 调参          → 最小可跑示例 + 性能
§5 SliceLayout 与 2D memcpy     → 跨步张量索引
§6 layout 与全局内存合并        → 转置、非连续、[::2]+transpose trick、in/out 不同
§7 convert_layout               → 何时不可避免、代价
§8 其他性能面 + 布局等价        → 归约、shared、LinearLayout
§9 附录                         → 速查、选型函数、源码
```

**术语约定**：

| 英文 | 含义 |
|------|------|
| **layout** | 张量元素在 CTA 内 thread/warp/register 上的分布编码 |
| **block / tile** | 单个 program 内 layout 覆盖的逻辑区域 |
| **size_per_thread** | 每 thread 在每维持有的连续元素数（即 R 或 `[R0,R1]`） |
| **order** | 铺砖顺序：先铺 `order[0]` 维，再 `order[1]`… |
| **wrap (tile)** | 张量大于 block：沿维重复 block |
| **broadcast** | 张量小于 block：沿 warp→thread→register 广播 |
| **coalescing** | 全局内存访问合并到 32B 扇区 / 128B cache line |

---

## 1. 为何 Gluon 要显式 layout

Gluon 中 **block 张量必须带 layout**。layout 回答：逻辑元素 `(i,j,…)` 落在 **哪个 thread 的哪个寄存器**。

层次（由内到外）：

```
register → lane(thread) → warp → CTA(program)
```

与经典 Triton 对比：

| | Triton | Gluon |
|---|--------|-------|
| tile 类型 | 编译器推断 encoding | **用户指定** `gl.BlockedLayout(...)` 等 |
| `arange` | 无 layout 参数 | `gl.arange(0, N, layout=...)` |
| 性能调优 | 主要 block 大小 / num_warps | 还要调 **layout 参数** |

**最小 1D 内核**（相对 Triton 多一行 layout）：

```python
@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    indices = gl.arange(0, XBLOCK, layout=layout)  # 入口：指定 layout
    offsets = start + indices
    mask = offsets < xnumel
    value = gl.load(in_ptr + offsets, mask=mask)
    gl.store(out_ptr + offsets, value, mask=mask)
```

`indices` 的 layout 经类型推断 **向前传播**；`load`/`store` 的 tile 继承它，一般只需在 `arange` 处写一次。

**本章小结**：Gluon = 显式管理寄存器 tile 的物理分布。  
**与后文**：§2 讲 BlockedLayout 四参数如何铺砖。

---

## 2. BlockedLayout 机制

### 2.1 四参数与块形状

```python
gl.BlockedLayout(
    size_per_thread=[2, 4],
    threads_per_warp=[16, 2],
    warps_per_cta=[2, 2],
    order=[1, 0],
)
```

**块形状**（逐维相乘）：

```
block_shape[d] = size_per_thread[d] × threads_per_warp[d] × warps_per_cta[d]
```

上例：`[2×16×2, 4×2×2] = [64, 16]`。

约束（教程 + 源码）：

- 各维均为 **2 的幂**
- 每 thread 元素数 = `∏ size_per_thread`，亦为 2 的幂
- `threads_per_warp` 各维之积 = **32**（NVIDIA 一 warp）
- `order` 是维度下标的排列，决定 **先铺哪一维**

### 2.2 铺砖层次（由内到外）

以 `order=[1,0]`（行主序）为例：

**① register（size_per_thread）** — 单 thread 内连续子块：

```
[[T:0, T:1, T:2, T:3],
 [T:4, T:5, T:6, T:7]]     # 2×4，寄存器在内维递增
```

若 `order=[0,1]`（列主序），同一物理块变为：

```
[[T:0, T:2, T:4, T:6],
 [T:1, T:3, T:5, T:7]]
```

**② warp（threads_per_warp）** — 32 个 thread 如何组成 warp 的 thread 格：

```
[[T0,T1], [T2,T3], …, [T30,T31]]   # [16,2] → 16×2=32
```

把每个格点换成该 thread 的 register 子块，得到 warp 在张量元素上的覆盖。

**③ CTA（warps_per_cta）** — 同理把 warp 子块拼成整个 block。

可视化记号：`T0:3` = thread 0 的第 3 个寄存器；`T0|T32` = 不同 warp 的 thread 广播同一逻辑元素。

### 2.3 与 `010_layout.md` 的对应

| Gluon Python | TTGIR |
|--------------|-------|
| `size_per_thread` | `sizePerThread` |
| `threads_per_warp` | `threadsPerWarp` |
| `warps_per_cta` | `warpsPerCTA` |
| `order` | `order` |

Gluon `BlockedLayout` 即 TTGIR `#ttg.blocked<…>` 的 Python 面。

**本章小结**：四参数定义 block 内 register→thread→warp→CTA 的铺砖；`order` 决定子块是行主还是列主。  
**与后文**：§3 处理 block 与张量形状不一致。

---

## 3. 张量 vs 块：tile 与 broadcast

### 3.1 张量 **大于** block → tile（wrap）

例：`128×128` 张量，block `[64,16]` → tile 数 `[2,8]`。

沿 `order=[1,0]` 在 block 维上再铺一层，每 thread 寄存器数 = block 内寄存器 × tile 数。上例每 thread 最终 **128** 个 f32。

**寄存器预算**：写内核时要心算「每 thread 多少寄存器」，避免 occupancy 暴跌。

### 3.2 张量 **小于** block → broadcast

例：`32×8` 张量，block 仍 `[64,16]` → 物理寄存器 `64×16=1024`，逻辑仅 `256` 元素。

广播顺序：**warp → thread → register**。可能出现多 warp 持有同一逻辑元素的副本（教程用 `T0:0|T32:0|T64:0|T96:0` 表示）。

**本章小结**：block 大小由 layout 固定；张量小则浪费寄存器，大则 tile。  
**与后文**：§4 在 1D 上实测 `size_per_thread[0]`（即 R）对带宽的影响。

---

## 4. 1D memcpy 与 R 调参

### 4.1 1D 有效 layout 空间很小

`num_warps=4` 时，1D 实质上只有：

```python
gl.BlockedLayout([R], [32], [4], [0])   # R = 2^k
```

且需满足：`R × 32 × 4 ≥ XBLOCK`，且无冗余 broadcast（教程：`XBLOCK=2048` 时 `R` 最大 16）。

### 4.2 性能实测要点（GB200，教程数据）

| R | 吞吐 (TB/s) | LDG 特征 |
|---|-------------|----------|
| 1 | ~6.57 | 每 warp 一次 LDG，128B 连续，对齐 cache line |
| 2–4 | ~6.47 | 指令变宽，扇区数不变，略慢 |
| 8 | ~6.50 | 2×128B load，stride 0x10 |
| 16 | ~6.21 | 4×128B load |

**机制（不必死记数字）**：

- 全局内存粒度：**32B 扇区**；GPU 合并同扇区访问
- `R=1`：每 warp 的 LDG 恰好 128B → 一条 cache line
- `R` 增大：单条 LDG 变宽，但 load 带 **write_barrier**，STG 的 `wait_mask` 更晚释放 → 流水线变差
- `R=8` 并不普适最优；换 `XBLOCK` 后排序会变 → **需 autotune**

**实践建议**：1D 连续 memcpy 从 `R=1, XBLOCK=8192` 起扫；用 `compiled_kernel.asm["sass"]` 看 LDG/STG。

**本章小结**：layout 直接改 SASS 向量宽度与 load/store 调度，即使「合并访问次数」相同也可能差 10%+。  
**与后文**：2D 可选 layout 多得多，且依赖 **张量 stride**。

---

## 5. SliceLayout 与 2D memcpy

### 5.1 为何需要 SliceLayout

2D 跨步索引需要 **两个 1D 偏移**（行、列），再 broadcast 成 `[XBLOCK, YBLOCK]`。

2D block 的 layout 是二维的；1D `arange` 需要 **降一维** → `SliceLayout(dim, parent)`。

```python
indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))
```

- `SliceLayout(dim=1, …)`：去掉列维 → 得到「沿行」的 1D 分布
- `SliceLayout(dim=0, …)`：去掉行维 → 「沿列」的 1D 分布

### 5.2 构造直觉（parent 为 2D blocked）

对 `dim=1` 切片：

1. 把 parent 映射的 **每一行** 合并成一条（行内寄存器用 `|` 连接）
2. 去掉 thread 内 **冗余寄存器**（同一逻辑列索引只保留一个）

结果：适合 **沿 dim=1 归约** 后的布局；归约结果被 **广播** 到多个 thread。

### 5.3 expand_dims 逆操作

`expand_dims` 沿 slice 维扩展 → 恢复 **parent layout** 的 2D tile。虚拟寄存器上 broadcast **零成本**。

```python
in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)
value = gl.load(in_ptr + in_offsets, mask=mask)
gl.store(out_ptr + out_offsets, value, mask=mask)
```

**本章小结**：2D kernel 用 `SliceLayout` 生成 1D 索引，再 `[:, None]` / `[None, :]` broadcast 成 2D。  
**与后文**：§6 讨论 stride 如何决定「哪个 order 能合并访问」。

---

## 6. layout 与全局内存合并

### 6.1 连续 inner 维（row-major，`stride(1)==1`）

教程选型（`num_warps=4`）：

```python
# 每 program 处理一行，内维连续 → order=[1,0]
gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
# 常配 XBLOCK=1, YBLOCK=2048
```

吞吐 ~6.26 TB/s，接近 1D。

### 6.2 转置张量（inner 维不连续）

同一 layout 跌至 **~0.77 TB/s**（无合并）。

**修复**：交换 block 与 layout 的维语义 — 等价于换 `order`：

```python
# inner 维在 dim0 连续 → order=[0,1]
gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
# XBLOCK=2048, YBLOCK=1
```

对转置输入可达 ~6.59 TB/s。本质是 **让 thread 访问的地址连续**。

### 6.3 非连续输入（如 `input[::2]`）与 transpose trick

教程 `memcpy_2d_contig` 段：对连续大张量隔行取样，再拷到连续 output（等价 `x.contiguous()`）。

```python
xnumel, ynumel = 32 * 1024, 64 * 1024
input = torch.randn((xnumel, ynumel), device="cuda")
input = input[::2]                    # 隔行，shape 减半
output = torch.empty_like(input)      # 连续
```

#### 6.3.1 物理内存 vs 逻辑 view

`.T` **不搬数据**，底层 storage 仍是分配时的字节序。变的是 **shape + stride** 这套索引：

| view | shape | stride | stride==1 的维 | 语义 |
|------|-------|--------|----------------|------|
| `input`（`[::2]` 后） | `(16384, 65536)` | `(131072, 1)` | **dim1（列）** | 行主 view |
| `input.T` | `(65536, 16384)` | `(1, 131072)` | **dim0** | 列主 view（沿 dim0 地址 +1） |

「列主」这里指 **view 的 stride 模式**（dim0 连续），不是把 RAM 重排成 Fortran 存储。

#### 6.3.2 方式 A：行主 view，沿列合并

```python
layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])  # order=[1,0]
# XBLOCK=1, YBLOCK=2048 → 每个 program：1 行 × 2048 列
impl = partial(memcpy_2d_impl, XBLOCK=1, YBLOCK=2048, layout=layout, num_warps=4)
```

内核偏移（`xstride=131072, ystride=1`）：

```
offset = 131072 * indices_x + indices_y
```

固定行 `j`、扫 2048 列 → `y` 连续 → **地址连续** → warp 可合并 LDG。

```
行 j  ●━━━━━━━━━━━━━━━━  2048 个 float（物理上仍隔行取样后的那一行）
      offset = 131072*j + i
```

教程实测：**~6.26 TB/s**（`torch.contiguous()` ~2.95 TB/s）。

#### 6.3.3 方式 B：`.T` + 换 layout（transpose trick）

```python
layout = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])  # order=[0,1]
# XBLOCK=2048, YBLOCK=1 → 每个 program：2048「行」× 1 列（在 .T 的坐标系里）
impl = partial(memcpy_2d_impl, XBLOCK=2048, YBLOCK=1, layout=layout, num_warps=4)
bench_memcpy_impl(input.T, output.T, impl)
```

`input.T` 上 `xstride=1, ystride=131072`：

```
offset = indices_x + 131072 * indices_y
```

`order=[0,1]` 让 thread 沿 **dim0** 铺 → dim0 连续 → 同样合并。

#### 6.3.4 为何两种方式性能相近：同一趟 gmem

逻辑等价（`.T` 的定义）：

```
input.T[i, j]  ==  input[j, i]
```

方式 A 读 `input[j, i..i+2047]`，方式 B 读 `input.T[i..i+2047, j]`，**同一串物理地址**：

```
方式 A:  offset = 131072*j + i
方式 B:  offset = i + 131072*j        # 加法交换律，完全相同
```

```
内存（物理不变）:
  …  [==== 2048 个连续 float ====]  …
         ↑ 两种方式各 program 读同一段

行主 view:  固定 j，沿列 i 扫   + order=[1,0]  → thread 沿 y(列) 合并
列主 view:  固定 j，沿 dim0 i 扫 + order=[0,1]  → thread 沿 x 合并
           （dim0 就是原来的列方向）
```

**不是两种拷贝算法**，而是 **换坐标系 + 换 layout**，让 warp 始终沿 **stride==1 的那一维**走。教程实测 transpose trick：**~6.40 TB/s**，略优于方式 A，差在 program 调度/块形状等细节，不是访问模式本质不同。

#### 6.3.5 与 §6.2「转置张量救场」的关系

| 场景 | 问题 | trick |
|------|------|-------|
| §6.2 整 tensor `.T` | 行主 view 下 inner 不连续 → ~0.77 TB/s | `.T` + `order=[0,1]` + 交换 X/YBLOCK → ~6.59 TB/s |
| §6.3 `input[::2]` | 列仍连续，方式 A 已 ~6.26 TB/s | trick 可选，略提速，**证明两种 view 物理等价** |

§6.2 是 **必须**换 layout 才能合并；§6.3 是 **两种等价路径都能合并**，trick 用于统一「沿 stride==1 维扫」的心智模型。

#### 6.3.6 实测汇总（教程 GB200）

| 方法 | 吞吐 |
|------|------|
| 2D memcpy（方式 A，`[::2]`） | ~6.26 TB/s |
| `torch.Tensor.contiguous()` | ~2.95 TB/s |
| 2D memcpy transpose trick（方式 B） | ~6.40 TB/s |

**本章小结（§6.3）**：layout 对齐 **view 的连续维**；`.T` 把连续维从 dim1 翻到 dim0，配 `order=[0,1]` 后物理 load 与行主 view 沿列扫 **相同**。

### 6.4 输入输出 layout **相反**

例：input 沿 dim1 连续，output 沿 dim0 连续。单一 layout 的 load **或** store 必有一侧非合并（~1 TB/s 级）。

**结论**：load 与 store 各用一套 layout，中间 `convert_layout`。

### 6.5 按 stride 选 layout（教程函数）

```python
def get_layout_for_gmem_access(tensor, num_warps):
    if len(tensor.shape) == 1:
        return gl.BlockedLayout([1], [32], [num_warps], [0])
    assert 1 in tensor.stride()
    if tensor.stride(1) == 1:
        return gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    else:
        return gl.BlockedLayout([1, 1], [32, 1], [num_warps, 1], [0, 1])
```

规则：**哪一维 stride==1，就让该维做「内维合并」对应的 order**。

**本章小结**：layout 必须匹配 **全局张量的连续维**，不是匹配逻辑 tile 形状 alone。  
**与后文**：in/out 不同时不可避免 layout 转换。

---

## 7. convert_layout

```python
value = gl.load(in_ptr + in_offsets, mask=mask_in)
value = gl.convert_layout(value, layout_out)
gl.store(out_ptr + out_offsets, value, mask=mask_out)
```

**代价**：常涉及 **跨 thread / 跨 warp** 数据移动；跨 warp 往往走 **shared memory** → 占 smem、降 occupancy、限制流水线深度。

**何时值得**：相对 **非合并 gmem**（10× 慢），convert 仍划算。in/out 异构连续维时 **无法避免**。

**编译器优化**：`layout_in == layout_out` 时多余路径会被消掉。

**双 layout 2D 实测**：~4.8 TB/s（含转换成本），远好于单 layout ~1 TB/s。

**本章小结**：`convert_layout` 是布局不匹配时的桥；成本可观但通常低于烂 gmem。  
**与后文**：§8 归约等操作也受 layout 影响，但策略不同。

---

## 8. 其他性能面与布局等价

### 8.1 归约 / 扫描 / gather

- layout 决定是否需要 **warp shuffle / shared** 归约
- 例：行内每元素在不同 thread → butterfly shuffle；每 thread 持整行 → **零通信**
- 编译器会生成合法归约，但 **先 convert 再归约** 往往更亏；能在源头选对 layout 更好

### 8.2 Shared memory

读写受 **shared layout** 与 **寄存器 layout** 共同影响（bank conflict）。编译器会缓解，但 layout 仍改变冲突次数。

### 8.3 无规范形式 + 等价布局

同一元素映射可用多种 layout 表示，例如：

```python
gl.BlockedLayout([1], [32], [4], [0])
gl.SliceLayout(1, gl.BlockedLayout([1, 1], [32, 1], [4, 1], [1, 0]))
```

 trivial 转换（仅寄存器重排）可用：

```python
gl.convert_layout(x, layout, assert_trivial=True)
```

### 8.4 DistributedLinearLayout（可选深读）

最通用的分布表示（论文 [arXiv:2505.23819](https://arxiv.org/abs/2505.23819)），用 `reg/lane/warp/block_bases` 显式给出。上例 1D 128 元素等价于：

```python
gl.DistributedLinearLayout(
    reg_bases=[],
    lane_bases=[[1], [2], [4], [8], [16]],
    warp_bases=[[32], [64]],
    block_bases=[],
    shape=[128],
)
```

即逻辑索引的位分解：低 5 位 → lane，高 2 位 → warp。适合高维 reshape / 复杂重排；日常 blocked + slice 够用。

**本章小结**：layout 影响通信量；Gluon 无唯一规范形，但可用 linear layout 统一理解。

---

## 9. 附录

### 9.1 场景速查

| 场景 | 推荐 layout / 动作 |
|------|-------------------|
| 1D 连续 memcpy | `BlockedLayout([1],[32],[W],[0])`，autotune `R`/`XBLOCK` |
| 2D row-major inner | `order=[1,0]`，`[1,1]×[1,32]×[1,W]` |
| 2D 隔行 `[::2]`（列仍连续） | `order=[1,0]` + `XBLOCK=1,YBLOCK=2048`；或 `.T` + `order=[0,1]` + `XBLOCK=2048,YBLOCK=1`（物理等价） |
| 2D col-major inner | `order=[0,1]`，`[1,1]×[32,1]×[W,1]` |
| 2D 索引 | `SliceLayout` + broadcast |
| in/out 连续维不同 | 双 layout + `convert_layout` |
| 仅重排寄存器 | `convert_layout(..., assert_trivial=True)` |
| MMA / TMA / TensorMem | 专用 layout（见 `TritonGPUAttrDefs.td`、后续教程） |

### 9.2 教程实验命令

```bash
cd triton/python/tutorials/gluon
python 02-layouts.py                              # 全部
python 02-layouts.py R_vs_throughput,LDG_STG_instructions
```

### 9.3 参数自检清单

- [ ] `∏ threads_per_warp == 32`（NVIDIA）
- [ ] `∏ size_per_thread × ∏ threads_per_warp × ∏ warps_per_cta` 覆盖 `XBLOCK×YBLOCK`（或接受 tile/broadcast）
- [ ] layout 的「内维」与张量 `stride==1` 的维一致
- [ ] 估算每 thread 寄存器数

### 9.4 源码索引

| 路径 | 内容 |
|------|------|
| `triton/python/tutorials/gluon/02-layouts.py` | 教程完整代码与 benchmark |
| `triton/python/triton/experimental/gluon/language/_layouts.py` | `BlockedLayout`、`SliceLayout`、`DistributedLinearLayout` |
| `triton/python/triton/experimental/gluon/language/_core.py` | `arange`、`convert_layout`、`expand_dims` |
| `include/triton/Tools/LinearLayout.h` | 线性 layout 数学 |
| `markdowns/010_layout.md` | TTGIR layout 语义 |
| `markdowns/500_gluon_syntax_guide.md` | Gluon 语法总览 |

### 9.5 概念总图

```
gl.arange(..., layout=L)
        │ 类型推断传播
        ▼
tensor<..., L>  ──load/store──►  gmem（合并与否取决于 L 与 stride）
        │
        ├─ SliceLayout ──expand_dims──► parent 2D layout
        │
        └─ convert_layout ──► 新 layout（可能 shuffle / smem）
```

### 9.6 教程三句话总结

1. Gluon **必须**显式管理 layout；种类多，各有用途。
2. layout 对 gmem、通信、shared **影响巨大**（可达 10×）。
3. layout 是写灵活高性能内核的杠杆，代价是心智负担与调参面。

---

## 附：阅读后的理解（编者）

1. **Layout 不是注解，是类型的一部分** — 它决定编译器生成何种 LDG/STG、shuffle 和 smem 搬运；与 `010_layout.md` 的 TTGIR encoding 一一对应，Gluon 只是把选择权交到 Python。

2. **选 layout 的第一性问题：下一个内存操作沿哪维连续？** — 教程用转置与 in/out 异构说明：合并访问 > 几乎一切微优化；`get_layout_for_gmem_access` 是可编码的启发式。

3. **BlockedLayout 四参数 = 分形铺砖** — `order` 控制子块行/列主序；`size_per_thread` 在 1D 上即向量宽度 R，连接 **寄存器占用** 与 **指令级并行** 的权衡。

4. **SliceLayout 是 shape 操作的影子** — 降维索引、升维 broadcast 与 layout 同构；理解 slice 即理解 `expand_dims` / 归约后广播为何零成本。

5. **convert_layout 是显式代价** — 能不用则不用；in/out 连续维不一致时无法省略。后续异步 copy / 流水线教程会讲如何 **隐藏** 该成本。

6. **与 Triton 分工** — 简单 elementwise 用 Triton 更省事；需要掌控 gmem 合并、MMA 操作数布局、多 stage pipeline 时，Gluon 的 layout 显式化是设计意图而非负担。

7. **Transpose trick** — `.T` 只改 view（连续维从 dim1→dim0），换 `order` 让 thread 沿 stride==1 维合并；物理上与行主 view 沿列扫是 **同一趟 gmem**（§6.3）。

---

*关联：`500_gluon_syntax_guide.md` §500.7；下一教程：异步复制与软件流水线。*
