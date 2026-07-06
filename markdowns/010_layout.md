# 010 Triton GPU Layout 排布说明

基于 `TritonGPUAttrDefs.td`、`CTAEncodingAttr.td`、`LinearLayoutConversions.cpp`，说明 TTGIR 中所有 layout 的含义与用法，涵盖 NVIDIA 与 AMD 目标。

---

## 0. 阅读路线

文档分 **三部分**；按章号顺序读即可，也可按下方「首次阅读」跳读。

```
Part A  基础（§1–§3）     IR 是什么、硬件层次、tile / wrap / broadcast
Part B  LinearLayout（§4） 读 bases、手算例、apply（深读见 020）
Part C  参考（§5–§10）    14 种 encoding 全景与各 encoding 详述
```

**章节目录**：

| 章 | 内容 | 必读？ |
|----|------|--------|
| §1 | IR 里 layout 挂在哪、Distributed vs Shared | ✅ |
| §2 | Thread → Warp → CTA → CGA、`order`、`CTALayout` | ✅ |
| §3 | 逻辑坐标 → thread；wrap / broadcast | ✅ **核心** |
| §4 | LinearLayout：basis、输入维、手算例 A/B | 读 `#ttg.linear` 时建议读 |
| §5 | 14 种 encoding 一览 + 场景索引 | 查表用 |
| §6 | Distributed：`blocked` / `linear` / `slice` | ✅ 日常最常用 |
| §7 | Shared memory 五种 | dot staging 时读 |
| §8 | MMA / Dot | matmul 时读 |
| §9 | TMEM（Hopper+） | 按需 |
| §10 | 附录：速查、dump IR、源码索引 | 查表用 |

**与 `020_linear_layout.md` 的分工**：

| 文档 | 视角 | 核心问题 |
|------|------|----------|
| **本文 §3、§6–§9** | 人读 IR、手算 tile | 这个 encoding **是什么意思**？ |
| **本文 §4** | 5–30 分钟直觉 | `#ttg.linear` 的 bases **怎么读**？ |
| **020 §1–§4** | 编译器内部 | `blocked` 参数 **怎么变成 bases**？ |
| **020 §5–§7** | 转换与 codegen | `invertAndCompose`、`applyLinearLayout` |

两者描述 **同一张映射表**：本文 §3 是 **逻辑→硬件**（tile 表），§4/020 是 **硬件→逻辑**（basis + XOR）。

**推荐首次阅读顺序**：

```
§1–§2  认出 IR 里的 layout 与硬件层次
  ↓
§3     建立 tile / wrap / broadcast（必读）
  ↓
§6.1   blocked 参数与 tile 大小（最常用）
  ↓
§5.3   按场景索引跳到 §7 / §8 / §9
  ↓
§4     需要读 `#ttg.linear` / 手推 bases 时再读（§4.2→4.4→4.5→4.6）
  ↓
020    需要读编译器实现或验证单测时再读
```

**术语约定**（与 IR 一致）：

| 英文 | 含义 |
|------|------|
| **tile** | layout 覆盖的基础区域；tensor 更大时沿维 **重复** |
| **rep** | repetition，tile 在张量上再铺一份 |
| **slot** | tile 内索引位置，对应 thread 分工 |
| **wrap / broadcast** | tensor 维 长于 / 短于 tile 时的两种分布 |
| **lane** | warp 内 thread id（0–31）；blocked 里由 `threadsPerWarp` 体现 |
| **instrShape** | MMA 指令一次处理的 [M,N,K] tile；**不等于** tensor shape |

---

## 1. 从 IR 入手：Layout 是什么

### 1.1 它出现在哪

Layout（encoding）是张量类型的 **第二个模板参数**，描述元素在 GPU 上的排布：

```mlir
tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>>
```

**有 layout**：`tt.load` / `tt.dot` 等产生 **ranked tensor**（如 `tensor<256xf32, #blocked>`）。

**无 layout**：标量或逐元素循环，只有 `!tt.ptr<f32>`、`f32`，TTGIR 里看不到 `#ttg.*`。

| 写法 | TTGIR 典型形态 |
|------|----------------|
| `for i: gl.load(ptr+i)` | 标量 load，无 encoding |
| `gl.load(ptr+arange(0,BLOCK))` | `tensor<BLOCKxf32, #blocked>` |

### 1.2 两大类

| 类别 | 存在位置 | 回答的问题 |
|------|----------|------------|
| **Distributed** | 寄存器（每 thread 私有） | 逻辑元素 **在哪个 thread** |
| **Shared** | `ttg.local_alloc` 共享内存 | 逻辑元素 **在 shared 哪个偏移** |

下文 §3–§4 为公共基础；§6 以 Distributed 为主；§7 讲 Shared。

### 1.3 与 module 属性的区别

```mlir
module attributes {"ttg.num-warps" = 8, "ttg.num-ctas" = 1, ...}
```

| 属性 | 层级 | 含义 |
|------|------|------|
| `ttg.num-warps` | **单个 CTA 内** | warp 个数 → CTA 内 thread 数 = `num_warps × 32` |
| `ttg.num-ctas` | **单个 cluster 内** | CTA（block）个数；`1` = 不用 cluster |

与 layout 里 `warpsPerCTA=[8]` 对应，但是 **launch 元数据**；不等于 grid 上有多少个 block（那由 `grid` / `program_id` 决定）。详见 §2。

**本章小结**：layout 挂在 **tensor 类型** 上；先确认 IR 里是 block tensor 还是标量。

---

## 2. 硬件层次：Thread → CGA

读 layout 参数前，先固定 **四级结构**（`DistributedEncodingTrait`）：

```
CGA (cluster，Hopper+ 可选)
  └── CTA (block，= 一个 program / program_id)
        └── Warp（32 thread）
              └── Thread
                    └── Register（每 thread 持有的元素）
```

Layout 参数落在不同层：

```
CTAs Per CGA  →  Warps Per CTA  →  Threads Per Warp  →  Values Per Thread
   (#5 CTA)         (blocked 等)      (blocked 等)         (sizePerThread)
```

### 2.1 与 CUDA / Python 对照

| 术语 | CUDA | Triton |
|------|------|--------|
| **Thread** | thread | CTA 内线性 id 的一部分 |
| **Warp** | warp（32 lane） | — |
| **CTA** | thread block | `program_id` 对应 **一个** CTA |
| **CGA** | thread block cluster | `num_ctas` > 1 且 SM90+ |

```python
kernel[grid](..., num_warps=8, num_ctas=1)
# grid 决定有多少个 CTA；num_ctas=1 表示每个 cluster 只有 1 个 block
```

### 2.2 `order`：最快变化维在前

`order = [0, 1]` → dim0 变化最快，线性 id 先沿 dim0 走：

```
shape=[4,4], order=[0,1]:
[0  4  8  12]
[1  5  9  13]
...
```

`blocked` 的 `threadsPerWarp` / `warpsPerCTA` 都按 **order** 解释各维。

### 2.3 `CTAEncodingAttr`（`#ttg.cta_layout`）

描述 **cluster 内 block id → 逻辑张量坐标**（`block` → `dim0, dim1, ...`），内嵌在 `blocked` / `swizzled_shared` / `nvidia_mma` 的 `CTALayout` 中。

遗留字段：`CTAsPerCGA`、`CTASplitNum`、`CTAOrder`（逐步由 `LinearLayout` 替代）。

**本章小结**：`warpsPerCTA` 只管 **一个 CTA 内部**；`num_ctas` 管 **cluster 里有几个 block**；`grid` 管 **一共 launch 多少 block**。

---

## 3. 核心语义：谁持有哪个元素

> **Part A 核心章**。建立 tile 表视角：逻辑坐标 → 哪个 thread 持有；wrap / broadcast 由此而来。编译器侧的 basis 表达见 **§4.4**。

本章建立 §6 Distributed 的公共模型；**先读本章再读 blocked 参数**。

### 3.1 映射在回答什么

```
输入：逻辑坐标 i = (i₀, i₁, …)     「T[i] 是哪个元素」
输出：thread id（或集合）            「哪些 thread 寄存器里有 T[i]」
```

| 步骤 | 对象 | 作用 |
|------|------|------|
| **$L$** | 基础 **tile** 表 | slot → thread id |
| **$\mathcal{L}(T)$** | 套上 tensor shape | 坐标 $i$ → 查 $L$（含 wrap/broadcast） |

**不做的事**：不改数值；不把 thread id 反查坐标（另用 `local_index` 等）。

**thread id**：CTA 内一维整数 $0 \ldots \texttt{num\_warps}\times 32 - 1$。二维 layout 表里写的也是这个 id：

$$
\text{thread\_id} = \text{layout\_row} \times L.\mathrm{shape}[1] + \text{layout\_col}
$$

### 3.2 wrap 与 broadcast（先记取模）

对每一维 $d$，实用规则：

$$
\text{slot}_d = i_d \bmod L.\mathrm{shape}[d]
$$

| 关系 | 名称 | 结果 |
|------|------|------|
| $T.\mathrm{shape}[d] > L.\mathrm{shape}[d]$ | **wrap** | tile 重复（rep）；**一 thread 多元素** |
| $T.\mathrm{shape}[d] < L.\mathrm{shape}[d]$ | **broadcast** | 多 slot 落到同一逻辑位置；**一元素多 thread** |

```
tile 宽度 = 4:  tensor 列:  0 1 2 3 | 4 5 6 7
                [   rep 0    ] [  rep 1  ]   ← wrap
```

形式化（`TritonGPUAttrDefs.td`）：

$$
\mathcal{L}(T)[i_d] = L\big[(i_d + k_d \cdot T.\mathrm{shape}[d]) \bmod L.\mathrm{shape}[d]\big]
$$

日常只需 $i_d \bmod L.\mathrm{shape}[d]$；$k_d$ 为公式中求和下标，勿理解为运行时 loop。

### 3.3 手算示例 A：1D wrap

$T.\mathrm{shape}=[8]$，$L.\mathrm{shape}=[4]$，$L=[t_0,t_1,t_2,t_3]$：

| $i_0$ | slot | thread |
|-------|------|--------|
| 0, 4 | 0 | $t_0$ |
| 5 | 1 | $t_1$ |

### 3.4 手算示例 B：2D wrap + broadcast（TD 官方）

$T$ 为 2×8，$L$ 为 4×4 tile（数字 = thread id）：

```
layout_col →   0    1    2    3
layout_row ↓
    0          0    1    2    3
    1          4    5    6    7
    2          8    9   10   11
    3         12   13   14   15
```

- 列维：$8>4$ → **wrap**（列 4 与列 0 同 slot）
- 行维：$2<4$ → **broadcast**（tensor 行 $r$ 对应 layout_row $\bmod 2 = r$ 的多行）

$T[0,0]$：col slot 0，layout_row $\in\{0,2\}$ → $L[0][0]=0$，$L[2][0]=8$ → **$\{0,8\}$**。

整张映射（每格为 thread 集合）：

```
        c=0      c=1      c=2      c=3      c=4..7（与 0..3 同 pattern）
r=0   {0,8}    {1,9}   {2,10}   {3,11}     wrap 重复
r=1  {4,12}   {5,13}   {6,14}   {7,15}
```

shape 完全匹配（4×4 / 4×4）时无集合：$T[r,c] \to L[r][c]=r\times 4+c$。

### 3.5 Rep（tile repetition）

tensor 大于一个 tile 时，沿各维 **重复同一 tile** 铺满；即 §3.2 wrap 的多维整体。

**本章小结**：layout = tile 表 $L$ + 与 $T.\mathrm{shape}$ 比较得到 wrap/broadcast；§6 `blocked` 是 $L$ 的具体参数化。

> **编译器侧对应**：wrap → 在 `register` 维 **追加非零 basis**（翻 bit 会换逻辑坐标）；broadcast → 把多余 bit 的 basis **改成全零**（翻 bit 不换逻辑坐标）。`apply()` 始终同一套 XOR，无需特殊分支。详见 §4.4 与 `020_linear_layout.md` §6。

---

<!-- Part B: LinearLayout -->

## 4. LinearLayout：统一数学语言（浅读）

> **定位**：Part B 核心章。建立读 `#ttg.linear` / bases 的直觉；**深读**（手推、codegen）见 `020_linear_layout.md`。
>
> **本章结构**：§4.2 语法约定 → §4.3 四个输入维 → §4.4 apply / wrap / broadcast → §4.5–4.6 手算例 → §4.7 order → §4.8 与 020 分工。

多种 encoding（`linear`、`CTAEncoding`、`padded_shared`）在编译器内部共用 **LinearLayout**（`LinearLayout.h`）。

### 4.1 目标：把「谁拿到哪个元素」写成可组合的函数

把一次访问里“硬件层次的坐标”（寄存器槽位 / thread / warp / CTA …）记为 \((t,w)\)，把“逻辑张量坐标”（多维 index）记为 \(L(t,w)\)。

**映射**：硬件坐标 → 逻辑张量坐标，且满足 XOR 线性：

$$
L(t_1 \oplus t_2,\, w_1 \oplus w_2) = L(t_1,w_1) \oplus L(t_2,w_2)
$$

这里 \(\oplus\) 是按位 XOR；含义是：硬件坐标如果按 bit 翻转，那么输出坐标也按相同方式做“线性叠加”（在 \(\mathbb{F}_2\) 上的线性变换）。这类表示特别适合 GPU：thread/lane/warp 等本来就是用 bit 拼出来的。

### 4.2 读 IR：语法与约定

因为 XOR 线性，只需定义 **basis**（输入为 \(1,2,4,\ldots\) 这些单 bit）时的输出，其余任意输入都是这些 basis 的 XOR 组合。

#### basis 到底是什么？

把 basis 想成一张 **「灵敏度表」**：每个输入维、每个 bit 位置各有一行，写清楚 **「只翻这一位时，逻辑坐标动多少」**。

**约定**（全文统一）：

| 列 | 含义 |
|----|------|
| **数组下标 \(k\)** | `lane = [[…], […], …]` 里 **第 \(k\) 个元素** ↔ 输入的 **bit \(k\)**（**不反转**：`[0]`=bit0，`[1]`=bit1） |
| **输入值** | 只有第 \(k\) 位为 1 时的整数，即 \(2^k\)（1, 2, 4, 8, …） |
| **basis 存的内容** | **输出逻辑坐标上的偏移**（不是输入编号）；1D 写成 `[d0]`，2D 写成 `[d0,d1]` |
| **标准 identity 时** | 偏移常 **碰巧** 等于 \(2^k\) 或 \(2^{k+s}\)（见下），但 swizzle 等一般 **不是** |

**`lane = [[1], [2]]` 怎么读这个数组**：

```
下标 k=0  →  输入 lane=2^0=1  →  输出偏移 [1]   （dim0 += 1）
下标 k=1  →  输入 lane=2^1=2  →  输出偏移 [2]   （dim0 += 2）
```

`warp = [[4], [8]]` 同理，但接在 lane 后面，占 dim0 的 **bit2、bit3**，所以偏移是 \(2^2=4\)、\(2^3=8\)，不是「warp 自己的 \(2^0,2^1\)」：

```
下标 k=0  →  warp=1  →  dim0 += 4
下标 k=1  →  warp=2  →  dim0 += 8
```

**2D 时每个元素是向量**：`register = [[0,1], [0,2]]` → 下标 0 是 `[0,1]`（dim1+1），下标 1 是 `[0,2]`（dim1+2）。

#### dim0 / dim1 是什么？和 §3 怎么接上

读 IR 时最容易混的是 **两套坐标**：

```
输入（硬件）                    输出（逻辑 / 张量）
register, lane, warp, block  →  dim0, dim1, …
「谁拿着数据」                  「拿着张量哪一个元素」
```

**`dim0`、`dim1` 就是 §3.1 里的逻辑下标 \(i_0, i_1\)**——和张量类型里的 shape 一一对应：

```mlir
tensor<16x16xf32, #ttg.linear<…>>
         ↑  ↑
       dim0 dim1   →  shape = [16, 16]
```

| 名字 | 是什么 | 在 §3 里 |
|------|--------|----------|
| **dim0** | 张量 **第 0 维** 的下标 | \(i_0\) |
| **dim1** | 张量 **第 1 维** 的下标 | \(i_1\) |
| **basis `[a, b]`** | 在 dim0 上加 `a`、在 dim1 上加 `b` | 逻辑坐标偏移 |

**和 §3 的关系**（方向相反，说的是同一件事）：

| §3（人读 tile） | LinearLayout / §4（编译器） |
|-----------------|------------------------------|
| 已知 `T[6, 3]`，问哪个 thread 拿？ | 已知 `reg,lane,warp`，问拿到的是 `T[?, ?]`？ |
| 逻辑坐标 → thread id | 硬件坐标 → **(dim0, dim1)** |

例子 B 手算得到 `(6, 3)` → 这个 thread 的寄存器里存的是 **`T[6, 3]`**（第 6 行、第 3 列那个元素，若把 dim0 当行、dim1 当列）。

**和 §2.2 `order`、blocked 参数的关系**（例子 B）：

```
shape = [16, 16]
order = [1, 0]     → dim1 变化更快（minor），dim0 更慢（major）

sizePerThread  = [1, 4]  → register 的 bit 铺在 dim1 上（列方向 4 个）
threadsPerWarp = [4, 1]  → lane 的 bit 铺在 dim0 上（行方向 4 个）
warpsPerCTA    = [4, 1]  → warp 的 bit 继续铺在 dim0 上
```

所以 IR 里才会出现：

- `register` 的 basis 形如 `[0, ?]` → **只改 dim1（列）**
- `lane` / `warp` 的 basis 形如 `[?, 0]` → **只改 dim0（行）**

**一张图串起来**（例子 B，固定 `warp=0` 的一小块）：

```
张量 T[dim0, dim1]          硬件谁拿（简化）
     dim1 →  0   1   2   3
dim0 ↓
  0        reg 在同一 thread 内沿 dim1 递增
  1        lane 沿 dim0 换 thread
  2
  3

L(reg=3, lane=2, warp=1) = (6, 3)
  → dim0=6：lane+warp 在行方向排出来的位置
  → dim1=3：register 在列方向第 3 个槽位
  → 元素 T[6, 3]
```

**1D 时只有一个 dim**：例子 A 只有 `dim0`，basis 退化成单个数 `[[1],[2]]`，就是 §6.1 里 `tensor<256xf32>` 的那一维下标。

#### IR 有几层「维」？`[]` 里几个元素等价于什么？

容易混：**`register`、`lane` 是 4 个并列的输入维名字**；每个名字右边的 `[[…], […]]` **不是**「里面还有子维」，而是 **这一维的 bit 灵敏度表**（几行 = 几个 bit）。

```
#ttg.linear<{  输入维名（硬件轴）     该维的 basis 表（每行 = 1 个 bit）
  register = [],              ← 0 行 = 0 bit
  lane     = [[1], [2]],      ← 2 行 = 2 bit
  warp     = [[4], [8]],      ← 2 行 = 2 bit
  block    = []               ← 0 行 = 0 bit
}>
  输出维：dim0（1D 时每个 basis 行是长度 1 的向量 [d0]）
```

**三层结构**（读任意一行 IR 都适用）：

| 层级 | 是什么 | 例子 A |
|------|--------|--------|
| **① 输入维（inDim）** | `register` / `lane` / `warp` / `block` 四个 **名字** | 4 个名字，其中 register、block **有效大小为 1** |
| **② bit 行数** | `lane = [[…],[…]]` 里 **有几个 `[…]`** = 该输入维 **几个 bit** | lane **2** bit，warp **2** bit，register/block **0** bit |
| **③ 每行向量长度** | 每个 `[…]` 里有几个数 = **几个输出维** dim0, dim1, … | 1D → 每行 1 个数，如 `[1]`、`[2]` |

**和 blocked 参数的等价说法**（例子 A）：

| IR | 等价于 | 该维取值范围 |
|----|--------|--------------|
| `register = []` | `sizePerThread = [1]`（每 thread 1 元素） | `register` 只有 **0** |
| `lane = [[1],[2]]` | `threadsPerWarp = [4]` | `lane ∈ {0,1,2,3}`（**4** 个 thread） |
| `warp = [[4],[8]]` | `warpsPerCTA = [4]` | `warp ∈ {0,1,2,3}`（**4** 个 warp） |
| `block = []` | 单 CTA / 无 cluster | `block` 只有 **0** |
| 输出只有 `dim0` | `tensor<16xf32>` 一维 | `dim0 ∈ {0,…,15}` |

**总 slot 数**（有效 bit 全翻一遍能区分多少硬件位置）：

\[
2^{\#\text{reg bit}} \times 2^{\#\text{lane bit}} \times 2^{\#\text{warp bit}} \times 2^{\#\text{block bit}}
= 1 \times 4 \times 4 \times 1 = 16
\]

这就是 tile 大小，与 shape `[16]` 一致。

**2D 时只变第 ③ 层**：`lane = [[1,0],[2,0]]` 仍是 **2 行（2 bit）**，但每行变成 **2 个数** `[dim0偏移, dim1偏移]`，表示翻 lane bit 时两个逻辑维各动多少。

**对应的 layout（与上表等价）**：

| 项目 | 值 |
|------|-----|
| Encoding | `#ttg.blocked<…>` 或 `#ttg.linear<…>`（bases 相同） |
| blocked 参数 | `sizePerThread=[1], threadsPerWarp=[4], warpsPerCTA=[4], order=[0]` |
| 默认 CTALayout | 单 CTA、无 cluster → `block=[]` |
| **shape** | **`[16]`**（tile = 1×4×4 = 16，与 shape 完全吻合） |
| 单测 | `LinearLayoutConversionsTest::SimpleBlocked` |

等价的 blocked IR：

```mlir
#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [4], warpsPerCTA = [4], order = [0]}>
tensor<16xf32, #ttg.blocked<…>>    // 或 tensor<16xf32, #ttg.linear<…>>
```

**§3 tile 表**（逻辑下标 dim0 → 哪个 `(lane, warp)`，每格 1 个元素）：

```
         lane 0   1   2   3
warp 0      0   1   2   3
warp 1      4   5   6   7
warp 2      8   9  10  11
warp 3     12  13  14  15
```

规律：`dim0 = lane + warp × 4`（bit 不重叠时 XOR = 加法）。**不是** §6.1 vector-add 的 256 元那版——那是 `threadsPerWarp=[32], warpsPerCTA=[8]`，lane/warp 会有 **更多行 basis**（如 `[[1],[2],[4],…]`）。

#### 常见误解：`[[0,1],[0,2]]` 不是「register 从 1 开始」

IR 里 **每一行 basis 是一个输出偏移向量** `[dim0, dim1]`，**不是** register 的编号。

| 你看到的 | 实际含义 |
|----------|----------|
| `register = [[0,1], [0,2]]` | 第 0 行：register 的 **bit0** 单独为 1 时，逻辑坐标 **+ [0,1]** |
| | 第 1 行：register 的 **bit1** 单独为 1 时，逻辑坐标 **+ [0,2]** |
| 行里的 `1`、`2` | 动的是 **dim1**（列方向偏移），不是「第 1 号 register」 |

**`register = 0` 完全合法**，表示所有 register bit 都是 0 → **不 XOR 任何 register basis** → 这个 thread 的 **第 0 号槽位**（起点）：

```
L(reg=0, lane=0, warp=0) = (0, 0)     ← 原点，不是「没有」
L(reg=1, lane=0, warp=0) = (0, 0) ⊕ [0,1] = (0, 1)   ← 第 1 号槽位
L(reg=2, lane=0, warp=0) = (0, 0) ⊕ [0,2] = (0, 2)   ← 第 2 号槽位
L(reg=3, lane=0, warp=0) = [0,1] ⊕ [0,2] = (0, 3)     ← 第 3 号槽位
```

**为什么表里写 `reg=1`、`reg=2` 而不是 `reg=0`？**

basis 只定义 **「单独翻某一个 bit」** 的情形。bit 编号 \(k\) 对应输入值 **\(2^k\)**（只有这一位为 1）：

| register 槽位编号 | 二进制 | 用哪些 basis XOR |
|-------------------|--------|------------------|
| **0** | `00` | （无，原点） |
| **1** | `01` | 第 0 行 `[0,1]` |
| **2** | `10` | 第 1 行 `[0,2]` |
| **3** | `11` | 第 0 行 ⊕ 第 1 行 |

`lane`、`warp` 同理：`lane=0` 是 warp 内 **0 号 thread**，不是「没有 thread」；表里写 `lane=1`、`lane=2` 是因为在定义 **bit0、bit1 各自单独为 1** 时的灵敏度。

**记法**：`bases[inDim][k]` = \(L(\text{inDim}=2^k,\ \text{其它}=0)\) 的坐标偏移；**槽位 0 永远存在**，只是不单独占一行 basis（线性性保证 \(L(0)=0\)）。

**1D vs 2D 对照**：

| | 1D 例子 A | 2D 例子 B |
|--|-----------|-----------|
| basis 形状 | `[[1], [2]]` 单列 | `[[1,0], [2,0]]` 两列 |
| lane 贡献 | 只动 dim0 | 只动 dim0（第二列恒 0） |
| register | 无（sizePerThread=1） | 只动 dim1（第一列恒 0） |
| 手算 | `lane=3,warp=2` → 11 | `reg=3,lane=2,warp=1` → (6,3) |

### 4.3 四个输入维：register / lane / warp / block

#### register 维是什么？

**一句话**：`register` = **同一个 thread 内部**，「我持有的第几个张量元素」的编号，**不是** CUDA 里某个物理寄存器 `$r0` 的硬件编号。

§2 硬件层次里最内层写的是 **Values per thread**；在 LinearLayout 里这个名字就叫 **`register` 输入维**：

```
(block) → warp → lane(thread) → register → 逻辑坐标 (dim0, dim1, …)
         跨 thread 分工          线程内第几个值
```

| 概念 | 对应 |
|------|------|
| blocked 参数 | `sizePerThread`：每 thread **每维** 持有几个元素 |
| register 维有几个 bit | \(\sum_d \log_2(\text{sizePerThread}[d])\)（各维乘起来再取 log） |
| `register = n` | 这个 thread 持有的 **第 n 号槽位**（从 0 起） |
| `apply(register=n, lane=ℓ, …)` | 回答：**第 n 号槽位**里装的是 `T[?, ?]` |

**例子 B**（`sizePerThread=[1,4]`）：每 thread 在 dim1 上持 **4** 个元素 → register 有 **2** 个 bit → `register ∈ {0,1,2,3}`：

| register | 沿 dim1 相对偏移（由 reg bases 决定） | 同一 `(lane,warp)` 下多槽位 |
|----------|--------------------------------------|----------------------------|
| 0 | +0 | 第 1 个元素 |
| 1 | +1 | 第 2 个 |
| 2 | +2 | 第 3 个 |
| 3 | +3 | 第 4 个 |

#### register 是元素还是字节？和 dtype 有关吗？

**一句话**：`register` 数的是 **张量逻辑元素**（element slot），**不是** shared/global 那种字节偏移；**`apply()` 本身与 dtype 无关**，输出永远是 `(dim0, dim1, …)` 这种 **元素下标**。

| 层面 | 单位 | 是否看 dtype |
|------|------|--------------|
| **LinearLayout `register` 维** | 第几个 **元素槽位** | ❌ `apply(reg, lane, …)` 不算字节 |
| **`sizePerThread[d]`** | dim \(d\) 上每 thread 几个 **元素** | ❌ 只数个数，不乘 `sizeof` |
| **register basis 偏移** | 逻辑坐标 **+1、+2**（沿某维多 1 个元素） | ❌ 例子 B 里 `+1` = dim1 下一列，不是 +4B |
| **构造 layout 时**（MMA、swizzle 等） | 有时按 **elementBitWidth** 选不同 basis | ✅ 影响「怎么建 LL」，不改变「register=元素槽」语义 |
| **Codegen / LLVM** | 物理寄存器、vector load 宽度 | ✅ `f32`/`f16`/`i8` 映射到不同硬件寄存器，在 layout **之后** |

**和字节的边界**：

- **Distributed layout（blocked 等）**：全程按 **元素** 理解即可。`f32` 与 `f16` 若 `sizePerThread=[1,4]`，都是每 thread **4 个元素**、register **2 bit**；差别在编译器后面用几条指令、几个物理寄存器装这 4 个值。
- **Shared layout**：另有一套 **字节偏移 + swizzle**（`swizzleByteWidth` 等），那是 **shared 寻址**，不是 register 输入维。
- **特例**：个别 MMA / FP4 路径在 **生成 LL** 时会把「逻辑元素」按更细粒度描述（例如 FP4 按 i8 元素写 basis），仍是 **索引空间里的元素**，不是「register 编号 × 字节数」。

**例子**（`tensor<8x16xf32>`，`sizePerThread=[1,4]`）：

```
register=2, lane=3, warp=1  →  apply → (6, 2)   // T[6,2] 这个 f32 元素
register=2, lane=3, warp=1  →  若改成 xf16，仍是 (6, 2)  // 仍是第 6 行第 2 列那个元素
```

dtype 变了，**layout 映射到的逻辑坐标不变**；变的是该元素占多少位、落在哪些物理寄存器里。

`reg_bases = [[0,1],[0,2]]` 就是说：register bit0 翻转让 dim1 **+1**，bit1 让 dim1 **+2**（XOR 组合出 0..3）。

#### register 和 lane 怎么区分？

| | **register** | **lane** |
|--|--------------|----------|
| **能否这么理解** | ✅ **同一个 thread 内**，第几个张量元素（槽位 0,1,2,…） | ✅ **基本可以** = warp 里的 **thread**（thread 在 warp 内的 id） |
| **范围** | 单 thread 内部 | 单 warp 内部（通常 0–31，layout 里可更少） |
| **不是** | CUDA 物理寄存器 `$r0`；也不是全局 `threadIdx` | 整个 CTA 的 thread id（还要加 **warp**） |
| **blocked 参数** | `sizePerThread` | `threadsPerWarp` |
| **例子 B 管哪维** | 只动 **dim1（列）** | 只动 **dim0（行）** 低 bit |

**层次关系**（固定 `warp`、`block`，看「谁」在变）：

```
同一个 warp、同一个 lane（= 同一个 thread）
  reg=0  →  T[行, 0]
  reg=1  →  T[行, 1]     ← register：换槽位，不换 thread
  reg=2  →  T[行, 2]
  reg=3  →  T[行, 3]

同一个 reg 槽位（例如 reg=0）
  lane=0 →  T[0, 0]
  lane=1 →  T[1, 0]       ← lane：换 thread，同一槽位编号
  lane=2 →  T[2, 0]
```

**和 CUDA 对照**（一个 CTA 内）：

```
threadIdx.x  ≈  lane + warp × 32        （线性化时）
register 维  ≈  这个 thread 私有的第几个「逻辑元素槽」
                 （编译器后来可能映射到不同物理寄存器，layout 不管这个）
```

**例子 A**（1D，`register=[]`）：每 thread **只有 1 个元素** → 没有 register 维；**只靠 lane + warp** 区分 16 个元素。此时「换 lane」= 换 thread = 换元素，没有「同 thread 多槽位」这一说。

**例子 B**：lane 换 **行**（dim0），register 换 **列**（dim1）——**正交**，所以可以分开记：lane ≈ 哪个 thread；register ≈ 这个 thread 手里第几份数据。

**和 lane / warp 的分工**：

| 输入维 | 分工范围 | 典型 blocked 参数 |
|--------|----------|-------------------|
| `register` | **thread 内** 多元素 | `sizePerThread` |
| `lane` | **warp 内** 多 thread | `threadsPerWarp` |
| `warp` | **CTA 内** 多 warp | `warpsPerCTA` |

所以：**不是**「全局第 n 个寄存器」，而是 **「这个 thread 自己手里的第 n 份数据」**。Codegen 里常对每个 `register` 槽位循环生成 load/store（`emitIndices` 会枚举 register 维）。

**两种容易混的「多 register bit」**：

| 情况 | 含义 |
|------|------|
| `sizePerThread` 大 | 本来每 thread 就持多个元素 → register 低位 bit 区分槽位 |
| shape wrap（§4.4） | tile 重复多份 → register **高位** bit 表示「第几轮 tile 副本」，同一 thread 仍可能只持 `sizePerThread` 个 **可见**槽位 |

例子 A 里 `register = []`：每 thread 只有 **1** 个元素（`sizePerThread=[1]`），不需要 register 编号，只靠 `lane`+`warp` 就能铺满 16 个元素。

常见输入维：`register`, `lane`（thread）, `warp`, `block`（CTA）。

### 4.4 apply() 与 wrap / broadcast

#### 怎么“算”一个坐标（概念流程）

- **输入**：给定某次访问的硬件坐标 \((r, \ell, w, b)\)（分别对应 `register/lane/warp/block` 的整数）。
- **拆 bit**：把每个整数拆成 bit（例如 \(\ell=\sum_k \ell_k 2^k\)）。
- **XOR 叠加**：输出坐标从 0 开始；对所有 \(\ell_k=1\) 的位，把 `lane[k]` 这个 basis 向量 XOR 进去；`register/warp/block` 同理。
- **结果**：得到一个多维逻辑坐标（比如 \([i,j]\)）。

这个表示的好处是：很多布局操作（切片/转置/拼接某些维度的 bit）都能转成对 basis 的局部变换，而不需要枚举整张 tile 表。

#### 先瞄一眼：basis = 0 是什么意思

| bit 编号 \(k\) | 输入值 | basis | 含义 |
|----------------|--------|-------|------|
| 1 | `lane = 2` | **`0`**（不是省略） | \(L(\text{lane}=2)\) 与 \(L(\text{lane}=0)\) **相同** → 该 bit **失灵** |

**不是**「把硬件上的 bit 清零」；硬件 `lane` 该是多少还是多少。只是表里写明：**这个 bit 翻 1 也不改逻辑坐标**（broadcast，见下）。

#### apply() 在干什么（三步）

给定硬件坐标 **`lane=3, warp=2`**（沿用上面完整小例子同一张表）：

1. **拆 bit**：`lane=3` → 二进制 `11` → **bit0、bit1** 为 1；`warp=2` → 二进制 `10` → 只有 **bit1** 为 1  
2. **查表 XOR**（bit 为 0 的行直接跳过）：

```
dim0 = basis(lane, k=0) ⊕ basis(lane, k=1) ⊕ basis(warp, k=1)
     = 1 ⊕ 2 ⊕ 8
     = 11
```

3. **结果**：逻辑坐标 `dim0 = 11`

#### 全零 basis = broadcast（一元素多 thread）

**§3.2**：tensor 比 tile **小** 时，多个 thread 持有 **同一** 逻辑元素。

编译器做法：把 **多余** 那些输入 bit 的 basis **改成全零**。

**例**（`ShapeSmallerThanLayout`）：`blocked({4},{4},{4})`，shape=`[8]`（tile 能盖 64 元素，只要 8 个）：

```
broadcast 前（概念上）          broadcast 后（实际存的）
lane:  bit0→(1)  bit1→(2)      lane:  bit0→(1)  bit1→(0)  ← bit1 失灵
warp:  bit0→(4)  bit1→(8)      warp:  bit0→(0)  bit1→(0)  ← 全失灵
register: (1), (2)              register: (1), (2)         ← 仍有效
```

**然后怎么办？** `apply()` **照常跑**，没有任何特殊分支：

| 硬件 | 有效 bit 贡献 | 逻辑 dim0 |
|------|--------------|-----------|
| reg=0, lane=0, warp=0 | 无 | **0** |
| reg=0, lane=1, warp=0 | lane bit0 → (1) | **1** |
| reg=0, lane=2, warp=0 | lane bit1 → **(0)**，无贡献 | **0** ← 与 lane=0 **相同** |
| reg=0, lane=3, warp=0 | (1)⊕(0) | **1** ← 与 lane=1 **相同** |

`lane=2` 和 `lane=0` 算出 **同一个** 逻辑坐标 → 就是 §3.2 的 **broadcast**（一元素多 thread）。  
basis 置零 **不是删硬件**，而是说：**这个 bit 随便怎么翻，都不改变「指着张量哪一格」**。

#### 追加非零 basis = wrap（一 thread 多元素）

**§3.2**：tensor 比 tile **大** 时，tile 沿维 **重复**。

编译器在 `register`（或 `offset`）维 **多加** 几个 **非零** basis：

```
shape=[128]，tile 只盖 16 元素 → register 追加：
  L(reg=16)=(16), L(reg=32)=(32), L(reg=64)=(64)
```

同一 `(lane, warp)`，reg 从 0 变 16 → 逻辑坐标 **+16** → 指向 tile 的 **第二份副本**。  
这就是 wrap：**用 register 高位 bit 表示「第几轮重复」**。

| 010 概念 | LinearLayout 做法 | 效果 |
|----------|-------------------|------|
| wrap | **加** 非零 basis | 翻某个输入 bit → 逻辑坐标 **变** → 指向另一份 tile |
| broadcast | 把 basis **改成 0** | 翻某个输入 bit → 逻辑坐标 **不变** → 多 hardware 同指一格 |

手算与单测：`020_linear_layout.md` §6.4、`§8.4`。

### 4.5 手算例 A：1D blocked，shape=`[16]`

> 教学用缩小版（`threadsPerWarp=4`）；生产 kernel 见 §6.1（`threadsPerWarp=32`）。golden：`020` §1.4 / `SimpleBlocked` 单测。

**IR 里的 bases**（1D 只有 `dim0`，向量退化成单个数）：

```mlir
#ttg.linear<{register = [],
             lane     = [[1], [2]],
             warp     = [[4], [8]],
             block    = []}>
```

**`[]` 不等于 `[[0], [0]]`**（常见误解）：

| 写法 | 该维有几个 bit | 含义 | 例子 A |
|------|----------------|------|--------|
| **`register = []`** | **0** | 这一维 **不存在**；`register` 恒为 0，每 thread **只有 1 个元素** | `sizePerThread=[1]` |
| **`register = [[0], [0]]`** | **2** | 有 2 个 register bit，但 basis 全零 → **broadcast**；翻 reg bit **不换**逻辑坐标 | 需 `sizePerThread=4` 且 broadcast |
| **`block = []`** | **0** | 只有 1 个 CTA，`block` 恒为 0 | 单 CTA、无 cluster |
| **`block = [[0], [0]]`** | **2** | 有 2 个 block bit，但全零 → 多个 block id **映到同一逻辑坐标** | shape 小于 tile 时的 broadcast |

**直觉**：`[]` = 「这维 **不用编号**」（大小为 1）；`[[0],…]` = 「这维 **有位要编**，但翻位 **不动** 张量下标」。  
单测对照：`SimpleBlocked` 用 `register={}`、`block={}`；`CTADuplication` 用 `block={{16},{0}}`（2 bit，其一 broadcast）；`ShapeSmallerThanLayout` 里 warp 会变成 `{{0},{0}}` 那种 **非空但全零**。

##### lane 4–31 是闲置还是 broadcast？

先分清 **三件事**：

| 问题 | 例子 A 的答案 |
|------|----------------|
| shape=`[16]` 与 tile=16 是否触发 broadcast？ | **否**。16 个 `(lane,warp)` 槽位 **各映不同元素**，无重复。 |
| layout 里有没有 lane 4–31？ | **没有**。`lane=[[1],[2]]` 只有 **2 bit** → lane 只能是 **0–3**，输入空间里 **不存在** lane 4–31。 |
| 物理 GPU warp 有 32 thread，其余 28 个呢？ | 见下表 |

**`threadsPerWarp=[4]`（layout）≠ `warp_size=32`（硬件）**：

| 层面 | 例子 A | 真实 Triton kernel（如 §6.1） |
|------|--------|------------------------------|
| blocked `threadsPerWarp` | **[4]**（教学缩小） | **[32]** |
| lane basis 行数 | 2 行 | 5 行 `[[1]…[16]]` |
| LL 的 `lane` 输入大小 | 4 | 32（须与 module 的 warp_size 一致） |

**三种「多出来的 thread」语义**（不要混）：

| 机制 | 何时发生 | lane 4–31 算什么 |
|------|----------|------------------|
| **① 未建模** | 教学例 `threadsPerWarp=4`，只定义 lane 0–3 | lane 4–31 **不在 layout 输入里**；既不是闲置也不是 broadcast，是 **没定义** |
| **② broadcast** | tensor **小于** tile，或 lane/warp **高位 bit 的 basis 置 0**（`ensureLayoutNotLargerThan`） | 多个 lane **故意**映 **同一** 逻辑元素；`apply(lane=2)` 可能与 `apply(lane=0)` 相同 |
| **③ 物理 warp 余量** | layout 只用到 4 lane，但 kernel 仍 launch `num_warps×32` 个 hardware thread | 编译器侧：有效 TTGIR 会把 lane 维 **扩到 32**，多出的 bit 通常 **basis=0 → 归入 ② broadcast**；这些 thread 仍参与 warp 同步，但对 tensor 是 **重复数据**，不是「空着不算」 |

**例子 A 结论**：在 **layout 数学** 里，16 个元素由 lane 0–3 × warp 0–3 **一对一** 分完，**没有 broadcast，也没有「闲置 slot」**。  
你问的 lane 4–31，是 **把 4-lane 的教学 layout 硬套到 32-lane 硬件** 时才出现的问题；真实编译路径会用 `threadsPerWarp=32`，或用 **零 basis 补齐** 成 broadcast。

单测对照：`ShapeSmallerThanLayout`（shape=8 < tile=64）才是显式 broadcast——`lane: [[4],[0]]`，`warp: [[0],[0]]`，多出来的 bit **翻不动** 逻辑坐标。

**`lane` 维：2 个 bit → 2 行 basis**

| bit 编号 \(k\) | 输入值（\(2^k\)） | 二进制 | basis `[dim0]` | 手算验证 |
|----------------|-------------------|--------|----------------|----------|
| **0** | `lane = 1` | `…001` | **1** | \(L(\text{lane}=1)=1\) |
| **1** | `lane = 2` | `…010` | **2** | \(L(\text{lane}=2)=2\) |

**`warp` 维：2 个 bit → 2 行 basis**

| bit 编号 \(k\) | 输入值（\(2^k\)） | 二进制 | basis `[dim0]` | 手算验证 |
|----------------|-------------------|--------|----------------|----------|
| **0** | `warp = 1` | `…001` | **4** | \(L(\text{warp}=1)=4\) |
| **1** | `warp = 2` | `…010` | **8** | \(L(\text{warp}=2)=8\) |

**组合**：`lane=3, warp=2`（`020` §1.4 里元素 11 所在的 thread）

```
lane=3  → 二进制 11  → bit0、bit1 都是 1
warp=2  → 二进制 10  → 只有 bit1 是 1

dim0 = 1 ⊕ 2 ⊕ 8 = 11
```

### 4.6 手算例 B：2D blocked，shape=`[16, 16]`

等价 blocked：`sizePerThread=[1,4], threadsPerWarp=[4,1], warpsPerCTA=[4,1], order=[1,0]`（dim1 上每 thread **4** 个元素 → register **2** 个 bit）。golden 见 `020_linear_layout.md` §4.4。

**IR 里的 bases**（输出是逻辑坐标 **`(dim0, dim1)`** = 张量下标 **`T[dim0, dim1]`**，见 §4.2）：

```mlir
#ttg.linear<{register = [[0, 1], [0, 2]],
             lane     = [[1, 0], [2, 0]],
             warp     = [[4, 0], [8, 0]],
             block    = []}>
```

#### 逐行读懂这段 IR

**整体结构**

```
#ttg.linear<{ 输入维名称 = [ 第0个bit的basis, 第1个bit的basis, … ], … }>
```

- 四个键 = LinearLayout 的四个 **输入维**（硬件坐标从哪来）。
- 每个键下面的 **每一行** = 该维 **某一个 bit 单独为 1** 时，逻辑坐标要 **加** 的偏移 `[dim0, dim1]`。
- 最终坐标 = 所有为 1 的 bit 对应行 **按分量 XOR**（dim0、dim1 各自 XOR）。

**分工总览**（谁管行、谁管列）：

| 输入维 | 行数 | 管哪个逻辑维 | 直觉 |
|--------|------|--------------|------|
| `register` | 2 | **只动 dim1（列）** | 同一 thread 里第几个元素，沿 **列** 排 |
| `lane` | 2 | **只动 dim0（行）** 低 bit | warp 里第几个 thread，沿 **行** 排 |
| `warp` | 2 | **只动 dim0（行）** 高 bit | 接在 lane 后面，继续沿 **行** 排 |
| `block` | 0 | — | 单 CTA，无 cluster |

看 basis 的 **哪一列非零** 就知道动行还是动列：`[?, 0]` 只动 dim0，`[0, ?]` 只动 dim1。

---

**第 1 行：`register = [[0, 1], [0, 2]]`**

| 行号 k | 仅 reg 的 bit k =1 时 | basis | 含义 |
|--------|------------------------|-------|------|
| 0 | `reg=1`（`01`） | `[0, 1]` | dim1 **+1**，dim0 不变 → **往右 1 列** |
| 1 | `reg=2`（`10`） | `[0, 2]` | dim1 **+2**，dim0 不变 → **往右 2 列**（单 bit 贡献） |

2 行 → 2 个 register bit → **4 个槽位** `reg ∈ {0,1,2,3}`：

| reg 槽位 | 逻辑坐标（固定 `lane=0,warp=0`） | 持有的元素 |
|----------|----------------------------------|------------|
| 0 | `(0, 0)` | `T[0, 0]` |
| 1 | `(0, 1)` | `T[0, 1]` |
| 2 | `(0, 2)` | `T[0, 2]` |
| 3 | `(0, 3)` | `T[0, 3]` |

同一 thread、同一 `(lane,warp)`，靠 **reg** 在 **一行上连续拿 4 个元素**。

---

**第 2 行：`lane = [[1, 0], [2, 0]]`**

| 行号 k | 仅 lane 的 bit k =1 时 | basis | 含义 |
|--------|-------------------------|-------|------|
| 0 | `lane=1` | `[1, 0]` | dim0 **+1** → 换到 **下一行**（行内第 1 个 thread） |
| 1 | `lane=2` | `[2, 0]` | dim0 **+2** |

2 行 → `lane ∈ {0,1,2,3}`。固定 `reg=0, warp=0`：

| lane | 逻辑坐标 | 元素 |
|------|----------|------|
| 0 | `(0, 0)` | `T[0, 0]` |
| 1 | `(1, 0)` | `T[1, 0]` |
| 2 | `(2, 0)` | `T[2, 0]` |
| 3 | `(3, 0)` | `T[3, 0]` |

---

**第 3 行：`warp = [[4, 0], [8, 0]]`**

lane 占满 dim0 的 bit0、bit1 后，warp 接 **bit2、bit3**：

| 行号 k | 仅 warp 的 bit k =1 时 | basis | 含义 |
|--------|-------------------------|-------|------|
| 0 | `warp=1` | `[4, 0]` | dim0 **+4**（\(2^2\)，接在 lane 2 bit 后） |
| 1 | `warp=2` | `[8, 0]` | dim0 **+8**（\(2^3\)） |

固定 `reg=0, lane=0`：

| warp | 逻辑坐标 | 元素 |
|------|----------|------|
| 0 | `(0, 0)` | `T[0, 0]` |
| 1 | `(4, 0)` | `T[4, 0]` |
| 2 | `(8, 0)` | `T[8, 0]` |
| 3 | `(12, 0)` | `T[12, 0]` |

---

**第 4 行：`block = []`**

空列表 = 没有 block 这一维的 bit → 整个 cluster 只有 **1 个 CTA**，`block` 恒为 0。

---

**dim0 的 bit 从哪来（拼接关系）**

```
dim0 的 bit:  [ bit0  bit1 | bit2  bit3 ]
               ←  lane   →  ←  warp  →
贡献的 basis:   +1  +2      +4  +8
```

所以 **`dim0 = lane 贡献 ⊕ warp 贡献`**（reg、block 不碰 dim0）。  
**`dim1 = register 贡献`**（lane、warp 不碰 dim1）。

在 bit 不重叠时，可暂时记成：`dim0 ≈ lane_id + warp_id×4`，`dim1 ≈ reg_id`（与 XOR 结果相同）。

---

**合起来：小 tile 里谁拿谁（ASCII）**

固定 `warp=0`，一个 warp 内 4 个 lane，每 lane 4 个 reg 槽位，填满 dim0×dim1 的 **4×4** 小块（CTA 内一块，完整 CTA 是 4 warp × 4 lane × 4 reg = 64 元素）：

```
         reg=0    reg=1    reg=2    reg=3
         T[*,0]   T[*,1]   T[*,2]   T[*,3]
lane=0   (0,0)    (0,1)    (0,2)    (0,3)
lane=1   (1,0)    (1,1)    (1,2)    (1,3)
lane=2   (2,0)    (2,1)    (2,2)    (2,3)
lane=3   (3,0)    (3,1)    (3,2)    (3,3)
```

`warp=1` 时整块在 dim0 上 **+4**（行从 4 起）：例如 `reg=3,lane=2,warp=1` → `(6,3)`，即 `T[6,3]`。

---

**一次完整手算**：`reg=3, lane=2, warp=1`

```
reg=3  → [0,1] ⊕ [0,2] = [0,3]
lane=2 → [2,0]
warp=1 → [4,0]
合计   → [6,3]  →  T[6, 3]
```

### 4.7 `order` 与 stride

**结论先说**：

| 问题 | 答案 |
|------|------|
| LinearLayout 里还有 `order` 吗？ | **没有同名参数**，但 **信息没丢**，编进 bases 里了 |
| 能当成 stride 吗？ | **简单 blocked 可以** 暂时当 stride；**一般情况** 应想成 **按 bit 的偏移表 + XOR** |

#### `order` 并没有消失，而是「烙」进 bases

`blocked` 的 `order=[1,0]` 含义（§2.2）：**dim1 变化更快（minor）**，dim0 更慢（major）。

转成 LinearLayout 时，`identityStandardND` 按 order **从 minor 维开始** 给 register/lane/warp **叠 bit**（`020` §4.2）：

```
order = [1, 0]
  → register 的低 bit 先铺在 dim1  →  basis 形如 [0, ?]
  → lane / warp 的 bit 铺在 dim0   →  basis 形如 [?, 0]
```

若改成 `order = [0, 1]`，同样是这些 `sizePerThread` / `threadsPerWarp`，**哪一维拿 minor bit** 会变，bases 的 **0/非零列分布** 也会变。

所以：

- **BlockedLayout**：用 `order` 参数 **声明** bit 先分给哪个 dim；
- **LinearLayout**：不再写 `order`，直接看 bases **哪一列非零**、**bit 谁前谁后**——那就是 order 的 **最终结果**。

例子 B 的 bases 本身就 **等价于** `order=[1,0]` 的一种编码，不是两套无关机制。

#### 和 stride 的关系：特例像，一般式不是

你写的：

```
dim0 ≈ lane_id * stride_lane + warp_id * warp_stride + …
```

在 **1D blocked、无 broadcast、各维 bit 在 dim0 上不重叠** 时成立，且 XOR = 普通加法：

```
dim0 = lane + warp * 4        （例子 A）
dim1 = reg                    （例子 B 里 register 只动 dim1）
```

这时每一行 basis 确实像 **stride**（翻一位输入，逻辑坐标加固定步长）：

| 输入 bit 翻转 | 像 stride |
|---------------|-----------|
| lane +1 | dim0 += **1** |
| lane +2 | dim0 += **2** |
| warp +1 | dim0 += **4** |
| reg +1 | dim1 += **1** |

编译器里甚至有 `actionAdditiveStrides`（`LayoutUtils.cpp`）——就是在认这种「可加性 stride」结构。

**但不能把 LinearLayout 定义成 stride**，因为：

| 情况 | stride 直觉 |
|------|-------------|
| 2D basis `[1,1]`（swizzle） | 翻一个 bit **同时** 动 dim0 和 dim1 → 没有单一 `lane_stride` |
| broadcast（basis 置零） | 某个「stride = 0」，多 id 映到同一点 |
| MMA / 非标准 layout | bit 分配不规则 |

更稳的记法：

```
order   →  bit 先分给哪个 dim（minor → major）→ 体现在 bases 哪列为零、谁先叠
basis   →  翻某一个输入 bit 时，各 dim 上加多少（灵敏度 / 局部 stride）
apply   →  把所有为 1 的 bit 的 basis 做 XOR 求和
```

**简单 blocked**：order 决定分工，basis 数值就是各 bit 的 stride，XOR 碰巧等于加法和。  
**一般 LinearLayout**：只保留 basis + XOR，order 已消化在表里；swizzle 等 **连加法 stride 都不是**。

### 4.8 与 §3、`blocked` 及 `020` 的分工

- **与 §3 的 tile/slot**：§3 用“tile 表 $L$”描述一个有限小块里的映射；`LinearLayout` 用“basis”描述 **同一张表**，但方向相反（硬件→逻辑），适合编译器做 XOR 组合与 `convert_layout`。
- **与 `blocked`**：`blocked` 的 `sizePerThread/threadsPerWarp/warpsPerCTA/order` 规定 tile 内 bit 如何分给 register/lane/warp；编译器通过 `BlockedEncodingAttr::toLinearLayout()` 转成 bases。手推过程见 `020_linear_layout.md` §4。

#### 何时需要读 020

| 你想… | 读本文 | 读 020 |
|--------|--------|--------|
| 看懂 TTGIR 里 `#blocked` / `#nvidia_mma` 参数 | §6–§8 | — |
| 手算 tile 表里某元素在哪个 thread | §3、§6 | — |
| 手推 bases、`apply()` 验证 | §4 直觉 | §4、§8.4 |
| 理解 `convert_layout` / reg→shmem | — | §7 |
| 查 14 种 encoding→LL 转换 | §5.2 | §5 + 附录 A |

**本章小结**：§4.2–4.3 学会 **读 bases**；§4.4 把 §3 的 wrap/broadcast 对应到 **置零 / 追加 basis**；§4.5–4.6 用两个 blocked 例 **手算验证**；细节推导与单测见 **020**。

---

<!-- Part C: 参考 -->

## 5. Layout 全景：总共有多少种 Layout

> 本节提供全局视角，回答「一共有哪些 layout」「分别在什么场景用」「代码中如何派发」。

### 5.1 总览

Triton 当前共有 **14 种**真正参与数据排布的 layout encoding（排除 `CTAEncodingAttr` 等辅助类型），分为三大类：

| 类别 | 数量 | 存储位置 | **人读语义**（§3 视角） | **LinearLayout 方向**（编译器） |
|------|------|----------|-------------------------|--------------------------------|
| **Distributed** | 7 | 寄存器（每 thread 私有） | 逻辑坐标 → 哪个 thread 持有 | **硬件** `(reg,lane,warp,block)` → **逻辑** `(dim0,…)` |
| **Shared** | 5 | `ttg.local_alloc` | 逻辑坐标 → shared 字节/元素偏移 | **硬件** `(offset,block)` → **逻辑** `(dim0,…)` |
| **TMEM** | 2 | Tensor Memory（Hopper+） | 逻辑坐标 → TMEM 偏移 | **硬件** TMEM 位置 → **逻辑** `(dim0,…)` |

> **不要混两列**：左列是你在 §3 **手算 tile 表**时的方向；右列是 `LinearLayout` / `apply()` 的 **唯一正向定义**。同一张表，方向相反、表示不同（见 `020` §1）。

#### LinearLayout 扮演什么角色？`toLinearLayout` 在干什么？

可以把 **14 种 encoding** 想成 **14 套人类/领域参数**（`sizePerThread`、`perPhase`、`instrShape`…），而 **LinearLayout** 是编译器里的 **统一中间表示（IR）**：

```
TTGIR 张量类型上的 #blocked / #swizzled_shared / #nvidia_mma / …
        │  各 encoding 自己的参数字段（语义各异）
        ▼
  toLinearLayout(shape)          ← LinearLayoutConversions.cpp 派发
        │  每种 *::toLinearLayout() 把参数「编译」成 bases
        ▼
  同一个 LinearLayout 对象
        │  输入维名因存储位置而异，输出永远是 dim0, dim1, …
        ▼
  apply() / invertAndCompose() / convert_layout codegen
```

**三个角色**（一句话记）：

| 角色 | 含义 |
|------|------|
| **统一数学语言** | 不管 Distributed / Shared / TMEM，都用 **basis + XOR** 描述排布 |
| **Lowering 桥梁** | 把 14 种 `#ttg.*` encoding **降级**成 codegen 能消费的同一种结构 |
| **可组合代数** | `operator*`、`invertAndCompose` 等在 **LinearLayout 层**做 layout 转换，不必为每种 encoding 写特例 |

**为什么 LinearLayout 固定为「硬件 → 逻辑」？**（`LinearLayout.h` 注释）

- codegen 典型问题：**这个 thread / 这个 shared offset 里装的是 `T[i,j]` 的哪个元素？** → 正向函数自然。
- 反向「`T[i,j]` 在哪个 thread？」可能 **一对多**（broadcast）→ 若把正向定义成逻辑→硬件，就不是函数了，代数更难做。

**`toLinearLayout` 不是换一种 layout 语义**，而是 **同一种语义的另一种写法**：

| 阶段 | 你看到的 | 方向 |
|------|----------|------|
| 读 IR / §3 手算 | `#blocked` 参数、tile 表 | 常按 **逻辑 → 硬件** 理解 |
| `toLinearLayout` 之后 | `bases` + `apply(reg,lane,…)` | 固定 **硬件 → 逻辑** |
| `invert()` / `invertAndCompose` | 需要时再做反向 | 在 LL 层求逆 |

**输入维因存储位置不同**（同一张映射表，不同的「硬件坐标」定义）：

| 存储 | LinearLayout 典型输入维 | 输出维 |
|------|-------------------------|--------|
| Distributed | `register`, `lane`, `warp`, `block` | `dim0`, `dim1`, … |
| Shared | `offset`, `block`（有时还有迭代维） | `dim0`, `dim1`, … |
| TMEM | TMEM 专用 offset 维 | `dim0`, `dim1`, … |

所以：**Distributed / Shared / TMEM 是「排布发生在哪、参数长什么样」**；**LinearLayout 是「编译器里怎么统一表示和变换这些排布」**。`#ttg.linear` encoding 则是 **跳过专用参数、直接在 IR 里存 bases** 的那一种（见 §6.2）。

代码入口：`TritonGPUDialect::toLinearLayout()` 的 dispatch（`LinearLayoutConversions.cpp:1183`）：

```cpp
// 1190–1219
if (auto distributed = dyn_cast<DistributedEncodingTrait>(layout)) {
  result = distributed.toLinearLayout(shape);   // 7 种 distributed
} else if (auto shared = dyn_cast<SwizzledSharedEncodingAttr>(layout)) { ... }
  else if (auto shared = dyn_cast<SharedLinearEncodingAttr>(layout)) { ... }
  else if (auto shared = dyn_cast<NVMMASharedEncodingAttr>(layout)) { ... }
  else if (auto sbl = dyn_cast<AMDRotatingSharedEncodingAttr>(layout)) { ... }
  else if (auto tensorMemoryEncoding = ...) { ... }
  else if (auto tensorMemoryScalesEncoding = ...) { ... }
```

#### Lowering 时能不能「直接用硬件参数算访存地址」？

**可以这么理解大方向**，但要拆成 **两段**，不是 `#blocked` 参数一步到位：

```
编译期                          运行期 / codegen
─────────                       ───────────────────────────────────
#blocked 等 encoding
    → toLinearLayout(shape)     得到 laneId, warpId, blockId, reg …
    → bases（常量）                    │
                                     ▼
                              applyLinearLayout(ll, 硬件坐标)
                                     │
                                     ▼
                              逻辑下标 (dim0, dim1, …)    ← LL 只到这里
                                     │
                                     ▼
                              + 基址指针、stride、tile 偏移、program_id …
                                     │
                                     ▼
                              global / shared / TMEM 的 **物理地址**
```

| 阶段 | 谁算 | 算什么 |
|------|------|--------|
| **① LL** | `apply()` / `emitIndices()` | `(reg,lane,warp,block)` 或 `(offset,block)` → **`T[dim0,dim1,…]` 是哪一个元素** |
| **② 访存** | load/store、`local_alloc` 等 lowering | 逻辑下标 + **指针算术** → 字节地址 |

典型代码路径（`TritonGPUToLLVM/Utility.cpp`）：

1. **`emitIndices`**：读当前 thread 的 `laneId`、`warpId`、`blockId`，枚举 `register` 槽位 → 调 **`applyLinearLayoutVec`**
2. **Global load/store**：用上一步的 `(dim0,…)` 去算 `ptr + linear_index`（还有 `program_id`、tensor 全局偏移等）
3. **Reg ↔ Shared**（`convert_layout`）：`regLayout.invertAndCompose(sharedLayout)` → 先统一到逻辑空间，再落到 **shared offset**（store/load 各走一半）

所以：

- **对**：lowering 不再为每种 encoding 手写一套「thread id → 下标」；统一 **`toLinearLayout` + `apply`**，硬件坐标进来，**逻辑元素下标**出去。
- **不完全对**：LL **不负责**最终字节地址；shared 的 swizzle、global 的 coalescing、TMEM 的行列编码，还在 **第二段** 用指针/offset 公式完成。
- **寄存器 tile**：很多 op 只用到 ①——「这个 thread 的 reg 槽里应是 `T[i,j]`」，值已在寄存器里，不涉及 global 地址。

**一句话**：LinearLayout 在 lowering 里解决的是 **「硬件上的这个位置，对应张量哪一格」**；**「那一格在内存里地址多少」** 是下一步指针运算。两步合起来，才是完整的访存地址生成。

#### 以前怎么算？现在统一到什么程度？

**以前（各 encoding 各写一套）**：codegen 里直接读 `#blocked` / `#nvidia_mma` 等 **参数字段**，按 `order` 做 `linearize` / `delinearize`，手写「thread id → 多维下标」：

```
旧思路（概念上）：
  threadId = f(lane, warp)
  offset[d] = (threadId 按 order 拆开) × sizePerThread[d] × …   // 每种 encoding 公式不同
  ptr + linearize(offset, stride)
```

每种 layout 在 **TritonGPUToLLVM** 里常有 **独立分支**（blocked 一套、MMA 一套、shared swizzle 又一套）；`getShapePerCTA`、`getElemsPerThread`、`getOrder()` 等工具函数到处散落。`LinearLayout.h` 注释写明长期目标：**逐步去掉这些专用 layout 分支**。

**现在（主路径已统一）**：TTGIR → LLVM 的 **索引 / layout 映射** 基本都走同一条路：

```
encoding + shape  →  toLinearLayout()  →  applyLinearLayout() / emitIndices()
                                              ↓
                                        逻辑 (dim0, dim1, …)
                                              ↓
                                        + 基址 / stride  →  访存地址
```

| 环节 | 现在 | 说明 |
|------|------|------|
| **寄存器 tile 下标** | `emitIndices` → `toLinearLayout` + `apply` | 主路径 ✅ |
| **compile-time offset 表** | `emitOffsetForLayout` | 内部已调 `ll.apply()` ✅ |
| **reg ↔ shared** | `invertAndCompose` | ✅ |
| **global load/store（NVIDIA）** | `LoadStoreOpToLLVM` + `applyLinearLayout` | ✅ |
| **IR 里的 encoding 种类** | 仍 14 种 `#ttg.*` | 未删；只是 codegen **先转 LL** |
| **少数遗留** | 如 `ScanOpToLLVM`（仍用 `order`+`linearize`）、`FMADotUtility`（直接读 `BlockedEncodingAttr`） | 注释里写「应改成 emitOffsetForLayout」类 TODO |

所以：

- **「现在的 lowering 是不是基本统一了 LinearLayout？」** → **对主路径而言：是。** 算「这个 thread / 这个 smem offset 对应哪一格」已统一为 `toLinearLayout` + `apply`；LLVM 阶段 **不再直接读** `sizePerThread` 等参数（见 `020` §7.5）。
- **「是不是所有东西都已 LL 化？」** → **还没有。** IR 层仍保留 14 种 encoding；个别 op（scan、FMA dot）和 analysis 工具仍碰 encoding 字段；最终字节地址仍要 **指针算术** 第二段。
- **趋势**：新 pass / Gluon 倾向直接用 `#ttg.linear` 存 bases，跳过「专用参数 → toLinearLayout」中间层。

### 5.2 逐表：所有 Layout 属性

#### Distributed（7 种）— 寄存器排布

| # | 编码类型 | IR mnemonic | GPU 目标 | `toLinearLayout` 入口 | 使用场景 |
|---|----------|-------------|----------|----------------------|----------|
| 1 | `BlockedEncodingAttr` | `#ttg.blocked` | 通用 | `BlockedEncodingAttr::toLinearLayout()` (L850) | 最常用的 global load/store；coalescing 友好，vector-add、matmul 准备 |
| 2 | `LinearEncodingAttr` | `#ttg.linear` | 通用 | `LinearEncodingAttr::toLinearLayout()` (.td) | 直接用 LinearLayout 作为 encoding；新 pass 方向，逐步替代专用 encoding |
| 3 | `SliceEncodingAttr` | `#ttg.slice` | 通用 | `SliceEncodingAttr::toLinearLayout()` (L1014) | `expand_dims` 的逆变换，优化 pass 中 squeeze 一维 |
| 4 | `DotOperandEncodingAttr` | `#ttg.dot_op` | 通用* | `DotOperandEncodingAttr::toLinearLayout()` (L1000) | `tt.dot` 的 A/B 操作数；`parent` 为 output layout |
| 5 | `NvidiaMmaEncodingAttr` | `#ttg.nvidia_mma` | NVIDIA (Volta+) | `NvidiaMmaEncodingAttr::toLinearLayout()` (L943) | Tensor Core 输出 C 矩阵；版本编码代际 |
| 6 | `AMDMfmaEncodingAttr` | `#ttg.amd_mfma` | AMD (CDNA) | `AMDMfmaEncodingAttr::toLinearLayout()` (L321) | MFMA matrix-core 输出 C |
| 7 | `AMDWmmaEncodingAttr` | `#ttg.amd_wmma` | AMD (RDNA) | `AMDWmmaEncodingAttr::toLinearLayout()` (L644) | WMMA matrix-core 输出 C |

\* `dot_op` 按 `parent` 类型派发到四个子函数：`fmaDotToLinearLayout()`(blocked parent)、`mfmaDotToLinearLayout()`、`wmmaDotOperandToLinearLayout()`、`nvidiaDotToLinearLayout()`。

#### Shared（5 种）— 共享内存排布

| # | 编码类型 | IR mnemonic | GPU 目标 | `toLinearLayout` 入口 | 使用场景 |
|---|----------|-------------|----------|----------------------|----------|
| 8 | `SwizzledSharedEncodingAttr` | `#ttg.swizzled_shared` | 通用 | `swizzledSharedToLinearLayout()` (L56) | 最常用的 shared memory；XOR swizzle 解决 bank conflict |
| 9 | `PaddedSharedEncodingAttr` | `#ttg.padded_shared` | 通用 | **无** — 直接存 `LinearLayout` 于 `getLinearComponent()` | 间隔 padding + 线性重映射 |
| 10 | `SharedLinearEncodingAttr` | `#ttg.shared_linear` | 通用 | `SharedLinearEncodingAttr::toLinearLayout()` (Dialect.cpp:1776) | LinearLayout 直接描述 shared 偏移；灵活性最高 |
| 11 | `NVMMASharedEncodingAttr` | `#ttg.nvmma_shared` | NVIDIA (Hopper+) | `nvmmaSharedToLinearLayout()` (L203) | WGMMA/MMA v3/v5 的 shared 输入 staging；2D 块状 shared |
| 12 | `AMDRotatingSharedEncodingAttr` | `#ttg.amd_rotating_shared` | AMD | `sharedToLinearLayoutAMDRotating()` (L104) | 类 swizzle 但 phase 与 block 号 XOR，AMD 专用 |

#### TMEM（2 种）— Tensor Memory 排布（Hopper+ SM90+）

| # | 编码类型 | IR mnemonic | GPU 目标 | `toLinearLayout` 入口 | 使用场景 |
|---|----------|-------------|----------|----------------------|----------|
| 13 | `TensorMemoryEncodingAttr` | `#ttng.tensor_memory_encoding` | NVIDIA (Hopper+) | `tensorMemoryToLinearLayout()` (L1038) | Tensor Memory (TMEM) 中的主数据排布 |
| 14 | `TensorMemoryScalesEncodingAttr` | `#ttng.tensor_memory_scales_encoding` | NVIDIA (Hopper+) | `tensorMemoryScalesToLinearLayout()` (L1130) | TMEM 中 MMAv5 的 scale factor 排布 |

### 5.3 快速场景索引

| 你想做什么 | 用哪个 layout | 详见 |
|-----------|-------------|------|
| Global load/store（最常用） | `blocked` | §6.1 |
| 灵活的寄存器排布（新 pass） | `linear` | §6.2 |
| `expand_dims` 逆变换 | `slice` | §6.3 |
| matmul A/B 操作数（旧 MMA） | `dot_op` | §8.1 |
| Tensor Core 输出（NVIDIA） | `nvidia_mma` | §8.2（语义）、`020` §5.4.1（bases） |
| MFMA 输出（AMD CDNA） | `amd_mfma` | §8.3 |
| WMMA 输出（AMD RDNA） | `amd_wmma` | §8.4 |
| 通用 shared memory staging | `swizzled_shared` | §7.1 |
| padding 对齐的 shared | `padded_shared` | §7.2 |
| LinearLayout 直接描述 shared | `shared_linear` | §7.3 |
| Hopper+ WGMMA shared 输入 | `nvmma_shared` | §7.4 |
| AMD rotating shared | `amd_rotating_shared` | §7.5 |
| Tensor Memory 主数据 | `tensor_memory_encoding` | §9.1 |
| TMEM 量化 scales | `tensor_memory_scales_encoding` | §9.2 |

### 5.4 与前后章的关系

- **前**：§4 是 LinearLayout 入门（basis、手算例）；实现细节在 `020_linear_layout.md`。
- **后**：§6–§9 逐一详解各 encoding 的 **语义与参数**；**bases 怎么算**见 020 §5 与附录 A。
- **代码**：所有 encoding 统一经 `TritonGPUDialect::toLinearLayout()`（`LinearLayoutConversions.cpp:1183`）转成 `LinearLayout` 再 codegen。

---

## 6. Distributed Layout 详述

前提：§2 层次 + §3 映射。以下回答「IR 里 `#ttg.xxx` 各字段是什么」。

### 6.1 BlockedEncodingAttr（`#ttg.blocked`）— 最常用

**用途**：寄存器 tensor；利于 global load/store **coalescing**。

**参数**：

| 参数 | 对应层次 | 含义 |
|------|----------|------|
| `sizePerThread` | Values per thread | 每 thread 每维持有元素数 |
| `threadsPerWarp` | Threads per warp | 每 warp 每维 thread 数 |
| `warpsPerCTA` | Warps per CTA | 每 CTA 每维 warp 数 |
| `order` | — | 最快变化维在前 |
| `CTALayout` | CTAs per CGA | 多 block 切分（§2.3） |

**tile 大小**（每 CTA 覆盖的元素，逐维乘）：

$$
\text{tile\_shape}[d] = \text{sizePerThread}[d] \times \text{threadsPerWarp}[d] \times \text{warpsPerCTA}[d]
$$

#### 示例 1：2D blocked（文档经典）

```mlir
#ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
```

16×16 张量、2 warp 时，数字为 thread id，`;` 分隔 warp：

```
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
...
```

#### 示例 2：1D blocked（vector-add TTGIR）

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
```

| 字段 | 值 | 含义 |
|------|-----|------|
| `sizePerThread` | `[1]` | 每 thread 1 个元素 |
| `threadsPerWarp` | `[32]` | 每 warp 32 thread 铺满 1D |
| `warpsPerCTA` | `[8]` | 8 warp |
| **tile 大小** | $1\times32\times8=256$ | 对应 `BLOCK=256` |

与 host / module 一致：

```python
vector_add_kernel[grid](..., BLOCK=256, num_warps=8)  # 8 = warpsPerCTA
```

```mlir
"ttg.num-warps" = 8
%v = tt.load ... : tensor<256xf32, #blocked>
%idx = tt.make_range {end = 256, start = 0} : tensor<256xi32, #blocked>
```

这是 §4.5 教学例的 **1D 缩小版**；生产路径用 `threadsPerWarp=[32]`（不是另一种 layout）。

> **bases 对照**：上述参数经 `toLinearLayout` 得到 `lane: {{1},{2}}`，`warp: {{4},{8},…}`（视 warps 而定）。手推见 `020_linear_layout.md` §4.3；golden 见 `LinearLayoutConversionsTest::SimpleBlocked`。

---

### 6.2 LinearEncodingAttr（`#ttg.linear`）

直接用 **§4** 的 `LinearLayout` 作为 encoding；新 pass 方向，逐步减少专用 encoding 分支。

| 参数 | 含义 |
|------|------|
| `linearLayout` | 一个 `LinearLayout` 实例，描述 register/lane/warp/block 到逻辑坐标的映射 |

**用途**：当 `blocked` 等高级 encoding 无法表达所需排布时（如自定义数据重排），可用 `linear` 直接对 basis 操作。编译器内部经常先将 `blocked` 等转为 `LinearLayout` 再 codegen。

Gluon 侧对应 `DistributedLinearLayout`；IR ↔ Python 见 `020_linear_layout.md` §5.7、§7.6。

---

### 6.3 SliceEncodingAttr（`#ttg.slice`）

从 `parent` layout **squeeze** 一维；用于 `expand_dims` 的逆变换（优化 pass）。参数：`dim`、`parent`。

---

**本章小结**：日常见最多的是 `blocked`；手算例见 **§4.5–§4.6**，参数→bases 见 **020 §4**；读 TTGIR 先对 `sizePerThread × threadsPerWarp × warpsPerCTA` 算 tile，再用 §3 判断 wrap/broadcast。

---

## 7. Shared Memory Layout

前提：数据在 **shared**，映射目标是 **偏移**，不是 thread id。用于 `ttg.local_alloc`、dot 操作数 staging；主要解决 **bank conflict**。

| mnemonic | 用途 | 平台 |
|----------|------|------|
| `swizzled_shared` | 通用 XOR swizzle；Ampere/Hopper dot 常用 | 通用 |
| `padded_shared` | padding + 线性变换 | 通用 |
| `shared_linear` | LinearLayout 直接描述偏移 | 通用 |
| `nvmma_shared` | MMAv3/v5 warpgroup matrix 专用格式 | NVIDIA Hopper+ |
| `amd_rotating_shared` | rotating swizzle，phase 与 block 号 XOR | AMD |

### 7.1 `swizzled_shared`

#### 目的：解决 shared memory 的 bank conflict

shared memory 物理上分成多个 **bank**（可理解为多条并行“小内存通道”）。一个 warp 在同一条指令里访问 shared 时：

- 如果每个线程落在**不同 bank**（或可被硬件广播的同地址），可以并行完成；
- 如果很多线程落在**同一个 bank 的不同地址**，就会发生 **bank conflict**，访问被拆成多次，等价于变慢。

冲突的根源是：shared 的 **bank 选择**通常只由地址的**低位**（按字节/按元素）决定；而很多 dot staging 的访问模式在逻辑上是“整齐的”（例如同一行、相邻列、固定步长），这些访问很容易让 bank 低位呈现强相关，导致大量线程撞到同一个 bank。

`swizzled_shared` 的做法就是：在把 `(row,col)` 线性化成 shared offset 之前，先对 `col` 做一个依赖 `row` 的 XOR 置换（`outCol = inCol ^ phase`，或 `vec` 分块版本）。这样：

- **同一行内部**：列号被“打散”，同一个 warp 里原本可能集中在某些 bank 的访问，被分配到更多 bank；
- **相邻行之间**：通过 `perPhase/maxPhase` 控制 `phase` 的变化速度与周期，让“打散模式”随行变化，避免固定模式反复撞 bank；
- **向量化访问不被打断**：通过 `vec` 把连续 `vec` 个元素绑成一块移动，保持块内连续/对齐，同时在块维度上打散 bank。

因此它不改变你要访问哪些逻辑元素，只改变这些元素在 shared 中的**物理地址分布**，用更均匀的 bank 使用来降低冲突。

更具体一点：把 shared 视为按 **word**（通常 4B）交错到 bank 上。简化写法：

$$
bank(addr) = \left\lfloor \frac{addr}{wordBytes} \right\rfloor \bmod numBanks
$$

如果一个 warp 的 32 个线程在同一条指令里访问的地址满足 `bank(addr_t)` 大量重复（且地址不同），就会变成多路串行（冲突阶数 = 该 bank 被命中的次数）。

对一个 2D 行主序对象（忽略 padding），`addr(row,col)=base + (row*stride + col)*elemBytes`。当访问模式是“同一 row，不同 col 的连续段”时，`bank` 通常均匀；但 dot staging 里常见的模式是“同一 warp 同时覆盖多个 row，且 col/stride 与 bank 低位强相关”（例如跨行、转置式读、或固定步长采样），这时很多线程会落到同一组 bank。

XOR swizzle 的关键点是：它改的是 `col` 的**低位组合**，而 bank 选择也主要看地址低位，因此能直接打散 `bank(addr)` 的重复。

- 对 `vec=1`：`outCol = inCol ^ phase`。当同一 warp 里线程来自不同 `row` 时，`phase` 不同 → 同一个 `inCol` 会被映射到不同 `outCol`，从而打散到不同 bank。
- `perPhase`：控制“多少行共享同一个 phase”。如果行数很小/很密，`perPhase>1` 可以让相邻行先共享同一个模式（避免过度打散破坏局部性），同时仍保持跨更大行跨度的打散。
- `maxPhase`：控制 phase 的取值范围与周期。太小会让很多行重复同一 swizzle（打散不够）；增大则提供更多不同的列置换，降低重复命中同一 bank 的概率。
- `vec`：把连续 `vec` 个元素视为不可拆开的块，只对块号做 XOR：`out=((col/vec)^phase)*vec + (col%vec)`。这样可以在保持向量化/对齐的前提下，把“块”在 bank 维度上打散。

直观理解：不 swizzle 时，不同 row 的相同 `col` 往往会落在同一组 bank；swizzle 让这些 row 的 `col` 依赖 `row` 发生变化，从而把同一时刻的访问从“撞同一 bank”变成“分散到更多 bank”。

| 参数 | 含义 |
|------|------|
| `vec`, `perPhase`, `maxPhase` | swizzle 强度 |
| `order` | 最快变化维在前 |
| `CTALayout` | 多 CTA |

**先记住一件事**：swizzle **只改“列放在哪”**，不改行号。逻辑上元素在 `(row, col)`，写入 shared 时落到列 `outCol`：

$$
outCol = inCol \oplus phase,\quad phase = (inRow / perPhase) \bmod maxPhase
$$

（`vec>1` 时见下文，先把 `vec=1` 看懂。）

下面都用 4×4 小矩阵，单元格是**逻辑编号**（不 swizzle 时就是行主序 0..15）：

```
不 swizzle（对照）          row0:  0  1  2  3
                           row1:  4  5  6  7
                           row2:  8  9 10 11
                           row3: 12 13 14 15
```

`phase` 可以理解成：**这一行要把列整体“翻转/交换”多少位**（XOR 一个整数到列号上）。`perPhase` / `maxPhase` 只决定 **phase 随 row 怎么变**。

#### `perPhase`：多少行用同一种“列交换方式”

| row | `perPhase=1` 时 phase | 效果 |
|-----|----------------------|------|
| 0 | 0 | 列不动 |
| 1 | 1 | 列 0↔1、2↔3 互换（XOR 1） |
| 2 | 2 | 列再换一档（XOR 2） |
| 3 | 3 | … |

`perPhase=1` → **每换一行，交换方式就变一次**（最“密”）。

| row | `perPhase=2` 时 phase（`maxPhase=4`） | 效果 |
|-----|--------------------------------------|------|
| 0,1 | 0 | 两行都**不交换** |
| 2,3 | 1 | 两行都按 XOR 1 交换 |

`perPhase=2` → **每 2 行才换一种交换方式**；所以 row0/1 看起来和不 swizzle 一样，row2/3 才乱。

对应排布（`vec=1, perPhase=2, maxPhase=4`）：

```
row0:  0  1  2  3    ← phase=0，和不 swizzle 一样
row1:  4  5  6  7
row2:  9  8 11 10    ← phase=1，列 0↔1、2↔3
row3: 13 12 15 14
```

#### `maxPhase`：一共有几种不同的“列交换方式”，然后循环

`maxPhase=4`：phase 依次为 0,1,2,3,0,1,2,3,…（4 种方式才轮回一次）。

`maxPhase=2`（`perPhase=1`）：phase 只有 0,1,0,1,…（只有“不交换”和“XOR 1”两种，更温和）：

```
vec=1, perPhase=1, maxPhase=2:
row0:  0  1  2  3    phase=0
row1:  5  4  7  6    phase=1
row2:  8  9 10 11    phase=0  ← 又回到“不交换”
row3: 13 12 15 14    phase=1
```

对比 `maxPhase=4`（同样 `perPhase=1`）：row2 是 phase=2，所以和 row0 **不一样**：

```
vec=1, perPhase=1, maxPhase=4:
row0:  0  1  2  3
row1:  5  4  7  6
row2: 10 11  8  9    ← phase=2，比 maxPhase=2 时更“花”
row3: 15 14 13 12
```

**小结**：`perPhase` 控制 phase **多久变一次**；`maxPhase` 控制 phase **最多有几种、多久轮回**。

#### `vec`：按“连续 vec 列”成块再交换（块内相对顺序不变）

把列按 `vec` 切成块：`[0,1]`、`[2,3]`、`[4,5]`、`[6,7]`… **只对“第几块”做 XOR**，块里的 `(col % vec)` 不变。

- `vec=1`：一块 = 1 列，等价于上面公式。
- `vec=2`：一块 = 2 列，**0 和 1 永远绑在一起移动**，2 和 3 绑在一起，…

`vec=2, perPhase=1, maxPhase=4` 时，row1 的 phase=1，块号 XOR 1：

```
逻辑列块:  [0,1] [2,3] [4,5] [6,7]
row0:       0  1   2  3   4  5   6  7     phase=0
row1:      10 11   8  9  14 15  12 13     phase=1 → 块0↔块1、块2↔块3（每块内 0,1 顺序不变）
```

**为什么需要 `vec`**：MMA/dot 常一次读 2/4 个连续元素；若 `vec=1` 把单个元素打散，向量化 load 会断；`vec` 保证**连续 vec 个元素仍相邻**，同时在更粗的“块”维度上打散 bank。

**三个参数合起来**：先由 `perPhase`、`maxPhase` 定本行的 `phase`；再由 `vec` 决定是按“列”还是按“vec 列一块”去做 `col XOR phase`；最终得到 shared 里的物理排布，减轻 bank conflict。

> **编译器表示**：swizzle 被编码进 `offset` 各 bit 的 basis（row 方向 basis 的 col 分量带 XOR）。手算见 `020_linear_layout.md` §5.5。

### 7.2 `padded_shared`

`intervals` / `paddings`：每隔若干元素插 padding（均为 2 的幂）。可带 `linearComponent` 做坐标重排。

注意 `padded_shared` **不经过** `TritonGPUDialect::toLinearLayout()` 派发，其 LinearLayout 直接存于 `getLinearComponent()` 中。

### 7.3 `shared_linear`

LinearLayout 直接描述 shared 偏移。参数：

| 参数 | 含义 |
|------|------|
| `linearLayout` | `LinearLayout` 实例（输入维为 `offset`） |
| `layoutAlignment` | 对齐要求 |

**用途**：当 swizzle/padding 等高级 encoding 无法满足排布需求时，可用 `shared_linear` 精确控制每一位到偏移的映射。映射目标不再是 thread id 而是 **shared 地址偏移**（字节级）。

### 7.4 `nvmma_shared`

Hopper+ WGMMA 输入；`swizzlingByteWidth` ∈ {0,32,64,128} 由连续维字节数推导。

**与 §6 关系**：global load 常用 `blocked` → 写入 `swizzled_shared` / `nvmma_shared` → MMA 读 shared。

---

### 7.5 `amd_rotating_shared`

AMD GPU 特有的 rotating swizzle。与 `swizzled_shared` 类似（同样有 `vec`、`perPhase`、`maxPhase`、`order`），但 phase 的计算方式不同：

```
phase = ((row / perPhase) ^ blockId) % maxPhase
```

即 phase **额外与 block 号做 XOR**，使得不同 CTA 在同一 row 上使用不同的 swizzle pattern，进一步提升多 CTA 场景下的 bank conflict 缓解效果。

**用途**：AMD CDNA 架构的 matmul dot staging。

---

## 8. MMA / Dot Layout

MMA layout 描述 **矩阵乘** 在寄存器里的特殊排布：与 `blocked` 不同，tile 几何由 **Tensor Core 指令** 固定，不由 `sizePerThread` 自由组合。

| mnemonic | 角色 | 平台 |
|----------|------|------|
| `dot_op` | MMA 的 A/B 操作数（`opIdx` 0/1，`kWidth`，`parent`=输出 layout） | 通用（派发至各 parent） |
| `nvidia_mma` | Tensor Core **输出 C**（`versionMajor` 1=Volta，2=Turing+） | NVIDIA |
| `amd_mfma` | MFMA matrix-core **输出 C** | AMD CDNA |
| `amd_wmma` | WMMA matrix-core **输出 C** | AMD RDNA |

**与 §3 的公共模型**：

| 概念 | blocked（§6） | MMA 输出（§8.2–8.4） |
|------|--------------|---------------------|
| tile 来源 | `sizePerThread × threadsPerWarp × warpsPerCTA` | `instrShape` 的 M×N（+ warp 扩展） |
| wrap | tensor 大于 tile → 重复 tile | 相同；大矩阵靠 register/warp 高位 bit 重复 |
| broadcast | tensor 小于 tile → 多 thread 同元素 | 相同；`warpsPerCTA` 不足时常见 |

**bases 手算**（编译器视角）统一见 `020_linear_layout.md` §5.4.1（NVIDIA）及该节 AMD 简述。

### 8.1 `dot_op` — MMA 操作数

`DotOperandEncodingAttr` 本身是通用容器，其具体排布由 `parent` 类型决定（`DotOperandEncodingAttr::toLinearLayout()` L1000）：

```cpp
if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(parent)) {
  return fmaDotToLinearLayout(*this, shape);         // FMA-based dot
} else if (auto mfmaLayout = mlir::dyn_cast<AMDMfmaEncodingAttr>(parent)) {
  return mfmaDotToLinearLayout(*this, shape);         // AMD MFMA operand
} else if (auto wmmaLayout = mlir::dyn_cast<AMDWmmaEncodingAttr>(parent)) {
  return wmmaDotOperandToLinearLayout(*this, shape);  // AMD WMMA operand
} else {
  auto mma = mlir::cast<NvidiaMmaEncodingAttr>(parent);
  return nvidiaDotToLinearLayout(shape, *this);       // NVIDIA MMA operand
}
```

| 参数 | 含义 |
|------|------|
| `opIdx` | A=0，B=1 |
| `parent` | 所属 dot 的输出 layout |
| `kWidth` | 沿 K 每 thread 连续元素数 |

**语义要点**：

- `parent` 是 **dot 输出** 的 layout（通常是 `nvidia_mma` / `amd_mfma` 等）；操作数 layout 由 parent **派生** 并沿 K 维 **broadcast**——多个 thread 故意持有同一 K 切片，供 `mma.sync` 消费。
- `opIdx=0`（A）：逻辑形状 `[*, M, K]`；`opIdx=1`（B）：`[*, K, N]`（行主序约定见 `getOrderForDotOperand`）。
- `kWidth` 越大，每个 thread 在 K 上连续持有越多元素。

**NVIDIA MMAv2 小例**（语义）：parent=`nvidia_mma`，A、`kWidth=8`，shape=`[16,64]`——M=16 与指令一致，K=64 由 lane/register bit 铺开，详见 `020_linear_layout.md` §5.4.1 例子 C。

Hopper 多数 dot 操作数走 **shared staging**（`nvmma_shared`），非 `dot_op`。

### 8.2 `nvidia_mma` — NVIDIA Tensor Core 输出

`nvidia_mma` 描述 **`tt.dot` 结果 C** 在寄存器里的排布（`mma.sync` / WGMMA 写回后的 fragment 布局）。

#### 版本代际

| `versionMajor` | 架构 | 说明 |
|----------------|------|------|
| 1 | Volta (SM70) | 第一代 Tensor Core |
| 2 | Turing/Ampere (SM75/SM80/SM86) | 第二代，含 MMA v2 |
| 2 + Hopper | SM90 | MMA v3 (WGMMA) |

#### 参数

| 参数 | 含义 |
|------|------|
| `versionMajor` / `versionMinor` | 代际编码 |
| `warpsPerCTA` | 每个 CTA 的 warp 数（扩展 tile 覆盖） |
| `instrShape` | 指令形状 **[M, N, K]**；2D 结果看 **M×N** |
| `CTALayout` | 多 CTA 切分 |

#### `instrShape` vs tensor shape（易混）

| | `instrShape` | tensor `shape` |
|--|-------------|----------------|
| 含义 | **一条 MMA 指令** 处理的 M×N×K 块 | 整个张量的逻辑大小 |
| 典型 Ampere v2 | `[16, 8, …]` → 单指令覆盖 **16×8** 的 C | 如 `[16,16]`、`[32,32]` |
| 大于 tile | — | **wrap**：同一指令 tile 沿维重复（§3.2） |
| 小于 tile | — | **broadcast**：多余 lane/warp bit 映到同一元素 |

**例**：`instrShape=[16,8]`，`warpsPerCTA=[1,1]`，`shape=[16,16]`。

- 指令 tile 覆盖 **16×8** 个 C 元素（register + lane 分工，固定 PTX 模式）。
- dim1：16 > 8 → 沿 N **wrap 一次**（第二份 8 列是 tile 的重复）。
- dim0：16 = 16 → 无 wrap。

tile 内 **谁拿哪个 C[i,j]** 不是 blocked 那种「thread id 递增」表，而是 Tensor Core 规定的 fragment 排布；用人脑手画整张表较繁琐，但 **wrap 规律与 §3 相同**。

#### 与 `blocked` 的对比

| | `#ttg.blocked` | `#ttg.nvidia_mma` |
|--|----------------|-------------------|
| 参数 | `sizePerThread`, `threadsPerWarp`, `warpsPerCTA`, `order` | `instrShape`, `warpsPerCTA`, version |
| tile 大小 | 三参数逐维乘 | 由 **指令** 固定（如 16×8） |
| 典型用途 | global load/store | `tt.dot` 输出 |
| bases 手算 | `020` §4 | `020` §5.4.1 |

**测试 golden**：`LinearLayoutConversionsTest::MMAv2_16x16`、`MMAv2_32x32`。

### 8.3 `amd_mfma` — AMD MFMA Output

MFMA 输出 C 的 layout；参数决定指令 tile 与每 warp 铺几块。

| 参数 | 含义 |
|------|------|
| `version` | 1–4（对应 gfx908/gfx90a/gfx942/gfx950） |
| `warpsPerCTA` | 每个 CTA 的 warp 数 |
| `instrShape` | 指令形状 [M,N,K] |
| `isTransposed` | 是否转置 |
| `tilesPerWarp` | 每 warp 覆盖的 tile 数 |
| `elementBitWidth` | 元素位宽 |
| `CTALayout` | 多 CTA 切分 |

**语义**：`instrShape=[32,32,8]` 表示 MFMA 指令的 M×N tile；`tilesPerWarp` 控制每 warp 覆盖几块；`isTransposed` 交换 M/N 排布。`warpsPerCTA={2,4}` 时沿 dim 扩展覆盖，大 shape 时与 §3 一样 wrap。

**bases 对照**：`020_linear_layout.md` §5.4.1 AMD 简述；golden 见 `LinearLayoutConversionsTest` 中 `mfmaNT` / `mfmaT` 系列。

### 8.4 `amd_wmma` — AMD WMMA Output

| 参数 | 含义 |
|------|------|
| `version` | 1–3（对应 gfx1100/gfx1200 等 RDNA 架构） |
| `warpsPerCTA` | 每个 CTA 的 warp 数 |
| `instrShape` | 指令形状 |
| `isTransposed` | 是否转置 |
| `tilesPerWarp` | 每 warp 覆盖的 tile 数 |
| `CTALayout` | 多 CTA 切分 |

WMMA 以 **Wave32** 模式运行（每 warp 32 lane，与 CDNA 的 64 lane MFMA 不同）。

**bases 手算**：`wmmaDotOperandToLinearLayout` / `AMDWmmaEncodingAttr::toLinearLayout`；见 020 附录 A。

### 8.5 本章小结

- **输出**（`nvidia_mma` / `amd_mfma` / `amd_wmma`）：tile 由 `instrShape` 固定；用 §3 判断 wrap/broadcast。
- **操作数**（`dot_op`）：在 parent 基础上沿 K broadcast；Hopper 新路径多用 shared。
- **实现**：统一 `toLinearLayout()` → `020` §5、§5.4.1。

---

## 9. TMEM Layout（Tensor Memory，Hopper+ SM90+）

Tensor Memory（TMEM）是 Hopper+ 架构引入的软件管理片上存储，独立于 shared memory。其 layout 定义在 `TritonNvidiaGPUAttrDefs.td`，由 `TensorMemoryEncodingAttr` 和 `TensorMemoryScalesEncodingAttr` 描述。

TMEM layout 是 **Plain `AttrDef`**（不继承 `LayoutEncodingTrait`），在 `TritonGPUDialect::toLinearLayout()` 中由独立 `dyn_cast` 分支处理（L1209–1215）：

```cpp
} else if (auto tensorMemoryEncoding =
               dyn_cast<TensorMemoryEncodingAttr>(layout)) {
  result = tensorMemoryToLinearLayout(shape, tensorMemoryEncoding);
} else if (auto tensorMemoryScalesEncoding =
               dyn_cast<TensorMemoryScalesEncodingAttr>(layout)) {
  result = tensorMemoryScalesToLinearLayout(shape, tensorMemoryScalesEncoding);
}
```

### 9.1 `tensor_memory_encoding`

| 参数 | 含义 |
|------|------|
| `blockM` / `blockN` | TMEM 中 M/N 维的 tile 大小 |
| `colStride` | 列步长 |
| `CTASplitM` / `CTASplitN` | 多 CTA 在 M/N 维切分 |
| `twoCTAs` | 是否双 CTA |

**用途**：描述 WGMMA/v5 指令写入 TMEM 时的数据排布，通常配合 `nvmma_shared` 做 shared→TMEM 的 staging。

### 9.2 `tensor_memory_scales_encoding`

| 参数 | 含义 |
|------|------|
| `CTASplitM` / `CTASplitN` | 多 CTA 切分 |

**用途**：MMAv5 量化 matmul 中 scale factor（FP8 block scaling 等）在 TMEM 中的排布。

### 与前后章的关系

- **前**：§8 MMA/Dot 中 `nvidia_mma` 的 v5 版本会涉及 TMEM；`nvmma_shared` 是 TMEM 的写入来源。
- **后**：附录 §10 提供参数速查。

---

## 10. 附录

### 10.1 使用场景速查

> 完整场景索引见 **§5.3**；下表为快速对照。

| 场景 | Layout | 平台 |
|------|--------|------|
| Global load/store | `blocked` | 通用 |
| 灵活 / 新 pass | `linear` | 通用 |
| Dot staging（通用） | `swizzled_shared` | 通用 |
| Dot staging（Hopper+ WGMMA） | `nvmma_shared` | NVIDIA |
| Dot staging（AMD rotating） | `amd_rotating_shared` | AMD |
| Linear shared 偏移 | `shared_linear` | 通用 |
| `tt.dot` A/B（旧 MMA） | `dot_op` | 通用* |
| `tt.dot` 结果（NVIDIA Tensor Core） | `nvidia_mma` | NVIDIA |
| `tt.dot` 结果（AMD MFMA） | `amd_mfma` | AMD CDNA |
| `tt.dot` 结果（AMD WMMA） | `amd_wmma` | AMD RDNA |
| `expand_dims` 逆 | `slice` | 通用 |
| TMEM 数据排布 | `tensor_memory_encoding` | NVIDIA Hopper+ |
| TMEM 量化 scales | `tensor_memory_scales_encoding` | NVIDIA Hopper+ |
| 多 block 切 tensor | 任意 + `CTALayout` | 通用 |

\* `dot_op` 的实际排布由 `parent` layout 决定。

### 10.2 如何 dump TTGIR

```bash
TRITON_PRINT_IR=1 python your_kernel.py
```

### 10.3 参数约定

| 约定 | 说明 |
|------|------|
| `order` | 最快变化维在前 |
| `opIdx` | A=0，B=1 |
| `kWidth` | 沿 K 每 thread 连续元素数 |
| `vec/perPhase/maxPhase` | shared swizzle |
| `CTALayout` | block → 逻辑维 |
| `versionMajor/Minor` | NVIDIA MMA 代际 |
| `instrShape` | MMA 指令形状 [M,N,K] |
| `isTransposed` | AMD MMA 是否转置 |

### 10.4 源码索引

| 文件 | 内容 |
|------|------|
| `TritonGPUAttrDefs.td` | layout 定义（通用） |
| `TritonNvidiaGPUAttrDefs.td` | NVIDIA 特有 layout（TMEM 等） |
| `CTAEncodingAttr.td` | CTA / cluster |
| `LinearLayout.h` | 线性 layout 数学 |
| `LinearLayoutConversions.h` | encoding → LinearLayout |
| `LinearLayoutConversions.cpp:1183` | `TritonGPUDialect::toLinearLayout()` 派发入口 |
| **`markdowns/020_linear_layout.md`** | bases 手推、`invertAndCompose`、codegen、14 种 encoding→LL |

### 10.5 与 `020_linear_layout.md` 交叉引用

| 概念 | 本文 | 020 |
|------|------|-----|
| tile / slot / wrap / broadcast | §3 | §1、§6 |
| `order` | §2.2、§4.7 | §4.2 |
| `blocked` 参数语义 | §6.1 | §4 手推 bases |
| LinearLayout 入门 | §4 全文 | 全文 |
| swizzle 语义 | §7.1 | §5.5（bases） |
| `nvidia_mma` / `dot_op` 语义 | §8 | §5.4.1（手算） |
| encoding→LL 转换表 | §5.2 | 附录 A |
| `convert_layout` / codegen | — | §7 |

### 10.6 概念关系总图

```
Python kernel
    │  block tensor? ──no──► 标量 IR，无 layout
    │  yes
    ▼
TTGIR tensor<..., #encoding>
    │
    ├─ Distributed（寄存器，7 种）
    │     ├─ Generic:     #blocked / #linear / #slice
    │     ├─ Dot operand: #dot_op（parent 决定派发）
    │     ├─ NVIDIA MMA:  #nvidia_mma
    │     └─ AMD MMA:     #amd_mfma / #amd_wmma
    │     └─ §3 映射 + §6 参数
    │
    ├─ Shared（共享内存，5 种）
    │     ├─ Generic:  #swizzled_shared / #padded_shared / #shared_linear
    │     ├─ NVIDIA:   #nvmma_shared
    │     └─ AMD:      #amd_rotating_shared
    │     └─ §7 偏移 + bank conflict
    │
    └─ TMEM（Tensor Memory，2 种，Hopper+）
          ├─ #tensor_memory_encoding
          └─ #tensor_memory_scales_encoding
          └─ §9 TMEM

§4 LinearLayout 入门 ──► 020 全文（bases、转换、codegen）
```

---

*Gluon 显式 layout API 见 `500_gluon_syntax_guide.md` 与 `_layouts.py`。编译器内部 bases 推导见 `020_linear_layout.md`。*
