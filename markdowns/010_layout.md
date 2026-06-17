# 010 Triton GPU Layout 排布说明

基于 `TritonGPUAttrDefs.td`、`CTAEncodingAttr.td`，说明 TTGIR 中 **NVIDIA** 相关 layout 的含义与用法。

---

## 0. 阅读路线

建议按章顺序读；每章末尾有 **与前后章的关系**。

```
§1 IR 里是什么          → 先能在 TTGIR 里认出 layout
§2 硬件层次              → Thread / Warp / CTA / CGA、num_warps / num_ctas
§3 映射语义              → 逻辑坐标 → thread；tile / wrap / broadcast
§4 LinearLayout         → 高级 encoding 的公共数学（可选深读）
§5 Distributed          → blocked（含 1D vector-add）、linear、slice
§6 Shared               → swizzle / padded / nvmma_shared
§7 Dot / MMA            → dot_op、nvidia_mma
§8 附录                 → 速查、dump IR、参数表、源码
```

**术语约定**（与 IR 一致，不强行中译）：

| 英文 | 含义 |
|------|------|
| **tile** | layout 覆盖的基础区域；tensor 更大时沿维 **重复** 同一 tile |
| **rep** | repetition，tile 在张量上再铺一份 |
| **slot** | tile 内索引位置，对应 thread 分工 |
| **wrap / broadcast** | tensor 维 长于 / 短于 tile 时的两种分布 |

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

下文 §3–§5 以 Distributed 为主；§6 讲 Shared。

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

本章建立 §5 Distributed 的公共模型；**先读本章再读 blocked 参数**。

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

**本章小结**：layout = tile 表 $L$ + 与 $T.\mathrm{shape}$ 比较得到 wrap/broadcast；§5 `blocked` 是 $L$ 的具体参数化。

---

## 4. LinearLayout：统一数学语言

多种 encoding（`linear`、`CTAEncoding`、`padded_shared`）共用 **LinearLayout**（`LinearLayout.h`）。

### 4.1 目标：把「谁拿到哪个元素」写成可组合的函数

把一次访问里“硬件层次的坐标”（寄存器槽位 / thread / warp / CTA …）记为 \((t,w)\)，把“逻辑张量坐标”（多维 index）记为 \(L(t,w)\)。

**映射**：硬件坐标 → 逻辑张量坐标，且满足 XOR 线性：

$$
L(t_1 \oplus t_2,\, w_1 \oplus w_2) = L(t_1,w_1) \oplus L(t_2,w_2)
$$

这里 \(\oplus\) 是按位 XOR；含义是：硬件坐标如果按 bit 翻转，那么输出坐标也按相同方式做“线性叠加”（在 \(\mathbb{F}_2\) 上的线性变换）。这类表示特别适合 GPU：thread/lane/warp 等本来就是用 bit 拼出来的。

### 4.2 basis：只写“每一位”对输出坐标的贡献

因为 XOR 线性，只需定义 **basis**（输入为 \(1,2,4,\ldots\) 这些单 bit）时的输出，其余任意输入都是这些 basis 的 XOR 组合。

直观上：每个输入维（`register/lane/warp/block`）都有若干 bit；第 \(k\) 个 bit 被置 1，就把该维的第 \(k\) 个 basis 向量 XOR 到输出坐标里。

```mlir
#ttg.linear<{register = [[0, 1], [8, 0], ...],
             lane = [[0, 2], [1, 0], ...],
             warp = [[16, 0], [32, 0]],
             block = []}>
```

常见输入维：`register`, `lane`（thread）, `warp`, `block`（CTA）。

### 4.3 怎么“算”一个坐标（概念流程）

- **输入**：给定某次访问的硬件坐标 \((r, \ell, w, b)\)（分别对应 `register/lane/warp/block` 的整数）。
- **拆 bit**：把每个整数拆成 bit（例如 \(\ell=\sum_k \ell_k 2^k\)）。
- **XOR 叠加**：输出坐标从 0 开始；对所有 \(\ell_k=1\) 的位，把 `lane[k]` 这个 basis 向量 XOR 进去；`register/warp/block` 同理。
- **结果**：得到一个多维逻辑坐标（比如 \([i,j]\)）。

这个表示的好处是：很多布局操作（切片/转置/拼接某些维度的 bit）都能转成对 basis 的局部变换，而不需要枚举整张 tile 表。

### 4.4 与 §3（tile/slot）以及 `blocked` 的关系

- **与 §3 的 tile/slot**：§3 用“tile 表 \(L\)”描述一个有限小块里的映射；`LinearLayout` 用“basis”描述同一个映射，但可直接对 bit 组合，适合在编译器里做变换与合成。
- **与 `blocked`**：`blocked` 的 `sizePerThread/threadsPerWarp/warpsPerCTA/order` 本质上是在规定哪些 bit 属于 `register/lane/warp`，以及这些 bit 如何投影到输出坐标；编译内部常把 `blocked` 等 **转为 LinearLayout** 再 codegen（`LinearLayoutConversions.h`）。

---

## 5. Distributed Layout 详述

前提：§2 层次 + §3 映射。以下回答「IR 里 `#ttg.xxx` 各字段是什么」。

### 5.1 BlockedEncodingAttr（`#ttg.blocked`）— 最常用

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

这是 §4.1 的 **1D 特例**（不是另一种 layout）。

---

### 5.2 LinearEncodingAttr（`#ttg.linear`）

直接用 §4 的 `LinearLayout` 作为 encoding；新 pass 方向，逐步减少专用 encoding 分支。

---

### 5.3 SliceEncodingAttr（`#ttg.slice`）

从 `parent` layout **squeeze** 一维；用于 `expand_dims` 的逆变换（优化 pass）。参数：`dim`、`parent`。

---

**本章小结**：日常见最多的是 `blocked`；读 TTGIR 先对 `sizePerThread × threadsPerWarp × warpsPerCTA` 算 tile，再用 §3 判断 wrap/broadcast。

---

## 6. Shared Memory Layout

前提：数据在 **shared**，映射目标是 **偏移**，不是 thread id。用于 `ttg.local_alloc`、dot 操作数 staging；主要解决 **bank conflict**。

| mnemonic | 用途 |
|----------|------|
| `swizzled_shared` | 通用 XOR swizzle；Ampere/Hopper dot 常用 |
| `padded_shared` | padding + 线性变换 |
| `shared_linear` | LinearLayout 描述偏移 |
| `nvmma_shared` | MMAv3/v5 warpgroup matrix 专用格式 |

### 6.1 `swizzled_shared`

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

### 6.2 `padded_shared`

`intervals` / `paddings`：每隔若干元素插 padding（均为 2 的幂）。可带 `linearComponent` 做坐标重排。

### 6.3 `nvmma_shared`

Hopper+ WGMMA 输入；`swizzlingByteWidth` ∈ {0,32,64,128} 由连续维字节数推导。

**与 §5 关系**：global load 常用 `blocked` → 写入 `swizzled_shared` / `nvmma_shared` → MMA 读 shared。

---

## 7. NVIDIA Dot / MMA Layout

| mnemonic | 角色 |
|----------|------|
| `dot_op` | MMA v1/v2 的 A/B 操作数（`opIdx` 0/1，`kWidth`，`parent`=输出 layout） |
| `nvidia_mma` | Tensor Core 输出 C（`versionMajor` 1=Volta，2=Turing+） |

Hopper 多数 dot 操作数走 **shared**（`nvmma_shared`），非 `dot_op`。

---

## 8. 附录

### 8.1 使用场景速查

| 场景 | Layout |
|------|--------|
| Global load/store | `blocked` |
| 灵活 / 新 pass | `linear` |
| Dot staging | `swizzled_shared` / `nvmma_shared` |
| `tt.dot` A/B（旧 MMA） | `dot_op` |
| `tt.dot` 结果 | `nvidia_mma` |
| `expand_dims` 逆 | `slice` |
| 多 block 切 tensor | 任意 + `CTALayout` |

### 8.2 如何 dump TTGIR

```bash
TRITON_PRINT_IR=1 python your_kernel.py
```

### 8.3 参数约定

| 约定 | 说明 |
|------|------|
| `order` | 最快变化维在前 |
| `opIdx` | A=0，B=1 |
| `kWidth` | 沿 K 每 thread 连续元素数 |
| `vec/perPhase/maxPhase` | shared swizzle |
| `CTALayout` | block → 逻辑维 |
| `versionMajor/Minor` | NVIDIA MMA 代际 |

### 8.4 源码索引

| 文件 | 内容 |
|------|------|
| `TritonGPUAttrDefs.td` | layout 定义 |
| `CTAEncodingAttr.td` | CTA / cluster |
| `LinearLayout.h` | 线性 layout 数学 |
| `LinearLayoutConversions.h` | encoding → LinearLayout |

### 8.5 概念关系总图

```
Python kernel
    │  block tensor? ──no──► 标量 IR，无 layout
    │  yes
    ▼
TTGIR tensor<..., #encoding>
    │
    ├─ Distributed (#blocked / #linear / #slice / #dot_op / #nvidia_mma)
    │     └─ §2 层次 + §3 映射 + §5 参数
    │
    └─ Shared (#swizzled_shared / #padded_shared / #nvmma_shared)
          └─ §6 偏移 + bank conflict

§4 LinearLayout ──► 内部表示与统一转换
```

---

*Gluon 显式 layout API 见 `500_gluon_syntax_guide.md` 与 `_layouts.py`。*
