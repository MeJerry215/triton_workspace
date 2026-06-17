# 520 Gluon 异步复制（Async Copy）阅读笔记

基于 [官方教程 async-copy](https://triton-lang.cn/main/getting-started/tutorials/gluon/async-copy.html) 与本地 `ops/gluon/02_async_copy.py`，说明 **cp.async 如何把全局内存异步搬到共享内存、如何用 commit/wait 组跟踪、如何用多缓冲做软件流水线**。前置：张量 layout 见 `510_gluon_layout.md`。

---

## 0. 阅读路线

```
§1 异步复制解决什么           → 与 gl.load 的差异、硬件前提
§2 最小 1D memcpy             → smem 描述符 + async_load + commit/wait
§3 基线：逐块 elementwise add → 无流水线、直接 gl.load
§4 cpasync 版 add（无流水线） → 双 smem tile、仍串行 wait
§5 smem layout 与 bank conflict → SwizzledSharedLayout、与寄存器 layout 耦合
§6 软件流水线与多缓冲         → issue_loads / perform_add、剥循环
§7 性能与寄存器压力           → benchmark、num_buffers、下一教程 TMA
§8 附录                       → API 速查、源码、概念总图
```

**术语约定**：

| 英文 | 含义 |
|------|------|
| **cp.async** | NVIDIA Ampere+ 全局→共享异步拷贝指令族 |
| **async copy** | 发出后不阻塞计算，靠 group 同步完成 |
| **commit_group** | 把此前已发出的 async 拷贝归入一组 |
| **wait_group(n)** | 等待直到「未完成组数 ≤ n」 |
| **software pipelining** | 循环内重叠「预取下一 tile」与「算当前 tile」 |
| **num_buffers / 流水线深度** | smem 多缓冲槽位数，决定可重叠的 in-flight 拷贝数 |
| **bank conflict** | 多 thread 同时打 smem 同一 bank 导致串行化 |

**环境**：`cuda` + `capability[0] >= 8`（Ampere 及更新）。导入：

```python
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
```

---

## 1. 异步复制解决什么

**白话**：`gl.load` 把数据直接进寄存器且通常阻塞；`cp.async` 在 **全局内存 (gmem) ↔ 共享内存 (smem)** 之间搬运，可与后续指令重叠，即「流水线化」内存。

| | `gl.load` / `gl.store` | `cp.async` + smem |
|---|------------------------|-------------------|
| 目标 | 寄存器文件 | 共享内存描述符 |
| 同步 | 隐式（load 完成才能用） | 显式 `commit_group` / `wait_group` |
| 重叠 | 需手写流水线 + 多缓冲 | 硬件异步 + 软件剥循环 |
| 布局 | 寄存器 `BlockedLayout` 等 | smem 另需 `SwizzledSharedLayout` 等 |

本教程聚焦 **NVIDIA**；其他厂商指令不同。

**本章小结**：异步拷贝 = gmem→smem 的非阻塞搬运 + 组同步。  
**与后文**：§2 最小内核看清 API 顺序。

---

## 2. 最小 1D memcpy（cp.async）

共享内存在 Gluon 里是 **带 layout 的描述符**，用法类似寄存器 tile：`allocate_shared_memory` → `async_load` → `wait` → `smem.load(layout)` → `gl.store`。

```python
@gluon.jit
def memcpy_1d_cpasync_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel

    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0])
    smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)

    cp.async_load(smem, in_ptr + offsets, mask=mask)  # 本地源码或为 async_copy_global_to_shared
    cp.commit_group()
    cp.wait_group(0)

    value = smem.load(layout)
    gl.store(out_ptr + offsets, value, mask=mask)
```

**API 顺序（固定模式）**：

```
async_load(…)
    → commit_group()      # 封存「这一批」拷贝
    → wait_group(k)       # k=0 表示等到全部组完成
    → smem.load(reg_layout)
```

**本章小结**：一条 gmem 向量先进 smem，再按寄存器 layout 读出写回。  
**与后文**：§3 二维 add 把「一次一块」放进循环。

---

## 3. 基线：逐块 elementwise add

每个 program 负责矩阵 **一行中的 XBLOCK 列**，内层 `for yoff in range(0, ynumel, YBLOCK)` 每次处理一个 `[XBLOCK, YBLOCK]` tile：

```python
layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
# …
for yoff in range(0, ynumel, YBLOCK):
    yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    a_val = gl.load(a_ptrs + ystride_a * yoffs[None, :], mask=mask)
    b_val = gl.load(...)
    gl.store(c_ptrs + ..., a_val + b_val, mask=mask)
```

无 smem、无重叠：load → add → store 串行。

**本章小结**：建立正确性与块遍历结构，作为性能对照。  
**与后文**：§4 把 load 换成 cp.async。

---

## 4. cpasync 版 add（无流水线）

在循环内对 A、B 各分配一块 smem，仍 **每轮 wait 完再算**：

```python
a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

for yoff in range(0, ynumel, YBLOCK):
    cp.async_load(a_smem, a_ptrs + ystride_a * yoffs[None, :], mask=mask)
    cp.async_load(b_smem, b_ptrs + ystride_b * yoffs[None, :], mask=mask)
    cp.commit_group()
    cp.wait_group(0)
    a_val = a_smem.load(layout)
    b_val = b_smem.load(layout)
    gl.store(...)
```

教程 32K×32K 实测（memory-bound，测 3× 元素流量）：

| 内核 | 带宽（约） |
|------|------------|
| `elementwise_add` | 1.48 TB/s |
| `elementwise_add_cpasync` | 3.97 TB/s |

**为何已更快**：即使未重叠循环，gmem→smem 路径 + 合适 smem layout 仍优于直接 `gl.load` 到寄存器（合并与 smem 带宽特性）。但 **wait 在 add 之前**，拷贝与计算仍未重叠。

**本章小结**：cpasync 单缓冲即可大幅提速；要重叠须 §6。  
**与后文**：§5 解释 smem_layout 参数。

---

## 5. smem layout 与 bank conflict

**共享内存**：连续 32 位元素映射到不同 **bank**（至多 32）；新 GPU 上 bank 常 **双端口**，超配则访问串行化。

教程示例（2D tile）：

```python
smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
```

- `order`：smem 地址维度的铺排顺序（与 `510` 中 `BlockedLayout.order` 类比，但作用于 **shared**）。
- **swizzle**：打乱地址映射，减轻 bank conflict。
- 本教程 `vec=1, per_phase=1, max_phase=1` 接近 **非混洗**；仍选 `order=[1,0]` 是为与寄存器 `BlockedLayout([1,1],[1,32],[1,4],[1,0])` 配合。

**关键耦合**：寄存器 layout 若让每个 warp 的 32 thread 各持 **连续 32 位**，则 `smem.load(同一 layout)` 在 **fp32** 下可无 conflict；**fp16/fp8** 时更依赖 swizzle 与 `vec`。

**本章小结**：smem layout 与 reg layout **成对选择**，目标是最小 bank conflict。  
**与后文**：§6 在 smem 上再加流水线维 `num_buffers`。

---

## 6. 软件流水线与多缓冲

### 6.1 思想

若「一次 gmem 拷贝延迟 ≈ 3× 一次加法」，应维持约 **3 组** in-flight 拷贝（流水线深度）。实现：**smem 形状加一维** `[num_buffers, XBLOCK, YBLOCK]`，用 `a_smem.index(i % num_buffers)` 轮转。

```
时间 →
拷贝:  [buf0][buf1][buf2][buf0]…
计算:       [buf0][buf1][buf2]…
         └─重叠区─┘
```

### 6.2 辅助函数

**发拷贝**（`copy_idx` 逻辑块号，非物理缓冲下标）：

```python
@gluon.jit
def issue_loads(copy_idx, a_smem, b_smem, ...):
    yoffs = copy_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    cp.async_load(a_smem.index(copy_idx % num_buffers), a_ptrs + ..., mask)
    cp.async_load(b_smem.index(copy_idx % num_buffers), b_ptrs + ..., mask)
    cp.commit_group()
    return copy_idx + 1
```

**计算 + 写回**（`read_idx` 对应要消费的缓冲）：

```python
@gluon.jit
def perform_add(read_idx, a_smem, b_smem, ...):
    a_val = a_smem.index(read_idx % num_buffers).load(layout)
    b_val = b_smem.index(read_idx % num_buffers).load(layout)
    gl.store(c_ptrs + ystride_c * yoffs[None, :], a_val + b_val, mask)
    return read_idx + 1
```

### 6.3 主循环三阶段（剥循环）

设 `n_tiles = gl.cdiv(ynumel, YBLOCK)`。

| 阶段 | 循环 | 作用 |
|------|------|------|
| **Prologue** | `static_range(num_buffers - 1)` | 只 `issue_loads`，填满流水线 |
| **Steady** | `range(n_tiles - (num_buffers - 1))` | 每轮：`issue_loads` → `wait_group(num_buffers-1)` → `perform_add` |
| **Epilogue** | `static_range(num_buffers - 1)` | `wait_group(num_buffers - 2 - i)` → `perform_add`，排空 |

**`wait_group(num_buffers - 1)` 含义**：允许队列里最多 `num_buffers-1` 个未完成组；刚发出的那组算第 `num_buffers` 个，最早发出的那组此时 **刚好完成**，可读对应 buffer。

**尾块**：`yoffs < ynumel` 的 mask 在 `issue_loads` 里处理 tile 数不足 `num_buffers-1` 的情况。

### 6.4 深度与实测

| `num_buffers` | 32K×32K 带宽（教程） |
|---------------|----------------------|
| 2（双缓冲） | ~4.20 TB/s |
| 3（三缓冲） | ~4.20 TB/s |

**解读**：相对单缓冲 cpasync 仅 **小幅** 提升；再加缓冲无益 → 内核已 **内存带宽饱和**，非计算 bound。

**本章小结**：流水线 = 多 smem 槽 + 剥循环 + `wait_group` 与 in-flight 深度对齐。  
**与后文**：§7 谈寄存器与 TMA。

---

## 7. 性能与寄存器压力

**寄存器**：每元素需保存结果、64 位地址、mask；双输入时压力更大（每 thread 上限 256）。教程用较小 tile **`XBLOCK=32, YBLOCK=64`**  partly 为此。

**瓶颈归纳**：

1. 无 cp.async → gmem 路径差（~1.5 TB/s）。
2. 有 cp.async 无流水线 → 已 ~4 TB/s，拷贝与算未重叠。
3. 流水线 → 略升，随即触顶 **DRAM 带宽**。
4. **寄存器 / 块大小** 限制更大 tile，进而限制吞吐上限。

**下一教程（TMA）**：张量描述符减轻寻址与寄存器负担，以灵活性换更少地址算术。

**本章小结**：async copy 是内存 bound 内核的杠杆；深度调优需同时看 smem layout、流水线深度、寄存器与块形状。

---

## 8. 附录

### 8.1 API 速查

| 调用 | 作用 |
|------|------|
| `gl.allocate_shared_memory(dtype, shape, layout=smem_layout)` | 分配 smem 描述符 |
| `cp.async_load(smem, gmem_ptr, mask=…)` | 异步 gmem→smem（别名见下） |
| `cp.commit_group()` | 当前批次拷贝入组 |
| `cp.wait_group(n)` | 等待 pending 组数 ≤ n |
| `smem.load(reg_layout)` | smem→寄存器 tile |
| `smem.index(i)` | 多缓冲选槽（`i % num_buffers`） |

本地 `02_async_copy.py` 中或为 `cp.async_copy_global_to_shared`，与教程 `async_load` 同族，以安装版本 API 为准。

### 8.2 流水线 checklist

- [ ] `num_buffers ≥ 2` 才有重叠意义
- [ ] prologue 次数 = `num_buffers - 1`
- [ ] steady 中 `wait_group(num_buffers - 1)` 与 `read_idx`/`copy_idx` 同步递增
- [ ] epilogue 中 `wait_group(num_buffers - 2 - i)` 递减排空
- [ ] A/B 两次 `async_load` 后 **一次** `commit_group`（同组）

### 8.3 本地运行

```bash
cd /root/triton_workspace/ops/gluon
python 02_async_copy.py   # 需 Ampere+ GPU
```

### 8.4 源码索引

| 路径 | 内容 |
|------|------|
| `ops/gluon/02_async_copy.py` | 教程复现 + benchmark |
| `markdowns/510_gluon_layout.md` | `BlockedLayout`、`SliceLayout`、smem 铺垫 |
| `markdowns/010_layout.md` | TTGIR layout 底层 |
| [async-copy 官方教程](https://triton-lang.cn/main/getting-started/tutorials/gluon/async-copy.html) | 原文与数字 |

### 8.5 概念总图

```
gmem ──cp.async_load──► smem[num_buffers, XBLOCK, YBLOCK]
                              │
                    commit_group / wait_group
                              │
                    smem.index(k).load(reg_layout)
                              │
                         compute (add)
                              │
                         gl.store ──► gmem

循环 peel:
  prologue:  issue × (D-1)
  steady:    issue → wait(D-1) → compute
  epilogue:  wait(D-2-i) → compute
```

### 8.6 教程四句话总结

1. **cp.async** 在 gmem 与 smem 间异步搬运，用 **commit/wait 组** 同步。
2. 即使 **不流水线**，cpasync + smem layout 也可相对 `gl.load` **数倍** 提速（memory-bound）。
3. **软件流水线** = smem 多缓冲 + 剥循环，重叠预取与计算；深度应匹配 **拷贝/计算延迟比**。
4. **smem layout** 与 **寄存器 layout** 共同决定 bank conflict；饱和带宽后加缓冲无效。

---

## 附：阅读后的理解（编者）

1. **同步点是 wait_group，不是 async_load** — 可在 wait 前插入无关计算（流水线版即在 wait 前发下一 tile 的 load）。
2. **copy_idx 与 read_idx 分离** — 逻辑 tile 序号单调增，物理缓冲 ` % num_buffers`；避免读写同一槽。
3. **与 510 的分工** — 510 管寄存器 tile 如何铺；520 管 **数据先进 smem** 再 `load` 到同一 layout，两条 layout 都要调。
4. **单缓冲 cpasync 已是大头** — 写内核时先换 async 路径，再考虑双缓冲；用 benchmark 验证是否已 bandwidth-bound。
5. **寄存器是隐形天花板** — 教程明确点出；大 tile + 双缓冲 + 掩码会挤占 occupancy，与 TMA 教程衔接。

---

*关联：`510_gluon_layout.md`；本地实践：`ops/gluon/02_async_copy.py`；下一教程：Gluon TMA。*
