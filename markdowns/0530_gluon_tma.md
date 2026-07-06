# 530 Gluon TMA（Tensor Memory Accelerator）阅读笔记

基于 [官方教程 tma](https://triton-lang.cn/main/getting-started/tutorials/gluon/tma.html) 与 `triton/python/tutorials/gluon/04-tma.py`，说明 **张量描述符、mbarrier、TMA 读写与 async/generic 代理间的 fence、流水线 elementwise add**。前置：`510_gluon_layout.md`（layout）、`520_gluon_async_copy.md`（cp.async 流水线剥循环）。

---

## 0. 阅读路线

```
§1 TMA 解决什么              → 寄存器压力、Hopper+、与 cp.async 对比
§2 张量描述符 TensorDescriptor → NVMMASharedLayout、block_shape、OOB 掩码
§3 最小 1D memcpy             → mbarrier + TMA load/store
§4 mbarrier 语义              → phase、expect 字节、init/invalidate
§5 双代理与 fence_async_shared → 何时要 fence、何时不必
§6 流水线 elementwise add TMA → 多缓冲 + perform_add 改法
§7 性能、smem 与块大小        → vs cp.async、228KB 预算
§8 附录                       → 限制、API 速查、源码
```

**术语约定**：

| 英文 | 含义 |
|------|------|
| **TMA** | Hopper+ 硬件：用描述符访问 gmem 中 N 维块，走 async proxy |
| **tensor descriptor** | 驻留 gmem 的元数据：shape/stride/基址/layout/block |
| **mbarrier** | smem 中 64 位同步对象，跟踪 TMA **读** 完成（phase + 字节计数） |
| **async proxy** | TMA gmem↔smem 路径 |
| **generic proxy** | 寄存器 `smem.load` / `smem.store` 路径 |
| **store_wait** | TMA **写** 完成跟踪（类似 cp.async 的 commit group，与之 **独立**） |
| **NVMMASharedLayout** | TMA 描述符要求的 smem 布局类型 |

**环境**：`cuda` + `capability[0] >= 9`（Hopper 及更新）。

```python
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
```

---

## 1. TMA 解决什么

**白话**：每条 `LDG/STG` 都要 64 位地址、掩码、结果寄存器；向量化只能缓解。TMA 用 **一块描述符** 代替逐元素寻址，事务走 **async proxy**，通常更快，但同步更复杂、寻址更僵。

| | `gl.load` / cp.async | TMA |
|---|----------------------|-----|
| 寻址 | 指针 + stride + mask（寄存器） | 描述符 + 块坐标 `[xoff, yoff]` |
| gmem↔smem | cp.async（Ampere+） | TMA 专用路径（Hopper+） |
| 读完成 | `commit_group` / `wait_group` | **mbarrier**（`expect` 字节 + `wait(phase)`） |
| 写完成 | cp 同组 | **`tma.store_wait`**（与 cp 组 **分开**） |
| smem layout | `SwizzledSharedLayout` 等 | **`NVMMASharedLayout`**（描述符绑定） |
| OOB | 用户 mask | 描述符 shape **自动** 处理 |

**与 520 关系**：流水线 **剥循环结构相同**；同步从 `wait_group` 换成 **mbarrier phase**，store 走 TMA 且需 **fence**。

**本章小结**：TMA = 低寄存器压力的块化 gmem↔smem + 独立同步域。  
**与后文**：§2 描述符如何构造。

---

## 2. 张量描述符 TensorDescriptor

描述符在 **gmem** 中，内核参数直接传 `in_desc` / `out_desc`，**不必**再传 stride 数组。

**Host 侧**：

```python
block_shape = [XBLOCK, YBLOCK]
layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)
a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
```

约束（教程强调）：

- 描述符 **必须** `NVMMASharedLayout`（可用 `get_default_for`，有时需手调）。
- 张量 **stride 16 字节对齐**。
- 每次 TMA 请求拷贝 **一个 block**；坐标为块原点偏移 `[xoff, yoff]`。
- `in_desc.block_type` / `layout` 与 smem 分配一致；读写描述符宜 `static_assert` 相等。

**内核侧**：`smem_layout = in_desc.layout`，`allocate_shared_memory(dtype, shape, layout=smem_layout)`。

**本章小结**：描述符 = gmem 张量的「块视图」+ smem 搬运 layout。  
**与后文**：§3 1D 最小读写闭环。

---

## 3. 最小 1D memcpy（TMA）

```python
@gluon.jit
def memcpy_1d_tma_kernel(in_desc, out_desc, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    smem = gl.allocate_shared_memory(in_desc.dtype, [XBLOCK], in_desc.layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(in_desc, [pid * XBLOCK], bar, smem)
    mbarrier.wait(bar, phase=0)   # 初态 phase 1 完成，等 phase 0
    mbarrier.invalidate(bar)

    tma.async_copy_shared_to_global(out_desc, [pid * XBLOCK], smem)
    tma.store_wait(pendings=0)
```

要点：

- **读**：不必 `smem.load` 再 `gl.store`；可从 smem **直接 TMA 写回** gmem。
- 此核几乎只用标量坐标 → `num_warps=1` 即可。
- 教程 API 别名：`tma.async_load` ≈ `async_copy_global_to_shared`。

**本章小结**：读用 mbarrier，写用 `store_wait`；无寄存器中转。  
**与后文**：§4 展开 mbarrier。

---

## 4. mbarrier 语义

| 步骤 | API | 含义 |
|------|-----|------|
| 分配 | `allocate_shared_memory(..., MBarrierLayout())` | smem 中 64 位对象 |
| 初始化 | `mbarrier.init(bar, count=1)` | 每完成一轮 TMA 读 **arrive 一次**；count 减到 0 则 **当前 phase 完成** 并切下一 phase |
| 预期字节 | `mbarrier.expect(bar, nbytes)` | 跟踪未完成字节；TMA 传 `bar` 时随事务 **原子递减**，到 0 则 arrive 一次 |
| 等待 | `mbarrier.wait(bar, phase=0/1)` | 按 **phase 奇偶** 查询完成 |
| 销毁 | `mbarrier.invalidate(bar)` | 用毕必须 invalidate |

**Phase 陷阱**：mbarrier 只可靠跟踪 **当前 + 上一** phase；若生产者 phase 跑太快，消费者 `wait` 会 **失步**。流水线里 `read_phase = read_index // num_buffers & 1` 即为此服务。

**读 vs 写同步**：

```
TMA read  ──► mbarrier
TMA write ──► store_wait（与 cp.async commit 组无关）
```

**本章小结**：mbarrier = 按字节完成的异步读屏障 + 二相状态机。  
**与后文**：§5 与 generic 代理的次序。

---

## 5. 双代理与 fence_async_shared

| 代理 | 访问方式 |
|------|----------|
| async | TMA `async_copy_global_to_shared` / `async_copy_shared_to_global` |
| generic | `smem.load()` / `smem.store()` |

**跨代理无顺序保证**，需 `fence_async_shared()` 建立可见性。

| 场景 | 是否需要 fence |
|------|----------------|
| `smem.load()` 后立刻 TMA 写入同 smem | **要**（load 未完成可能被打断） |
| `smem.store()` 后立刻 TMA 读出 smem | **要**（store 未完成可能被读） |
| `mbarrier.wait` 在 TMA **读** 之后 → `smem.load()` | **不要**（读已完成，async 写 smem 已结束） |
| `smem.store` → `mbarrier.arrive/wait` → TMA store | **仍要** fence（arrive/wait 不保证 generic store 对 async 可见） |

流水线 `perform_add` 典型序列：

```python
mbarrier.wait(bars.index(...), read_phase)
a_val = a_smem.index(...).load(layout)
# … add …
tma.store_wait(pendings=0)
c_smem.store(c_val)
fence_async_shared()
tma.async_copy_shared_to_global(c_desc, [xoff, yoff], c_smem)
```

**本章小结**：凡 generic 与 TMA 交错碰同一块 smem，默认加 fence；仅「TMA 读 + wait 后 load」可省略。  
**与后文**：§6 完整流水线核。

---

## 6. 流水线 elementwise add（TMA）

相对 `520` 的 `elementwise_add_pipelined`：

| 组件 | cp.async 版 | TMA 版 |
|------|-------------|--------|
| 参数 | 裸指针 + stride | `a_desc, b_desc, c_desc` |
| 发读 | `issue_loads` + `commit_group` | `issue_loads` + `mbarrier.expect` + `tma.async_copy_global_to_shared` |
| 等读 | `wait_group(num_buffers-1)` | `mbarrier.wait(bar, read_phase)` |
| 写回 | `gl.store` | `c_smem.store` + fence + `tma.async_copy_shared_to_global` |
| 额外 smem | 仅 A/B 多缓冲 | **+** `c_smem` 单缓冲 tile |
| 额外同步 | 无 | 每缓冲一个 `mbarrier` |

**issue_loads**（A/B 共用一个 bar）：

```python
bar = bars.index(copy_index % num_buffers)
mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
tma.async_copy_global_to_shared(a_desc, [xoff, yoff], bar, a_smem.index(...))
tma.async_copy_global_to_shared(b_desc, [xoff, yoff], bar, b_smem.index(...))
```

**主核三阶段**（与 520 相同 peel 逻辑）：

1. `static_range(num_buffers - 1)`：只预取  
2. `range(n_tiles - (num_buffers - 1))`：`issue_loads` + `perform_add`  
3. `static_range(num_buffers - 1)`：只 `perform_add`  
4. 收尾：`mbarrier.invalidate` 全部 bar；`tma.store_wait(0)`

`xoff = pid * XBLOCK`（标量），不再用 `gl.arange` 算 gmem 指针。

**本章小结**：流水线骨架沿用 520；同步与 store 路径换 TMA + mbarrier + fence。  
**与后文**：§7 benchmark 与 smem 上限。

---

## 7. 性能、smem 与块大小

32K×32K，`XBLOCK=32, YBLOCK=64, num_buffers=2`（教程）：

| 内核 | 带宽（约） |
|------|------------|
| `elementwise_add_pipelined`（cp.async） | 4.20 TB/s |
| `elementwise_add_tma` | 5.50 TB/s |

**为何 TMA 更快**：更少地址/掩码寄存器；编译器可在内层循环交错 smem load / add / smem store；TMA 路径带宽更好。

**放大块**（`64×128×fp32`，三缓冲）→ **5.90 TB/s**：

- 峰值寄存器仍可控（交错指令），**主要限制是 smem**。
- 每 SM **228 KB** shared：`128×128×f32` 双缓冲输入不够；`64×128` 三缓冲约 **224 KB** 勉强放下。

**本章小结**：TMA 在同等流水线下优于 cp.async；进一步提速靠 **更大 tile + 更深缓冲**，受 smem 硬顶。

---

## 8. 附录

### 8.1 TMA 硬限制

- 最内维坐标 **16 字节对齐**（如 fp16：`[8,4]` 非法，`[4,8]` 合法）。
- `fp4_padded` smem layout 时，最内维 **128 字节** 对齐。
- 全局张量 layout/stride 不满足时 **无法** 建描述符（灵活性代价）。

### 8.2 API 速查

| 调用 | 作用 |
|------|------|
| `TensorDescriptor.from_tensor(t, block_shape, layout)` | 构造描述符 |
| `tma.async_copy_global_to_shared(desc, [coords…], bar, smem)` | 异步 gmem→smem |
| `tma.async_copy_shared_to_global(desc, [coords…], smem)` | 异步 smem→gmem |
| `mbarrier.init / expect / wait / invalidate` | 读同步 |
| `tma.store_wait(pendings=0)` | 等 TMA store 完成 |
| `fence_async_shared()` | async ↔ generic smem 次序 |

### 8.3 与 520 对照

| 话题 | 520 cp.async | 530 TMA |
|------|--------------|---------|
| 硬件 | Ampere+ | Hopper+ |
| 读同步 | `wait_group(k)` | `mbarrier.wait(phase)` |
| 写同步 | 常 `gl.store` | `store_wait` |
| 掩码 | 用户 `mask` | 描述符 shape |
| 流水线 peel | 相同 | 相同 |

### 8.4 运行

```bash
cd triton/python/tutorials/gluon
python 04-tma.py
```

本地可同步到 `ops/gluon/03_tma.py`（当前可为空）。

### 8.5 源码索引

| 路径 | 内容 |
|------|------|
| `triton/python/tutorials/gluon/04-tma.py` | 教程全文 + benchmark |
| `triton/python/tutorials/gluon/03-async-copy.py` | 对照 `elementwise_add_pipelined` |
| `markdowns/520_gluon_async_copy.md` | cp.async 流水线 |
| `markdowns/510_gluon_layout.md` | 寄存器 layout、合并访问 |
| [tma 官方教程](https://triton-lang.cn/main/getting-started/tutorials/gluon/tma.html) | 中文文档 |

### 8.6 概念总图

```
Host: TensorDescriptor.from_tensor(..., NVMMASharedLayout)
                    │
                    ▼
Kernel args: a_desc, b_desc, c_desc
                    │
    ┌───────────────┴───────────────┐
    ▼                               ▼
tma gmem→smem (async proxy)     smem.load/add/store (generic)
    │ mbarrier.expect/wait          │ fence_async_shared
    └──────── smem multibuffer ─────┘
                    │
            tma smem→gmem → store_wait
```

### 8.7 教程四句话总结

1. **TMA** 用描述符块化访问 gmem，减轻 **寄存器压力**，走更快 async 路径。  
2. **读** 用 mbarrier（字节 + phase），**写** 用 `store_wait`，与 cp.async 提交组 **分离**。  
3. **async / generic** 代理访问 smem 需 **`fence_async_shared`**（TMA 读 wait 后 load 除外）。  
4. 流水线可复用 520 peel；性能上 TMA > cp.async，再调 **块大小 / num_buffers** 受 **228KB smem** 约束。

---

## 附：阅读后的理解（编者）

1. **描述符是寻址压缩** — 内核里只剩块坐标；代价是 stride 对齐、layout 固定、非规整张量可能用不了 TMA。  
2. **mbarrier phase 与多缓冲绑定** — `read_index // num_buffers & 1` 不是技巧，是防 phase 赛跑。  
3. **store 路径多一块 c_smem** — 加法结果先进 smem 再 TMA 写出，故 store 前必有 fence；与 520 直接 `gl.store` 不同。  
4. **寄存器腾出 → 更大 tile** — 这是 TMA 相对 cp.async 的第二个收益，第一个是纯 gmem 带宽。  
5. **下一教程** — Warp-Group MMA 等会假定 NVMMA/TMA 搬运已就绪。

---

*关联：`510_gluon_layout.md`、`520_gluon_async_copy.md`；教程源码：`04-tma.py`。*
