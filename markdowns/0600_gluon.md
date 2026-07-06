# Gluon

Gluon 在 Triton 3.6 最大的变动点是暴露了share memory 和layout 给外部编排使用，同样的也提升了变成难度。

那么暴露 share memory 给外部使用有什么好处呢？ 典型的算法 GQA 类型的 FA 算法用 triton 实现性能慢。

---

## 稀疏 GQA MLA 伪代码

`flash_mla_sparse_with_kv_cache_fa_kernel`（单段 kv cache、无 attn_sink）。DeepSeek v4 decode：**GQA**（`H_Q=8` query head、`num_kv_heads=1`）、**MLA**（K/V 同一张 `kv_cache`）、**稀疏 topk**（`indices` 指定 token，非连续 sequence）。

**张量**

| 名 | 形状 | 说明 |
|----|------|------|
| `q` | `[B, 1, H_Q, D_V]` | query |
| `kv_cache` | `[num_blocks, PAGE_SIZE, 1, D_V]` | paged KV，K/V 同址 |
| `indices` | `[B, 1, TOPK]` | 稀疏 token 全局下标 |
| `topk_length` | `[B]` | 有效 topk 长度 |

**Launch**：`grid = (⌈H_Q/BLOCK_H⌉, B)`，每个 program 负责 `(head_block, batch_token)`，内层 `for group_idx` 扫 `⌈topk_length/GROUP_SIZE⌉` 组。

```
# 常驻状态（跨 group 的 online softmax，等价 FA 的 m / l / acc）
acc[BLOCK_H, D_V] = 0
running_max[BLOCK_H] = -inf
running_denom[BLOCK_H] = 0

for group_idx in range(⌈context_len / GROUP_SIZE⌉):
    slot = group_idx * GROUP_SIZE + [0 .. GROUP_SIZE-1]
    idx  = indices[batch, 0, slot]              # 稀疏 gather 下标
    valid = (slot < context_len) & (idx >= 0)
    block = idx // PAGE_SIZE
    pos   = idx % PAGE_SIZE

    # ---- QK^T：K 维 D_V 按 BLOCK_D 分块累加 ----
    scores[BLOCK_H, GROUP_SIZE] = 0
    for d in range(0, D_V, BLOCK_D):
        Q = q[batch, head_block*BLOCK_H:+, d:d+BLOCK_D]     # [BLOCK_H, BLOCK_D]
        K = kv_cache[block, pos, 0, d:d+BLOCK_D]             # gather → [GROUP_SIZE, BLOCK_D]
        scores += Q @ K.T * sm_scale

    scores[~valid] = -inf
    group_max  = rowmax(scores)                              # [BLOCK_H]
    exp_scores = exp(scores - group_max) * valid             # 组内未归一化 exp
    exp_sum    = rowsum(exp_scores)

    # ---- PV：一次 dot，K 维 = GROUP_SIZE，N 维 = D_V ----
    V = kv_cache[block, pos, 0, :]                          # [GROUP_SIZE, D_V]，再次 gather
    partial = exp_scores @ V                                 # [BLOCK_H, D_V]

    # ---- 跨 group 合并（与 FA 块间 online softmax 同构）----
    new_max     = max(running_max, group_max)
    scale_old   = exp(running_max - new_max)
    scale_group = exp(group_max - new_max)
    acc           = acc * scale_old + partial * scale_group
    running_denom = running_denom * scale_old + exp_sum * scale_group
    running_max   = new_max

out[batch, head_block*BLOCK_H:+, :] = acc / running_denom
```

1. **内层粒度是 `GROUP_SIZE` 个稀疏 token**，不是连续 `BLOCK_N` 列；KV 经 `(block, pos)` **随机 gather**。
2. **PV 不做组内归一化**，直接用 `exp_scores`，靠 `scale_group` 在跨 group 合并时修正——数学等价 online softmax，无需 fd kernel 写 `middle`。
3. **GQA**：`BLOCK_H` 行 Q 共享同一组 `indices/K/V`，但代码里每行 Q 独立 load，K/V gather **没有在 smem 跨 head 复用**。

---

## GROUP_SIZE 为什么开不大

`GROUP_SIZE` 是一组里参与 softmax 的稀疏 token 数，也决定 PV 矩阵乘的宽度。每轮 group 要在寄存器里同时放下 scores、exp_scores、输出累加器，以及 **整列 `D_V` 的 V**——没有 smem 中转，全从 HBM gather 进寄存器。

### 1. 寄存器是硬顶

`GROUP_SIZE` 越大，`scores`/`exp_scores` 线性变大；PV 还要 hold `[GROUP_SIZE, D_V]` 的 V。`D_V=512` 时，仅 V 这一块在 `GROUP_SIZE=64` 就已占约 64KB 量级寄存器需求，提到 256 则翻四倍，CTA 极易 spill、occupancy 暴跌，**反而更慢**。

连续 FA 可把 K/V 放进 smem，`BLOCK_N` 能开大；本实现 K/V 只在寄存器里过一遍，**GROUP_SIZE 上限绑死在寄存器文件**，实际常用 64 左右，很难像 FA 那样推到 128+。

### 2. softmax 不允许把 group 再拆小

online softmax 要在 **同一组内所有 slot** 上取 rowmax 再 merge 到全局。若为了省寄存器把一组拆成两次算，两次的 max 不在同一归约里，数值会错。所以不能用「更小的 GROUP_SIZE、多跑几趟 group」来换寄存器——group 内 token 必须一次算完，**GROUP_SIZE 和寄存器压力绑在一起**。

QK 还要沿 `D_V` 分块累加，GROUP_SIZE 越大，每块 gather 的 K 越宽，**单 group 耗时也随 GROUP_SIZE 涨**。

---

## 这种实现性能为什么差

### 1. K/V 每 group 从 HBM gather 两次

QK 阶段 load `k_vals`，PV 阶段再 load `v_vals`（同址但仍是一次独立 gather）。有效 HBM 流量 ≈ `2 × GROUP_SIZE × D_V × num_groups × H_Q`（未计 Q 与 indices），**带宽 bound** 时几乎线性翻倍。

### 2. 稀疏 gather 无法合并访存

`indices` 映射的 `(block, pos)` 通常 **跨 page、跨 cache line**。不像连续 FA 的 `K[start_n:start_n+BLOCK_N]` 可 coalesce；每个 slot 一次 scattered load，**memory latency 隐藏差**，开大 GROUP_SIZE 只加宽 scatter 宽度、不改善 locality。

### 3. GQA 未复用 KV

8 个 Q head 共用 1 组 KV，但 program 按 `head_block` 划分，**同一 batch 的 8 head 各跑一遍完整 gather**（8 个 program 读同一 `indices`/KV）。理想 GQA：KV 进 smem 一次、8 组 Q 复用；当前实现 **H_Q 倍 KV 读放大**。

### 4. Q 在 QK 分块里重复读

沿 head 维分块算 QK 时，每一轮都要重新从 HBM 读同一段 Q。Q 对当前 group 不变，本应驻留 on-chip 复用；没有显式 smem 时，**同一份 Q 被重复 load**。

### 5. 无 load/MMA 流水线

算 group `g` 的 dot 时无法 cp.async/TMA 预取 group `g+1` 的 KV；group 间严格串行。FA v2/v3 靠 smem 双缓冲把 memory latency 藏进 MMA pipeline，本 kernel **compute 与 memory 串行**。

### 6. GROUP_SIZE 小 → 算术强度低

`TOPK=128, GROUP_SIZE=64` 仅 2 次 group；每次 group 做两次 dot（QK 分块 + PV 一次），相对 indices load、softmax、online rescale 的 **固定开销**，有效 **FLOPs/byte 偏低**，GPU 算力利用不足。

### 7. 小结：瓶颈链

```
稀疏 gather（无 coalesce）
    → K/V 读两次 / group
    → GQA 8 head 重复读
    → GROUP_SIZE 受寄存器限制开不大
    → group 数少也救不了算术强度
    → 无 smem 流水线隐藏 latency
    → 性能远低于 CUDA FA / Gluon 显式 smem 版
```

Gluon 侧可把 KV tile 放进 smem，QK 和 PV 共用；多 head 共享同一份 KV；预取下一组 KV，与当前组计算重叠。

上文就是 fa 无法使用triton 实现的瓶颈。还有一点需要说明的 fa 和 fd 都是比较常见的算法，在实践中，通常结合使用。

通常来说，这个取决于 硬件的 num_sm的个数，fa在小batch 情况下并行度不够，occupancy就低，而fd 可以弥补这一点。

当然也有context-len 太长，内部使用 fa 切出一个比较大的group_size ，然后叠加fd 使用。

也就基本只能使用 fd版本 gqa attention实现，prefill是做不了的，num_tokens 太大，容易爆内存。