# Flash Attention + Online Softmax 完整公式推导

本文以稀疏 GQA MLA decode 场景为例，给出 Flash Attention（FA）单 kernel 路径中 **online softmax** 的完整数学推导，并映射到一组具体配置。

---

## 1. 问题配置

| 符号 | 含义 | 本例取值 |
|------|------|----------|
| \(B\) | batch size | 4 |
| \(H\) | query head 数 | 64 |
| \(D\) | head dim | 512 |
| \(L\) | 有效 context 长度（topk） | 2048 |
| \(G\) | group size（每轮 softmax 参与的 token 数） | 256 |
| \(P\) | KV cache page size | 64 |
| \(T\) | group 总数 | \(\lceil L/G \rceil = 8\) |

**张量形状**

| 张量 | 形状 | 说明 |
|------|------|------|
| `q` | `[B, 1, H, D]` = `[4, 1, 64, 512]` | decode 单 token query |
| `kv_cache` | `[num_blocks, P, 1, D]` = `[·, 64, 1, 512]` | paged KV，K/V 同址（MLA） |
| `indices` | `[B, 1, TOPK]` | 稀疏 token 全局下标 |
| `topk_length` | `[B]` | 每个 batch 的有效长度 = 2048 |

**缩放因子**

\[
\text{sm\_scale} = \frac{1}{\sqrt{D}} = \frac{1}{\sqrt{512}} \approx 0.04419
\]

**GQA 说明**：本例可视为 64 个 Q head 共享同一组稀疏 `indices` 与 KV（`num_kv_heads = 1`）。每个 head 有独立的 \(q_h\)，但 gather 的 K/V 相同。

---

## 2. 标准 Scaled Dot-Product Attention

对固定的 batch 下标 \(b \in \{0,\ldots,3\}\) 和 head 下标 \(h \in \{0,\ldots,63\}\)，设：

- \(q \in \mathbb{R}^{D}\)：query 向量
- \(\{k_j, v_j\}_{j=0}^{L-1}\)：由 `indices` gather 得到的 key/value，\(k_j, v_j \in \mathbb{R}^{D}\)

**Score 矩阵**（理论上形状 \([L]\)，此处不物化）：

\[
s_j = \text{sm\_scale} \cdot q^\top k_j, \quad j = 0, 1, \ldots, 2047
\]

**Attention 输出**：

\[
O = \sum_{j=0}^{L-1} \underbrace{\frac{e^{s_j}}{\sum_{k=0}^{L-1} e^{s_k}}}_{p_j} \, v_j \in \mathbb{R}^{D}
\]

向量形式：

\[
O = \mathrm{softmax}(s)^\top V, \quad s \in \mathbb{R}^{L},\; V \in \mathbb{R}^{L \times D}
\]

**若物化完整 score 矩阵**：每个 \((b, h)\) 需 \(L = 2048\) 个 float，共 \(B \times H \times L = 4 \times 64 \times 2048 = 524{,}288\) 个元素，fp32 约 **2 MB**。FA 的目标就是 **避免存储** 这 \(L\) 个 score，改为分 group 流式计算。

---

## 3. Softmax 数值稳定化

直接算 \(e^{s_j}\) 可能溢出。设全局最大值：

\[
M = \max_{0 \le j < L} s_j
\]

则 softmax 等价于：

\[
p_j = \frac{e^{s_j - M}}{\sum_{k=0}^{L-1} e^{s_k - M}}
\]

输出可写为：

\[
O = \frac{\displaystyle\sum_{j=0}^{L-1} e^{s_j - M}\, v_j}{\displaystyle\sum_{j=0}^{L-1} e^{s_j - M}}
= \frac{N}{D_{\text{sum}}}
\]

其中：

- **分子** \(N = \sum_j e^{s_j - M}\, v_j \in \mathbb{R}^{D}\)
- **分母** \(D_{\text{sum}} = \sum_j e^{s_j - M} \in \mathbb{R}\)

关键：\(M\) 在分母、分子中同时出现，任意统一的"基准 max"都给出相同结果。Online softmax 就是利用这一点，在 **不知道全局 \(M\)** 的情况下，逐块更新 \((m, \ell, \mathrm{acc})\)。

---

## 4. 分块（Tiling）：按 group 切分

将 \(L = 2048\) 个 token 按 `group_size = 256` 切成 \(T = 8\) 组：

| group \(t\) | slot 范围 | token 数 |
|-------------|-----------|----------|
| 0 | \(j \in [0, 255]\) | 256 |
| 1 | \(j \in [256, 511]\) | 256 |
| 2 | \(j \in [512, 767]\) | 256 |
| 3 | \(j \in [768, 1023]\) | 256 |
| 4 | \(j \in [1024, 1279]\) | 256 |
| 5 | \(j \in [1280, 1535]\) | 256 |
| 6 | \(j \in [1536, 1791]\) | 256 |
| 7 | \(j \in [1792, 2047]\) | 256 |

记第 \(t\) 组的 key/value 为 \(\{k_{t,j}, v_{t,j}\}_{j=0}^{G-1}\)（\(G=256\)）。

---

## 5. 单组内的量（组内不做归一化）

对第 \(t\) 组，先算组内 score：

\[
s_{t,j} = \text{sm\_scale} \cdot q^\top k_{t,j}, \quad j = 0, \ldots, 255
\]

**组内最大值**：

\[
m_t = \max_{j} s_{t,j}
\]

**组内未归一化 exp**（注意：不是概率，组和不必为 1）：

\[
\tilde{p}_{t,j} = e^{s_{t,j} - m_t}
\]

**组内 exp 之和**：

\[
\ell_t = \sum_{j=0}^{255} \tilde{p}_{t,j}
\]

**组内未归一化 PV**（形状 \([D]\)）：

\[
\tilde{O}_t = \sum_{j=0}^{255} \tilde{p}_{t,j}\, v_{t,j} = \tilde{p}_t^\top V_t
\]

其中 \(\tilde{p}_t \in \mathbb{R}^{256}\)，\(V_t \in \mathbb{R}^{256 \times 512}\)。

> **为什么不组内 softmax？** 因为全局 max 可能出现在后面的 group。组内归一化后再合并会丢失跨 group 的基准，必须保留未归一化的 \(\tilde{p}\)，在跨 group 合并时统一 rescale。

---

## 6. Online Softmax 跨组递推（核心推导）

维护三组状态（每个 head 独立一套）：

\[
m^{(0)} = -\infty, \quad \ell^{(0)} = 0, \quad \mathrm{acc}^{(0)} = \mathbf{0} \in \mathbb{R}^{D}
\]

处理第 \(t\) 组时（\(t = 0, \ldots, 7\)）：

### 6.1 更新全局 max

\[
m' = \max\bigl(m^{(t-1)},\, m_t\bigr)
\]

### 6.2 重缩放因子

把旧状态从基准 \(m^{(t-1)}\) 换算到新基准 \(m'\)：

\[
\alpha = e^{\,m^{(t-1)} - m'}
\]

把第 \(t\) 组从基准 \(m_t\) 换算到 \(m'\)：

\[
\beta = e^{\,m_t - m'}
\]

### 6.3 更新分母与分子

\[
\ell^{(t)} = \ell^{(t-1)} \cdot \alpha + \ell_t \cdot \beta
\]

\[
\mathrm{acc}^{(t)} = \mathrm{acc}^{(t-1)} \cdot \alpha + \tilde{O}_t \cdot \beta
\]

\[
m^{(t)} = m'
\]

### 6.4 最终输出

\[
O = \frac{\mathrm{acc}^{(8)}}{\ell^{(8)}}
\]

### 6.5 正确性证明

**命题**：上述递推满足，在处理完前 \(t\) 组后：

\[
\ell^{(t)} = \sum_{g=0}^{t-1} \sum_{j \in G_g} e^{s_{g,j} - m^{(t)}}
\]

\[
\mathrm{acc}^{(t)} = \sum_{g=0}^{t-1} \sum_{j \in G_g} e^{s_{g,j} - m^{(t)}}\, v_{g,j}
\]

**归纳法**：

- **基例** \(t=1\)：\(m^{(1)} = m_0\)，\(\ell^{(1)} = \ell_0\)，\(\mathrm{acc}^{(1)} = \tilde{O}_0\)。成立。

- **归纳步**：设 \(t-1\) 时成立。处理第 \(t\) 组时，令 \(m' = \max(m^{(t-1)}, m_t)\)。

  对任意旧项 \(e^{s - m^{(t-1)}}\)，乘以 \(\alpha = e^{m^{(t-1)} - m'}\) 得 \(e^{s - m'}\)。

  对第 \(t\) 组新项 \(\tilde{p}_{t,j} = e^{s_{t,j} - m_t}\)，乘以 \(\beta = e^{m_t - m'}\) 得 \(e^{s_{t,j} - m'}\)。

  故：

  \[
  \ell^{(t)} = \underbrace{\ell^{(t-1)} \cdot \alpha}_{\text{旧组换基准}} + \underbrace{\ell_t \cdot \beta}_{\text{新组换基准}} = \sum_{g=0}^{t} \sum_{j \in G_g} e^{s_{g,j} - m'}
  \]

  \(\mathrm{acc}^{(t)}\) 同理。令 \(m^{(t)} = m'\)，归纳成立。

处理完全部 8 组后，令 \(M = m^{(8)} = \max_{0 \le j < L} s_j\)（全局 max），则：

\[
\ell^{(8)} = \sum_{j=0}^{L-1} e^{s_j - M} = D_{\text{sum}}
\]

\[
\mathrm{acc}^{(8)} = \sum_{j=0}^{L-1} e^{s_j - M}\, v_j = N
\]

因此 \(O = \mathrm{acc}^{(8)} / \ell^{(8)} = N / D_{\text{sum}}\)，与第 3 节标准 softmax 完全一致。∎

---

## 7. 数值 Walk-through（单 head、单 batch）

取 \(b=0, h=0\)，仅展示 **group 0 → group 1** 的合并（其余 group 同理）。

### Group 0（token 0–255）

假设计算得：

- \(m_0 = 3.0\)
- \(\ell_0 = \sum_{j} e^{s_{0,j} - 3.0} = 12.5\)
- \(\tilde{O}_0 \in \mathbb{R}^{512}\)（由 \(\tilde{p}_0^\top V_0\) 得到）

初始化：\(m^{(0)} = -\infty,\; \ell^{(0)} = 0,\; \mathrm{acc}^{(0)} = \mathbf{0}\)

\[
m' = \max(-\infty,\; 3.0) = 3.0
\]
\[
\alpha = e^{-\infty - 3.0} = 0, \quad \beta = e^{3.0 - 3.0} = 1
\]
\[
\ell^{(1)} = 0 \cdot 0 + 12.5 \cdot 1 = 12.5
\]
\[
\mathrm{acc}^{(1)} = \mathbf{0} \cdot 0 + \tilde{O}_0 \cdot 1 = \tilde{O}_0
\]
\[
m^{(1)} = 3.0
\]

### Group 1（token 256–511）

假设 group 1 有更大的 score：

- \(m_1 = 5.0\)
- \(\ell_1 = 8.0\)
- \(\tilde{O}_1 \in \mathbb{R}^{512}\)

\[
m' = \max(3.0,\; 5.0) = 5.0
\]
\[
\alpha = e^{3.0 - 5.0} = e^{-2} \approx 0.1353
\]
\[
\beta = e^{5.0 - 5.0} = 1
\]
\[
\ell^{(2)} = 12.5 \times 0.1353 + 8.0 \times 1 \approx 1.69 + 8.0 = 9.69
\]
\[
\mathrm{acc}^{(2)} = \tilde{O}_0 \times 0.1353 + \tilde{O}_1 \times 1
\]
\[
m^{(2)} = 5.0
\]

**解读**：group 0 的贡献被乘以 \(e^{3.0 - 5.0}\)，因为全局 max 升到了 5.0；旧项的 exp 权重需按新基准缩小，否则分母分子尺度不一致。

继续 group 2–7 后，一次除法得到 \(O \in \mathbb{R}^{512}\)。

---

## 8. Paged KV Cache 寻址

稀疏 `indices[b, 0, j]` 给出 token 的 **全局线性下标** `idx`。`page_size P = 64` 时：

\[
\text{block} = \left\lfloor \frac{\text{idx}}{64} \right\rfloor, \quad \text{pos} = \text{idx} \bmod 64
\]

\[
k_j = \texttt{kv\_cache}[\text{block},\, \text{pos},\, 0,\, :], \quad v_j = k_j \quad \text{（MLA 同址）}
\]

**例**：`idx = 130`

\[
\text{block} = 130 // 64 = 2, \quad \text{pos} = 130 - 2 \times 64 = 2
\]

\[
k = \texttt{kv\_cache}[2,\, 2,\, 0,\, 0{:}512]
\]

每个 group 256 个 slot 的 `idx` 通常 **随机分布** 在不同 page，因此 K/V 是 scattered gather，无法像连续 FA 那样 coalesce。

---

## 9. Kernel 执行映射（Triton FA）

以 `gqa_flash_mla_sparse_fa_kernel` 为例，典型 tile：`BLOCK_H=16, GROUP_SIZE=256, D=512`。

### 9.1 Grid

\[
\text{grid} = \left(\left\lceil \frac{H}{\text{BLOCK\_H}} \right\rceil,\; B\right) = \left(\left\lceil \frac{64}{16} \right\rceil,\; 4\right) = (4,\; 4)
\]

共 **16 个 program**，每个负责：

- 一个 `batch_idx` \(\in \{0,1,2,3\}\)
- 连续 **16 个 head**（`head_block_idx × 16 + [0..15]`）

### 9.2 每个 program 的常驻状态（正确 FA）

| 变量 | 形状 | 本例寄存器量级（fp32） |
|------|------|------------------------|
| `running_max` | `[BLOCK_H]` = `[16]` | 16 × 4 B = 64 B |
| `running_denom` | `[16]` | 64 B |
| `acc` | `[16, 512]` | 16 × 512 × 4 B = **32 KB** |

**每个 group 的临时量**（循环内，可复用）：

| 变量 | 形状 | 说明 |
|------|------|------|
| `scores` | `[16, 256]` | QK^T，16 KB |
| `exp_scores` | `[16, 256]` | 组内 exp，16 KB |
| `k_vals` / `v_vals` | gather 自 KV | 256 × 512 per tile |

### 9.3 内层循环结构

```
for group_idx in 0..7:                    # T = 8
    gather 256 个 idx → (block, pos)
    scores[16, 256] = Q[16, 512] @ K[512, 256]^T * sm_scale
    group_max[16] = rowmax(scores)
    exp_scores[16, 256] = exp(scores - group_max)
    exp_sum[16] = rowsum(exp_scores)
    partial[16, 512] = exp_scores @ V[256, 512]   # 未归一化 PV

    # online softmax merge
    new_max = max(running_max, group_max)
    scale_old = exp(running_max - new_max)
    scale_group = exp(group_max - new_max)
    acc = acc * scale_old + partial * scale_group
    running_denom = running_denom * scale_old + exp_sum * scale_group
    running_max = new_max

out[16, 512] = acc / running_denom
```

### 9.4 与 PyTorch 参考的对应

`31_gqa_flash_mla_with_kv_cache.py` 中（所有 head 批量版）：

```python
running_max   # [H_Q] = [64] per batch
running_denom # [64]
acc           # [64, 512]

# 每个 group:
scores        # [64, 256]
partial_unnorm = exp_scores @ k  # [64, 512]
acc = acc * scale_old[:, None] + partial_unnorm * scale_group[:, None]
running_denom = running_denom * scale_old + exp_sum * scale_group
```

---

## 10. 全配置下的计算量一览

| 量 | 值 |
|----|-----|
| 每 \((b, h)\) 的 group 数 \(T\) | 8 |
| 每 group QK FLOPs | \(2 \times D \times G = 2 \times 512 \times 256 \approx 262\text{K}\) |
| 每 group PV FLOPs | \(2 \times G \times D \approx 262\text{K}\) |
| 每 \((b, h)\) 总 FLOPs（QK+PV） | \(\approx 8 \times 524\text{K} \approx 4.2\text{M}\) |
| 全部 \((B, H)\) | \(4 \times 64 \times 4.2\text{M} \approx 1.07\text{G FLOPs}\) |
| 物化 score 矩阵大小（若不做 FA） | \(B \times H \times L = 524{,}288\) floats ≈ 2 MB |

---

## 11. 错误实现警示：`sum_values [G, D]`

一种错误写法是为每个 slot 维护完整输出向量：

```python
sum_values = tl.zeros((group_size, head_size))  # [256, 512] → 512 KB / program
```

问题：

1. **内存**：应为 `acc [D]`（2 KB / head），却变成 `acc [G, D]`（512 KB），放大 **256 倍**；`head_size=512, group_size=256` 时单 tensor 即 **512 KB** 寄存器压力。
2. **算法**：若 `head_max/sum_values` 按 `page_offs ∈ [0, 255]` 索引，**每个 group 会覆盖上一 group 的同 slot**，仅当 \(L \le G\) 时碰巧正确；本例 \(L=2048, G=256\) 时 **结果错误**。
3. **正确状态数**：每个 head 只需 **3 个** 跨 group 累加器 `(running_max, running_denom, acc)`，不是 \(G\) 套。

---

## 12. 符号对照表

| 数学符号 | Triton / PyTorch 变量 | 形状（本例，per program） |
|----------|----------------------|---------------------------|
| \(m^{(t)}\) | `running_max` | `[16]` |
| \(\ell^{(t)}\) | `running_denom` | `[16]` |
| \(\mathrm{acc}^{(t)}\) | `acc` | `[16, 512]` |
| \(m_t\) | `group_max` | `[16]` |
| \(\ell_t\) | `exp_sum` | `[16]` |
| \(\tilde{O}_t\) | `partial_unnorm` | `[16, 512]` |
| \(\alpha\) | `scale_old` | `[16]` |
| \(\beta\) | `scale_group` | `[16]` |
| \(O\) | `out` | `[16, 512]` |

---

## 13. 总结

1. **目标**：\(O = \mathrm{softmax}(\text{sm\_scale} \cdot q K^\top) V\)，不物化长度 2048 的 score 向量。
2. **手段**：按 `group_size=256` 切 8 组，组内算未归一化 \(\tilde{p}\) 和 \(\tilde{O}_t\)，组间用 \((m, \ell, \mathrm{acc})\) 在线合并。
3. **核心递推**：\(m' = \max(m_{\text{old}}, m_t)\)，\(\alpha = e^{m_{\text{old}}-m'}\)，\(\beta = e^{m_t-m'}\)，分母分子同步 rescale。
4. **本例配置**：\(B=4, H=64, D=512, L=2048, G=256, P=64\) → 8 次 group 循环，16 个 Triton program，每 program 16 head × 512 dim 的 `acc`。
5. **状态规模**：\(O(D)\) per head，**不是** \(O(G \times D)\)。
