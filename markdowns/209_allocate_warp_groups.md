# `tritongpu-allocate-warp-groups` Pass 分析

```python
# triton/third_party/nvidia/backend/compiler.py
passes.ttgpuir.add_allocate_warp_groups(pm)
```

## 一、概述

| 属性 | 值 |
|------|-----|
| **Pass 名称（CLI）** | `tritongpu-allocate-warp-groups` |
| **TableGen 定义** | `def TritonGPUAllocateWarpGroups : Pass<"tritongpu-allocate-warp-groups", "mlir::ModuleOp">` |
| **C++ 实现** | `triton/lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp` |
| **Python 绑定** | `ADD_PASS_WRAPPER_0("add_allocate_warp_groups", createTritonGPUAllocateWarpGroups)` |
| **操作粒度** | `mlir::ModuleOp` |
| **关键依赖** | `ttg::WarpSpecializeOp`、`ttg::WarpSpecializePartitionsOp` |

**摘要（来自 TableGen）**：

> 为 GPU 程序执行 warp group 分配。当 GPU 程序包含 warp specialization 时，除了"默认"warp group 之外还会启动额外的 warp。默认 warp group 执行 `tt.func` 中的顶层代码，其大小由用户通过 `num_warps` 参数指定。
>
> 本 pass 分析程序中的 `ttg.warp_specialize` 操作，确定所需 warp 总数，然后将 warp ID 范围附加到每个 warp group 函数。

### 什么是 Warp Group？

NVIDIA GPU 从 Hopper (SM90) 架构开始引入了 **warp group** 的概念：

| 概念 | 说明 |
|------|------|
| **Warp** | 32 个线程组成的执行单元 |
| **Warp Group** | 4 个 warp（128 个线程）组成的组 |
| **用途** | 支持 warp specialization、TMEM（Tensor Memory）分配等特性 |
| **对齐要求** | warp group 的起始 warp ID 必须是 4 的倍数 |

SM100 (Blackwell) 及其后续架构广泛使用 warp group 来实现高级并发特性，是 **Tensor Memory（TMEM）** 和 **Warp Specialization** 的基础单位。

---

## 二、背景：Warp Specialization 的硬件模型

在传统 GPU 执行模型中，程序的所有 warp 执行相同的指令流（SIMT）。Warp Specialization 打破了这一约束，允许**不同的 warp 组执行不同的代码路径**。

```
传统模型（无 warp specialization）：
┌─────────────────────────────────────┐
│  所有 warp 执行同一份 kernel 代码    │
│  warp 0 ────┐                      │
│  warp 1 ────┤ 同一 PC              │
│  warp 2 ────┤                      │
│  warp 3 ────┘                      │
└─────────────────────────────────────┘

Warp Specialization 模型：
┌─────────────────────────────────────┐
│  Default WG (num_warps=4)           │
│  warp 0-3 ──── 主程序逻辑           │
├─────────────────────────────────────┤
│  Partition 0 (num_warps=1)          │
│  warp 4     ──── 数据加载单元       │
├─────────────────────────────────────┤
│  Partition 1 (num_warps=8)          │
│  warp 5-12  ──── 计算单元（MMA）    │
└─────────────────────────────────────┘
```

`ttg.warp_specialize` 在 MLIR 中用如下结构表达：

```mlir
ttg.warp_specialize()
default {
  // 默认 warp group 执行的代码
  ttg.warp_yield
}
partition0() num_warps(1) {
  // 分区 0 执行的代码
  ttg.warp_return
}
partition1() num_warps(8) {
  // 分区 1 执行的代码
  ttg.warp_return
} : () -> ()
```

---

## 三、核心功能

本 pass 完成以下三件事：

| 功能 | 输入 | 输出 | 说明 |
|------|------|------|------|
| **① 填充到完整 warp group** | `ttg.warp_specialize` 的分区 | 补充分区使总 warp 数为 4 的倍数 | 所有线程必须到场才能"上交"寄存器 |
| **② 分配 start ID** | 各分区的 warp 数量 | `warpGroupStartIds` 属性 | 确保大 warp group 获得较低的起始 ID |
| **③ 分配寄存器 budget** | `requestedRegisters` 属性 | `actualRegisters` 属性 + `ttg.maxnreg` | 按 warp group 计算寄存器上限 |

### 流程总览

```
ModuleOp
   │
   ▼
┌────────────────────────────────────────────────┐
│ Step 1: 扫描所有 ttg.warp_specialize           │
│         找出最大的 partition 总 warp 数          │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ Step 2: padToMaxWarpGroups                     │
│         补充分区使每个 warp_specialize 的       │
│         partition warp 总数 = maxWG * 4        │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ Step 3: 分配 warp 起始 ID                      │
│         大分区优先分到较低的 start ID            │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ Step 4: 为每个 warp group 计算寄存器 budget    │
│         设置 actualRegisters 和 maxnreg        │
└───────────────────────────────────────────────┘
                 │
                 ▼
            ModuleOp (annotated)
```

---

## 四、Step 1 + 2：填充到最大 warp group（`padToMaxWarpGroups`）

### 4.1 为什么要填充？

**关键约束**：在 GPU 上，所有 warp 必须能同时执行 warp specialization 的调度，但当不同 `ttg.warp_specialize` 用了不同数量的 warp 时，跨 kernel 的资源分配（尤其是寄存器）就会不一致。

具体来说，当 `warp_specialize` 需要在 warp group 之间**重新分配寄存器**（register surrender）时，整个 warp group 的所有 warp 必须同时在场。

### 4.2 填充算法

```cpp
static void padToMaxWarpGroups(WarpSpecializeOp op, int numExtraWarpGroups) {
  // 当前 warp_specialize 的 partition warp 总数
  int numExtraWarps = op.getTotalPartitionWarps();
  // 需要补充的 warp 数
  int warpsToAdd = numExtraWarpGroups * 4 - numExtraWarps;

  // 用 2 的幂填充，确保分区大小对齐
  SmallVector<int> paddingPartitionSizes;
  while (warpsToAdd > 0) {
    int paddingSize = llvm::NextPowerOf2(warpsToAdd) / 2;
    paddingPartitionSizes.push_back(paddingSize);
    warpsToAdd -= paddingSize;
  }
  // ... 创建空操作分区（pad partition），占用 warp 但不做实际工作
}
```

**填充示例**：

```
module 中有两个 warp_specialize:
  ws1: partitions 使用 1+8+4 = 13 个 warp
  ws2: partitions 使用 2+1   = 3 个 warp

maxExtraWarps = max(13, 3) = 13
numExtraWarpGroups = ceil(13/4) = 4  → 需要 4×4=16 个 warp

ws1 已用 13 warp，补充 3 warp → 填充 2+1 的两个空分区
ws2 已用 3 warp，补充 13 warp → 填充 8+4+1 的三个空分区
```

### 4.3 填充后的 `ttg.warp_specialize`

```
// 填充前
ttg.warp_specialize()
default { ... }
partition0() num_warps(1) { ... }
partition1() num_warps(8) { ... }
partition2() num_warps(4) { ... }

// 填充后（假设需要 16 个额外 warp）
ttg.warp_specialize()
default { ... }
partition0() num_warps(1) { ... }
partition1() num_warps(8) { ... }
partition2() num_warps(4) { ... }
partition3() num_warps(2) { ... }   ← 空分区（warp_return 直接返回）
partition4() num_warps(1) { ... }   ← 空分区
```

---

## 五、Step 3：分配 warp 起始 ID

### 5.1 分配策略

分配策略的核心原则：**最大的 warp group 获得最低的起始 warp ID**。

```
用 baseNumWarps 表示默认 warp group 的大小（如 4 warp）。

分配步骤：
1. 将 (partition_index, num_warps) 按 num_warps 降序排序
2. 从 baseNumWarps 开始分配
3. 将 startIds 写回原始 partition 顺序

示例：
  partition0: num_warps=1  → startId=18（最后分配）
  partition1: num_warps=8  → startId=4（最先分配）
  partition2: num_warps=4  → startId=12
  partition3: num_warps=2  → startId=16
  partition4: num_warps=1  → startId=19
```

### 5.2 为何大分区优先？

**原因**：warp group 需要在连续的 warp ID 上运作。大分区更有可能跨越多个 warp group，让大分区先分配可以确保它们获得连续的、低地址的 warp ID，减少 warp ID 碎片化。

---

## 六、Step 4：寄存器分配（`actualRegisters` + `maxnreg`）

### 6.1 寄存器预算计算

这是最复杂的步骤，只有当 `requestedRegisters` 属性已设置且 partition warp 数为 4 的倍数时才会执行。

```
registerBudget = maxnreg × baseNumWarps × threadsPerWarp

对每个 warp group:
  registerBudget += (maxnreg - wg.maxRequestedRegs) × wg.numWarps × threadsPerWarp

maxnreg 的确定：
  如果用户设置了 ttg.maxnreg → 直接使用
  否则 → (64KB) / (baseNumWarps + numExtraWarpGroups×4) / threadsPerWarp
         → 向下取整到 8 的倍数
```

### 6.2 寄存器流向逻辑

```
┌─ 寄存器预算 ──────────────────────────────────────────────────┐
│                                                                │
│  默认 warp group:                                              │
│    maxnreg × baseNumWarps × 32  (寄存器总量)                   │
│    │                                                           │
│    ├── 默认 warp group 自身使用: maxnreg × baseNumWarps × 32  │
│    │                                                           │
│    └── 从各 warp group 回收:                                   │
│        (maxnreg - 该 wg 实际需求) × 该 wg warp 数 × 32       │
│              ↑                      ↑                          │
│         省下的寄存器        该 warp group 包含的 warp 数       │
│                                                                │
│  回收的总寄存器 → 重新分配给默认 warp group                    │
│  leftover = registerBudget / (baseNumWarps × 32)               │
│  → 向下取整到 8 的倍数                                        │
│  → 如果 < 24，放弃分配（太少）                                 │
└────────────────────────────────────────────────────────────────┘
```

### 6.3 PTX 对应

最终生成的 `actualRegisters` 和 `ttg.maxnreg` 属性会被后续的 LLVM/PTX 代码生成阶段转换为 `setmaxnreg` PTX 指令，用于在 warp group 切换时动态调整寄存器文件。

---

## 七、Pipeline 中的位置

```
make_llir pipeline (NVIDIA backend):
  ...
  passes.ttgpuir.add_combine_tensor_select_and_if(pm)
  passes.ttgpuir.add_allocate_warp_groups(pm)    ← 第 2 个 pass
  passes.convert.add_scf_to_cf(pm)
  passes.gluon.add_inliner(pm)
  nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, ...)
  nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
  ...
  passes.ttgpuir.add_allocate_global_scratch_memory(pm)
  ...
  nvidia.passes.ttgpuir.add_to_llvmir(pm, ...)
  ...
  nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
  nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
  ...
```

本 pass 位于 `combine-tensor-select-and-if` 之后、`scf-to-cf` 之前。这一位置确保了：

1. **在控制流结构化之前** — warp specialization 的结构化 IR（`ttg.warp_specialize`）必须在转换为 CFG 之前完成分析和标注
2. **在内存分配之前** — warp group 数量和寄存器预算会影响后续的 shared memory、tensor memory 和 global scratch memory 分配
3. **在 LLVM 转换之前** — warp group 的标注信息最终会被 `warp-specialize-to-llvm` pass 使用

---

## 八、已有测试文件分析

测试文件位于 `triton/test/Conversion/allocate_warp_groups.mlir`，包含四个测试场景：

### 测试 1：空模块

```mlir
// CHECK: "ttg.total-num-warps" = 4 : i32
module attributes {"ttg.num-warps" = 4 : i32} {}
```

没有 `ttg.warp_specialize` 时，total-num-warps = num-warps（4）。

### 测试 2：基础 warp specialization 填充 + ID 分配

```mlir
// input: 3 个 partition (1+8+4=13 warps)
// padded to: 16 warps → 4 个 warp group
// CHECK: warpGroupStartIds = array<i32: 18, 4, 12, 16, 19>
```

- 最大 warp 数 = 13 → `numExtraWarpGroups=4` → 需 16 个 extra warp
- 补充 3 个 warp → 填充 2+1 两个空分区
- 分区排序：8(partition1) > 4(partition2) > 2(pad0) > 1(partition0) > 1(pad1)
- startId 从 4(baseNumWarps) 开始分配：4, 12+4=16, 16+2=18, 18+1=19
- 恢复原始顺序后：partition0=18, partition1=4, partition2=12, pad0=16, pad1=19

### 测试 3：两个 `ttg.warp_specialize`

验证多个 warp_specialize 时填充到相同的 warp group 数量且各自独立分配 start ID。

### 测试 4：寄存器分配（`setmaxnreg`）

```mlir
// input: num_warps=8, requestedRegisters=[48, 80, 48]
// CHECK: ttg.maxnreg = 168 : i32
// CHECK: actualRegisters = array<i32: 208, 80, 80, 80>
```

- `baseNumWarps=8`，partitions 使用 1+2+1=4 个 warp
- `numExtraWarpGroups=1`，`maxnreg=(64K)/(8+4)/32=42.67→40`（向下取整到 8 的倍数）
- FIXME: 实际输出是 `168`，需要进一步追踪代码

### 测试 5：从默认 warp group 借用寄存器

```mlir
// input: num_warps=8, requestedRegisters=[192]
// 一个 partition 使用 8 warp
// CHECK: actualRegisters = array<i32: 64, 192>
```

当 warp group 需要的寄存器超过均分预算时，从默认 warp group 回收。

---

## 九、关键设计要点

1. **填充到 warp group 边界**：所有 warp_specialize 的 partition warp 总数被填充到相同的 warp group 数（4 的倍数），确保资源分配的一致性。

2. **大分区优先分配**：在分配 warp start ID 时，最大的分区获得最低的起始 ID，最小化 warp ID 碎片化。

3. **寄存器回收机制**：warp group 没用完的寄存器预算可以被回收并重新分配给默认 warp group。这充分利用了 warp specialization 的时间复用特性——当一个 warp group 不活跃时，其寄存器可以被其他 warp group 使用。

4. **空分区（padding partition）**：填充创建的额外分区是"空操作"分区——它们直接执行 `warp_return` 不做任何工作。这些分区确保了 warp group 的对齐但不增加实际计算。

5. **全模块分析**：pass 在 `ModuleOp` 级别对整个模块进行全局分析，确保所有 `ttg.warp_specialize` 的标注一致。

6. **`ttg.total-num-warps` 属性**：pass 在所有 `ttg.warp_specialize` 之外，还会在 module 上设置 `ttg.total-num-warps` 属性，记录程序所需的总 warp 数（包括默认和所有 extra warp）。

---

## 十、与相关 Pass 的关系

| Pass | 关系 |
|------|------|
| `tritongpu-optimize-partition-warps` | 前一步，确定每个 partition 需要的 warp 数量 |
| `tritonnvgpu-warp-specialize-to-llvm` | 后一步，使用本 pass 标注的 ID 和寄存器信息生成 LLVM IR |
| `tritonnvgpu-allocate-tensor-memory` | 在后执行，warp group 的对齐会影响 TMEM 的分配 |
| `tritongpu-combine-tensor-select-and-if` | 在本 pass 之前执行，精简 IR |

---

## 十一、深入理解：Warp Specialization 的原理

### 11.1 需要硬件支持吗？

**是的，warp specialization 需要特定硬件架构支持。**

| 架构 | 世代 | 支持 Warp Specialization？ | 原因 |
|------|------|--------------------------|------|
| SM70 (Volta) | V100 | ❌ | 所有 warp 共享 PC，执行同一指令流 |
| SM80 (Ampere) | A100 | ❌ | 虽然引入了独立线程调度，但不支持 warp group 寄存器隔离 |
| SM90 (Hopper) | H100 | ✅ | 引入 Warp Group 概念、TMEM、独立 warp 调度 |
| SM100 (Blackwell) | B100/B200 | ✅ (增强) | 更大的寄存器文件、更灵活的 warp group 切换 |

**硬件需要提供以下能力**：

```
1. 多程序计数器（Multiple PC）
   传统 GPU: 32 个 warp 共享 1 个 PC
             └── 同一时刻所有 warp 执行同一行代码

   Warp Specialization: 每个 warp group 有独立的 PC
             └── WG-0 执行加载，WG-1 执行计算，互不干扰

2. 寄存器隔离与移交（Register Surrender）
   当一个 warp group 切换到另一个时，硬件必须支持：
     - 保存当前 warp group 的寄存器状态
     - 加载目标 warp group 的寄存器状态
   → 对应 PTX 指令: setmaxnreg

3. 独立 Warp 调度
   不同 warp group 可以被调度到不同的硬件资源上并行执行
```

如果没有硬件支持（比如在 SM80 上模拟），不同 warp 虽然也可以跑不同代码，但**寄存器文件无法隔离**，所有 warp 共享同一份寄存器预算，导致 warp specialization 无法安全切换。这是为什么本 pass 的核心工作之一是**寄存器预算计算**。

### 11.2 在 kernel 内部，确实是根据 warp_id 进行 dispatch

**完全正确。** `allocate-warp-groups` pass 分配 `warpGroupStartIds` 后，kernel 内部通过 warp_id 来决定每个 warp 去哪个 partition 执行。

```
GPU 启动 kernel:
  ┌───────────────────────────────────────────────────────┐
  │  总共启动 20 个 warp (默认 4 + extra 16)              │
  │                                                       │
  │  warp_id 0-3  →  默认 warp group（主函数逻辑）        │
  │  warp_id 4-11 →  partition1 (MMA 计算，8 warp)       │
  │  warp_id 12-15 → partition2 (后处理，4 warp)          │
  │  warp_id 16-17 → partition0 (加载，1 warp 但占用 2)  │
  │  warp_id 18-19 → padding（空转，不做事）              │
  └───────────────────────────────────────────────────────┘

硬件 dispatch 逻辑（伪代码）：
  int start_ids[] = {18, 4, 12, 16, 19};  // warpGroupStartIds
  int sizes[]     = { 1, 8,  4,  2,  1};  // partition 大小
  
  int my_warp_id = get_warp_id();
  
  for (int i = 0; i < num_partitions; i++) {
    if (my_warp_id >= start_ids[i] &&
        my_warp_id < start_ids[i] + sizes[i]) {
      execute_partition(i);  // 跳转到对应 partition 的代码
      break;
    }
  }
  // 如果没匹配到 → 执行默认代码（default）
```

**warp_id 是一个硬件提供的唯一标识**，每个 warp 都知道自己在 warp 空间中的位置。

### 11.3 不同 warp 之间如何协作？核心是"生产者-消费者"流水线

**这正是 warp specialization 的精髓所在。** 它不是让不同 warp 处理不同数据（那是传统的数据并行），而是让不同 warp 分工负责流水线的不同阶段：

```
┌─────────────────────────────────────────────────────────┐
│                    传统模型（数据并行）                    │
│                                                         │
│  所有 warp 执行同一份代码：                              │
│                                                         │
│  warp 0: 加载 → 计算 → 存回                              │
│  warp 1: 加载 → 计算 → 存回                              │
│  warp 2: 加载 → 计算 → 存回                              │
│  ...                                                     │
│                                                         │
│  问题: 加载阶段所有 warp 都在等内存，计算阶段又一起忙    │
│        导致资源利用不均衡，延迟无法隐藏                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│               Warp Specialization 模型（流水线并行）      │
│                                                         │
│  不同 warp 各司其职，形成生产者-消费者链：                │
│                                                         │
│  WG 加载 (warp 4-5)    → 从 global memory 加载到        │
│      │                     shared memory                 │
│      ▼                                                  │
│  WG 计算 (warp 6-13)   → 从 shared memory 读取，MMA 计算│
│      │                                                   │
│      ▼                                                  │
│  WG 存储 (warp 14-15)  → 从 shared memory 写回            │
│                                                         │
│  三者可以同时进行（流水线重叠）：                         │
│  时间 ──────────────────────────────────────→             │
│  WG加载: [加载 tile 0]  [加载 tile 1]  [加载 tile 2]     │
│  WG计算:     [计算 tile 0]   [计算 tile 1]   [计算 tile 2]│
│  WG存储:        [存回 tile 0]   [存回 tile 1]            │
│                                                         │
│  效果: 流水线填满后，三个阶段并行执行                    │
└─────────────────────────────────────────────────────────┘
```

### 11.4 为什么能更好地平衡 workload？

理解这一点需要回到 GPU 的**延迟隐藏**模型：

```
传统 GPU 的延迟隐藏方式：
  ┌─ 大量 warp 做同一件事 ──────────────────────────┐
  │                                                   │
  │  当 warp 0 等待内存时，切换到 warp 1 执行          │
  │  但所有 warp 都在做同样的事，所以：                │
  │    - 如果都在等内存 → 也都切换到内存等待          │
  │    - 如果都在计算 → 计算单元饱和，延迟不存在      │
  │                                                   │
  │  问题在于：内存命中和不命中的 warp 互相等待        │
  └─────────────────────────────────────────────────┘

Warp Specialization 的延迟隐藏方式：
  ┌─ 不同 warp 做不同的事 ──────────────────────────┐
  │                                                   │
  │  加载 warp: 永远在发 TMA 请求                     │
  │      → 完全不会被计算阻塞                          │
  │      → 内存带宽利用率接近 100%                    │
  │                                                   │
  │  计算 warp: 永远在算 MMA                          │
  │      → 数据永远在 shared memory 中等它            │
  │      → 计算单元利用率接近 100%                    │
  │                                                   │
  │  两者没有资源竞争（寄存器独立），可以同时全速运行  │
  └─────────────────────────────────────────────────┘
```

**核心洞察**：传统 GPU 用大量 warp 的"时间分片"来隐藏延迟；warp specialization 用**功能划分**来消除延迟。

| 对比维度 | 传统数据并行 | Warp Specialization |
|----------|-------------|-------------------|
| **warp 关系** | 独立、相同 | 协作、分工 |
| **隐藏延迟方式** | warp 间时间分片 | 流水线并行 |
| **内存带宽** | 受计算阻塞 | 持续饱和 |
| **计算利用率** | 受内存阻塞 | 持续饱和 |
| **寄存器需求** | 所有 warp 均等 | 不同 warp group 不同 |
| **适用场景** | 通用计算 | HPC/AI 内核（GEMM、Attention） |
| **编程模型** | 单 kernel，所有 warp 同代码 | Triton `warp_specialize` 分区 |

### 11.5 Triton 中的实际编程模式

在 Triton 中，用户通过以下方式表达 warp specialization：

```python
@triton.warp_specialize
def my_kernel(X, Y, Z, ...):
    # 默认 warp group：主控制流
    pid = tl.program_id(0)
    
    # partition 0：数据加载（1 warp）
    @triton.warp_partition(1)
    def load():
        data = tl.load(X + offsets)
        tl.store(shared_buf, data)
    
    # partition 1：矩阵计算（8 warp）
    @triton.warp_partition(8)
    def compute():
        a = tl.load(shared_buf)
        b = tl.load(shared_buf + offset)
        acc = tl.dot(a, b)
        tl.store(shared_result, acc)
    
    # partition 2：结果存回（1 warp）
    @triton.warp_partition(1)
    def store():
        result = tl.load(shared_result)
        tl.store(Y + offsets, result)
```

编译器会将此转换为：
1. `optimize-partition-warps` pass — 确定每个 partition 的最佳 warp 数
2. `allocate-warp-groups` pass — 分配 warp ID 和寄存器（本 pass）
3. `warp-specialize-to-llvm` pass — 生成实际 dispatch 代码

---

*本文档对应源代码：[AllocateWarpGroups.cpp](../../triton/lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp)*
