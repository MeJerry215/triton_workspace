# `allocate-shared-memory` Pass 分析

```python
# NVIDIA backend: triton/third_party/nvidia/backend/compiler.py
nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)

# Generic pass:  passes.ttgpuir.add_allocate_shared_memory(pm)
```

## 一、概述

| 属性 | 值 |
|------|-----|
| **Pass 名称（Generic CLI）** | `allocate-shared-memory` |
| **Pass 名称（NVIDIA CLI）** | `allocate-shared-memory-nv` |
| **操作粒度** | `mlir::ModuleOp` |
| **关键依赖** | `ModuleAllocation` 分析框架、`ttg.shared` 属性 |

**摘要（来自 Generic TableGen）**：

> 此 pass 使用 `ModuleAllocation` 分析来：
> - 在 module 上标注 shared/local memory 使用的总量
> - 在每个操作上标注其在 shared/local memory 中的偏移量

### 本 pass 的输出

本 pass 在 IR 上添加两种关键属性：

| 属性 | 位置 | 含义 |
|------|------|------|
| `allocation.offset` | 每个需要 shared memory 的操作上 | 该操作在 shared memory 中的起始偏移量（字节） |
| `ttg.shared` | module 上 | 整个模块所需 shared memory 总大小（字节） |

这些属性被后续的 LLVM IR 代码生成阶段用来生成实际的 shared memory 基址指针和偏移量计算。

### Pipeline 中的位置

```
make_llir pipeline (NVIDIA backend):
  │
  ├─ ... 前列 pass ...
  ├─ ④ gluon-inliner                              ← 内联
├─ ⑤ allocate-shared-memory / allocate-shared-memory-nv  ← 本 pass
  ├─ ⑥ tritongpu-to-llvmir                        ← TTGPUIR → LLVM IR
  ├─ ⑦ nvgpu-to-llvm / warp-specialize-to-llvm
  └─ ⑧ canonicalizer + cse + symbol-dce
```

在 NVIDIA backend 中，本 pass 运行在 `gluon-inliner` 之后、`triton-to-llvmir` 之前。这一位置确保了：

1. **内联已完成** — 所有函数调用已被内联，shared memory 的 liveness 分析在单一函数内完成
2. **所有需要 shared memory 的操作均已确定** — `convert_layout`、`reduce`、`scan` 等操作在此时已经存在，它们的 scratch buffer 需求可以被准确计算
3. **在 LLVM 转换之前** — 标注的 offset 和 size 属性将被后续的 LLVM 代码生成阶段使用

---

## 二、核心概念：`ModuleAllocation` 分析框架

本 pass 的核心是一个全局共享内存分配算法，实现在 `triton/lib/Analysis/Allocation.cpp` 的 `ModuleAllocation` 类中。

### 2.1 问题描述

GPU kernel 中有多个操作需要共享内存（shared memory / local memory）：

| 需要 shared memory 的操作 | 原因 |
|--------------------------|------|
| `ttg.convert_layout` | 在不同数据布局之间转换时，需要暂存 buffer |
| `tt.reduce` | 跨 warp 归约时，需要 shared memory 进行数据交换 |
| `tt.scan` | 扫描操作需要 shared memory 暂存中间结果 |
| `tt.gather` | gather 操作需要 scratch buffer |
| `tt.histogram` | 直方图操作需要 shared memory 暂存 |
| `tt.atomic_rmw` / `tt.atomic_cas` | 原子操作需要 shared memory 暂存 |
| `ttg.local_alloc` | 用户显式分配的 shared memory buffer |

这些操作的生命周期可能重叠，也可能不重叠。**目标是在满足所有生命周期约束的前提下，最小化 shared memory 的总占用。**

### 2.2 分配算法：图着色（Graph Coloring）

`ModuleAllocation` 使用基于**图着色（graph coloring）**的寄存器分配风格算法：

```
┌─ 输入：函数内所有需要 shared memory 的操作 ──────────────────────┐
│                                                                   │
│  Step 1: Liveness Analysis                                        │
│  对每个操作，计算其在 CFG 中的活跃范围（live range）               │
│  └─ 一个操作的 live range 从定义点开始，到最后一次使用结束        │
│                                                                   │
│  Step 2: Interference Graph Construction                          │
│  如果两个操作的 live range 存在重叠 → 它们不能共享同一块内存       │
│  └─ 构建干涉图：节点 = 操作，边 = 活跃范围重叠                    │
│                                                                   │
│  Step 3: Graph Coloring (Offset Assignment)                       │
│  对干涉图进行着色，不同颜色代表不同的偏移量                       │
│  └─ 使用贪心算法，按操作大小降序分配偏移量                         │
│                                                                   │
│  Step 4: Total Size Computation                                   │
│  所有颜色的最大偏移量 + 其大小 = 总 shared memory 大小            │
│  └─ 结果写入 `ttg.shared` 属性                                    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 2.3 `ModuleAllocation` 构造函数关键参数

```cpp
class ModuleAllocation : public CallGraph<Allocation> {
public:
  ModuleAllocation(ModuleOp moduleOp,
                   AllocationAnalysisScratchSizeFn scratchSizeGetter = 
                       defaultAllocationAnalysisScratchSizeFn);
```

- **`scratchSizeGetter`** 是一个回调函数，用于计算**每个操作所需的 scratch buffer 大小**（以字节为单位）
- 不同的 backend 可以传入不同的实现，以针对特定硬件进行优化
- 默认实现 `defaultAllocationAnalysisScratchSizeFn` 支持所有通用操作

---

## 三、两种实现的对比

Triton 中有两种 `allocate-shared-memory` 实现：

| 实现 | 文件 | Pass 名称 | Scratch Size 函数 | 特性 |
|------|------|-----------|-------------------|------|
| **Generic** | `lib/Conversion/TritonGPUToLLVM/AllocateSharedMemory.cpp` | `allocate-shared-memory` | `defaultAllocationAnalysisScratchSizeFn` | 通用实现，使用标准 swizzling |
| **NVIDIA** | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp` | `allocate-shared-memory-nv` | `getNvidiaAllocationAnalysisScratchSizeFn` | 针对 CUDA GPU 优化，使用 TargetInfo 感知的 swizzling |

### 3.1 Generic 实现

```cpp
// triton/lib/Conversion/TritonGPUToLLVM/AllocateSharedMemory.cpp
struct AllocateSharedMemory
    : public AllocateSharedMemoryBase<AllocateSharedMemory> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod);  // 使用默认 scratchSizeGetter
    attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};
```

**特点**：
- 使用 `defaultAllocationAnalysisScratchSizeFn` 计算每个操作的大小
- 对 `ConvertLayoutOp` 使用标准 swizzling（`getNumScratchElemsSwizzledCvt`）

### 3.2 NVIDIA 实现

```cpp
// triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp
struct AllocateSharedMemoryNv
    : public AllocateSharedMemoryNvBase<AllocateSharedMemoryNv> {
  AllocateSharedMemoryNv(int32_t computeCapability, int32_t ptxVersion)
      : AllocateSharedMemoryNvBase({computeCapability, ptxVersion}) {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    TargetInfo targetInfo(computeCapability, ptxVersion);
    ModuleAllocation allocation(
        mod, getNvidiaAllocationAnalysisScratchSizeFn(targetInfo));
    attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};
```

**特点**：
- 接受 `computeCapability` 和 `ptxVersion` 参数
- 使用 `getNvidiaAllocationAnalysisScratchSizeFn(targetInfo)` 计算 scratch buffer 大小
- 对 `ConvertLayoutOp` **始终使用 CUDA 特定的 swizzling**（注释："In cuda we always swizzle"）
- 使用 `optimalSwizzling` 函数，基于目标架构计算最优的 swizzling tile 尺寸

---

## 四、Shared Memory 分配策略详解

### 4.1 Swizzling（交错）

Swizzling 是一种通过重新排列 shared memory 中的地址映射来减少 bank conflict 的技术。

#### 背景：什么是 Bank Conflict？

NVIDIA GPU 的 shared memory 被划分为 **32 个 bank**（以较新架构为例），每个 bank 的带宽为 **4 字节**。地址到 bank 的默认映射规则是：

```
bank_id = (byte_address / 4) % num_banks
```

即：**同一个 warp 中的 32 个线程，如果同时访问同一个 bank 中的不同地址，就会发生 bank conflict，导致访问串行化。**

下图用 **4 个 bank** 简化演示（实际为 32 个 bank）。图中 **`[x]` 表示第 x 个 4 字节 word**，即字节地址范围 `x*4 ~ x*4+3`。

以 **4×4 矩阵按行优先存储在 shared memory** 为例。warp 的 4 个线程**访问同一列**（例如访问矩阵的第 0 列）：

```
             col0   col1   col2   col3
             ----   ----   ----   ----
row 0:       [0]    [1]    [2]    [3]
row 1:       [4]    [5]    [6]    [7]
row 2:       [8]    [9]    [10]   [11]
row 3:       [12]   [13]   [14]   [15]

             4 个线程分别访问 [0]、[4]、[8]、[12]（第 0 列）
```

背景：硬件解码 bank 的规则永远是 `bank = phys_addr % 32`（取模）。这是一条固定规则，**不能**独立于地址去改变。所以改变 bank 的唯一方法是**改变访问的物理地址本身**。

Swizzling 的核心思路就是：**硬件在计算 bank 之前，先对地址做一次 XOR 变换**。也就是说：

```
no  swizzle: physical_addr = logical_addr                 → bank = logical_addr % N
with swizzle: physical_addr = XOR(logical_addr, params)   → bank = XOR(logical_addr) % N
```

这个 XOR 是硬件**在地址通路上自动完成**的，对软件透明。软件写入 `[4]`，硬件实际写到物理地址 `XOR(4)` 处；软件读 `[4]`，硬件从物理地址 `XOR(4)` 处读取。**数据在 SRAM 中的物理位置确实变了，但软件从头到尾只看得到逻辑地址 `[4]`。**

对比两种模式下同一个访问序列 `[0]`、`[4]`、`[8]`、`[12]` 的硬件行为：

| 模式 | 逻辑地址 | 硬件 XOR 变换 | 物理地址 | bank = phys_addr % 4 | 结果 |
|------|---------|--------------|---------|---------------------|------|
| no swizzle | [0] | 不变 → 0 | 0 | 0 % 4 = 0 | |
| no swizzle | [4] | 不变 → 4 | 4 | 4 % 4 = 0 | ← 冲突 |
| no swizzle | [8] | 不变 → 8 | 8 | 8 % 4 = 0 | ← 冲突 |
| no swizzle | [12] | 不变 → 12 | 12 | 12 % 4 = 0 | ← 冲突 |
| swizzle `x^(x>>2)` | [0] | 0^0=0 → 0 | 0 | 0 % 4 = 0 | |
| swizzle `x^(x>>2)` | [4] | 4^1=5 → 5 | 5 | 5 % 4 = 1 | ← 分散 |
| swizzle `x^(x>>2)` | [8] | 8^2=10 → 10 | 10 | 10 % 4 = 2 | ← 分散 |
| swizzle `x^(x>>2)` | [12] | 12^3=15 → 15 | 15 | 15 % 4 = 3 | ← 分散 |

关键理解：
- **bank 解码规则没变**，永远是 `% 4`
- 变的是 **地址本身**——swizzle 模式下的逻辑地址 `[4]` 被硬件 XOR 成了物理地址 5，所以落到了 bank 1
- 这个地址 XOR 变换对软件完全透明，编译器通过 `#ttg.swizzled_shared<{vec, perPhase, maxPhase, order}>` 属性指定 XOR 参数

注意：上面这个 XOR 规则只化解了 stride=4 的冲突。如果 warp 访问的是 stride=5 的地址序列，它会带来新的冲突。实际应用时，编译器会根据 kernel 的典型访问 stride 选择不同的 XOR 参数组合（通过 `vec`/`perPhase`/`maxPhase` 调优）。

#### Swizzling 参数详解

Swizzling 算法通过三个核心参数控制 XOR 映射规则，这三个参数共同决定了地址位如何被划分为 vector / bank / segment 三段：

| 参数 | 含义 | 硬件映射 |
|------|------|---------|
| `vec` | **单线程单次向量化 load/store 的元素个数** | 对应硬件向量化指令的单次访存宽度。上限 `128 / bitwidth`：f32 上限 4，f16 上限 8，i8 上限 16。`vec × bitwidth ≤ 128` bit（一个 cache line）。线程束内 32 个线程各 load `vec` 个元素，总吞吐为 `32 × vec × bitwidth` bit |
| `perPhase` | 每个 phase 包含的行数 | 对应一个 bank segment 覆盖的地址空间。`perPhase × vec × bitwidth = 1024` bits（32 banks × 32 bits） |
| `maxPhase` | phase 数量（XOR key 种类数） | 覆盖 tensor 剩余所有地址位。XOR key 循环使用，周期为 `maxPhase` |
| `order` | 维度的遍历顺序 | 指定 tensor 的哪个维度是连续的（行优先 `[1, 0]` 或列优先 `[0, 1]`） |

在 NVIDIA GPU 上，swizzling 通过 `swizzled_shared` 布局实现，编码为：

```
#ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
```

##### 参数如何控制 XOR 变换

三个参数组合起来决定地址的 XOR 变换规则：

```
物理地址 = 逻辑地址 XOR (vec × ((row / perPhase) % maxPhase))
```

展开每个参数的作用：

| 参数 | 控制 | 直观理解 |
|------|------|---------|
| `vec` | XOR 的**粒度** | 每 `vec` 个连续元素为一组整体 XOR。`vec=2` 表示元素成对交换位置 |
| `perPhase` | XOR key 切换的**周期** | 每 `perPhase` 行共享一个 XOR key。`perPhase=2` 表示 2 行使用相同 pattern |
| `maxPhase` | XOR key 的**种类数** | 最多 `maxPhase` 种不同 XOR key 循环使用，之后重复 |

**示例一：`vec=1, perPhase=1, maxPhase=4`（最细粒度）**

每行一个不同的 XOR pattern，遍历全部 4 行后重复：

```
Row 0:  [0, 1, 2, 3]     XOR with 0  (phase 0)
Row 1:  [4, 5, 6, 7]  →  [5, 4, 7, 6]   XOR with 1  (phase 1)
Row 2:  [8, 9,10,11]  →  [10,11,8,9]    XOR with 2  (phase 2)
Row 3:  [12,13,14,15] →  [15,14,13,12]  XOR with 3  (phase 3)
Row 4:  回到 phase 0 (重复)
```

**示例二：`vec=2, perPhase=2, maxPhase=2`（粗粒度）**

两个连续元素成对 XOR，每 2 行共享 key，共 2 种 key 循环：

```
Row 0-1:  [0,1, 2,3, 4,5, 6,7]     XOR with 0       (phase 0)
Row 2-3:  [8,9,10,11,12,13,14,15] → [10,11,8,9, 14,15,12,13]
                                    XOR with 2       (每对整体 XOR)
Row 4-5:  回到 phase 0 (重复)
```

#### Swizzling 参数的代码计算深度分析

三个参数并非硬编码，而是由编译器根据 src 和 dst 的寄存器/线程束布局**自动推导**出来的。核心实现在 `triton/lib/Tools/GenericSwizzling.cpp` 中。Triton 中有三套不同的计算路径，适用于不同场景。

##### 前置知识：LinearLayout 与基向量（Basis Vector）

在深入代码之前，需要理解 Triton 的 `LinearLayout` 如何表达数据布局。一个 layout 有多个**输入维度**（如 `register`、`lane`、`warp`）和一个**输出维度**（扁平化的 element 索引）。每个输入维度的每一位都是一个**基向量（basis）**，表示该位如何映射到输出地址位。

`flatten` 函数提取某一输入维度的基向量序列：

```cpp
// triton/lib/Tools/GenericSwizzling.cpp:41
SmallVector<int32_t> flatten(const LinearLayout &ll, StringAttr dim) {
  auto outDim = *ll.getOutDimNames().begin();
  SmallVector<int32_t> vec;
  for (int i = 0; i < ll.getInDimSizeLog2(dim); ++i)
    vec.push_back(ll.getBasis(dim, i, outDim));
  return vec;
}
```

例如 `flatten(src, "register")` 返回 `[1, 2, 4]`，表示 register 索引的第 0 位映射到输出地址的 bit 0，第 1 位映射到 bit 1，第 2 位映射到 bit 2。

```
输出地址位（从低位到高位）:
───────────────────────────────────────────
  bit 0  ← register 位 0  (基向量 1)
  bit 1  ← register 位 1  (基向量 2)
  bit 2  ← register 位 2  (基向量 4)
  bit 3  ← lane 位 0      (基向量 8)
  bit 4  ← lane 位 1      (基向量 16)
  ...
```

在 swizzling 计算中，关键的基向量有：

| `flatten` 调用 | 含义 |
|-------------|------|
| `flatten(src, "register")` | src layout 中，输出地址哪些位由寄存器索引决定 |
| `flatten(src, "lane")` | src layout 中，输出地址哪些位由线程束内 lane ID 决定 |
| `flatten(src, "warp")` | src layout 中，输出地址哪些位由 warp ID 决定 |

`intersectionBasis(regSrc, regDst, dim)` 找出两个 layout 的寄存器基向量的**交集**——即两个 layout 都使用相同寄存器位来索引的元素维度。这些维度可以被安全向量化。

```cpp
// triton/lib/Tools/GenericSwizzling.cpp:218
SmallVector<int32_t> intersectionBasis(ArrayRef<int32_t> b1,
                                       ArrayRef<int32_t> b2, int32_t dim) {
  if (llvm::all_of(b1, isPowerOf2) && llvm::all_of(b2, isPowerOf2)) {
    // 如果所有基向量都是 2 的幂，直接做集合交集
    SetVector<int32_t> set2(b2.begin(), b2.end());
    for (int32_t b : b1) {
      if (b != 0 && set2.contains(b))
        result.push_back(b);
    }
    return result;
  }
  // 否则用零空间（nullspace）计算
  auto ns1 = nullspaceBasis(b1, dim);
  auto ns2 = nullspaceBasis(b2, dim);
  return nullspaceBasis(concat(ns1, ns2), dim);
}
```

##### 路径一：`optimalSwizzlingLdSt` — 通用 ConvertLayoutOp

用于大多数通过 shared memory 做 layout 转换的场景。核心实现在 `triton/lib/Tools/GenericSwizzling.cpp:456-596`。

**调用链：**

```
getNumScratchElemsSwizzledCvt(srcTy, dstTy, bitwidth)
  → toLinearLayout(srcTy), toLinearLayout(dstTy)
  → actionRemoveBroadcastedRegs     (移除广播寄存器维度)
  → optimalSwizzlingLdSt(src, dst, bitwidth)
      → 返回 LinearLayout (包含 vector/bank/segment/reps 维度)
  → 提取 reps 维度大小 → 计算 scratch size
```

**Step 1 — 提取寄存器公共基向量（计算 `vec`）：**

```cpp
// triton/lib/Tools/GenericSwizzling.cpp:468
SmallVector<int32_t> vbasis = intersectionBasis(regSrc, regDst, dim);
// vec = 2^vbasis.size()
```

`vbasis` 即 src 和 dst 共享的寄存器基向量。但需要受两个硬件约束裁剪：

**约束 1：不超 128 位（一个 cache line）**

```cpp
auto maxVecBases = llvm::Log2_32(128 / bitwidth);   // e.g. f16 → 3 (vec≤8)
if (vbasis.size() > maxVecBases)
    vbasis.resize(maxVecBases);
```

**约束 2：至少 32 位（一个 bank word）**

当 `vec * bitwidth < 32` 时，一次向量访问连一个 bank 都填不满，需要从 warp 级布局借基向量来补足：

```cpp
if ((1 << vbasis.size()) * bitwidth < 32) {
    auto basesPerBank = llvm::Log2_32(32 / bitwidth);
    // 尝试从 regSrc ∩ warpDst 中补充...
    auto regSrcWarp = intersectionBasis(removeVec(regSrc), warpDst, dim);
    auto regDstWarp = intersectionBasis(removeVec(regDst), warpSrc, dim);
    // 选择提供最大向量化的方向
    vbasis.append(largest.begin(), largest.end());
    // 还不够就补 warpSrc ∩ warpDst 的公共基向量
    if (vbasis.size() < basesPerBank) {
        auto warpSrcWarp = intersectionBasis(warpSrc, warpDst, dim);
        vbasis.append(warpSrcWarp.begin(), warpSrcWarp.end());
    }
    // 再不够就暴力补（至少在一个方向消除 conflict）
    while (vbasis.size() < basesPerBank && i < max(warpSrc.size(), warpDst.size()))
        vbasis.push_back(warpSrc[i] 或 warpDst[i]);  // 交替取
    // 裁剪到 basesPerBank（防止过度向量化导致不对称 bank conflict）
    if (vbasis.size() > basesPerBank)
        vbasis.resize(basesPerBank);
}
```

**填充举例：**

```
假设 src = blocked layout (reg=[1,2]), dst = mma_layout (reg=[4,8]), bitwidth=32

intersectionBasis([1,2], [4,8]) = {} → vbasis 空 → vec=1

检查下界: 1*32 = 32 ≥ 32, ok
→ vec = 1 (无法向量化，因为 src 和 dst 用完全不同的寄存器集)
```

```
假设 src = blocked layout (reg=[1,2,4,8]), dst = swizzled_shared (reg=[1,2,4]), bitwidth=16

intersectionBasis([1,2,4,8], [1,2,4]) = [1,2,4] → vbasis.size()=3 → vec=8

检查上界: maxVecBases = log2(128/16) = 3, ok (vec=8 不需要裁剪)
检查下界: 8*16 = 128 ≥ 32, ok (不需要填充)
→ vec = 8
```

**Step 2 — 计算 bank 段（计算 `perPhase`）：**

确定 `vbasis` 后，下一步是计算 bank 段的位数：

```cpp
// triton/lib/Tools/GenericSwizzling.cpp:422-426
constexpr int32_t bankBits = 32 * 32;  // 32 banks × 32 bits = 1024 bits
const int32_t lenBbasis = std::min<int32_t>(
    llvm::Log2_32(bankBits / ((1 << vbasis.size()) * bitwidth)),
    dim - vbasis.size());
```

公式推导：

```
bankBits / (vec * bitwidth)  =  1024 / (vec * bitwidth)  =  perPhase (行数)
lenBbasis = log2(perPhase)
```

例子：

```
bitwidth=16, vec=8
→ perPhase = 1024 / (8 * 16) = 8 行
→ lenBbasis = log2(8) = 3
```

**Step 3 — 计算 segment（计算 `maxPhase`）：**

segment 占剩余的地址位：

```cpp
// 所有剩下的地址位分配给 segment
const int32_t lenSbasis = dim - lenBbasis - vbasis.size();
// maxPhase = 2^lenSbasis
```

具体的 segment 基向量通过 `computeSegment` 函数计算，它考虑了 src/dst bank 专用基向量的差异，优先选择冲突最少的位：

```cpp
// triton/lib/Tools/GenericSwizzling.cpp:144
SmallVector<int32_t> computeSegment(..., int32_t lenSegment) {
  // 1. 优先取既不在 bankSrc 也不在 bankDst 中的位（无冲突位）
  // 2. 不够时取 bankSrc 和 bankDst 的差集
  //    - 差集 A 与 B 的 XOR (A^B) 放在前面 → conflict-free
  //    - 差集 A/B 原样放在后面 → 有冲突
  // 3. 裁剪到 lenSegment
}
```

实际 bank 基向量（`bbasis`，决定 `perPhase`）是 `vbasis ∪ sbasis` 的**补集**：

```cpp
SmallVector<int32_t> unionBasis;
unionBasis.append(vbasis.begin(), vbasis.end());
unionBasis.append(sbasis.begin(), sbasis.end());
SmallVector<int32_t> bbasis = complementBasis(unionBasis, dim);
// perPhase = 2^bbasis.size()
```

**Step 4 — 构建最终 LinearLayout 与 `buildReps`：**

三个基向量集合组合为三维的 LinearLayout：

```cpp
LinearLayout basis1D({
    {"vector", unflatten(vbasis)},   // vec = 2^vbasis.size()
    {"bank",   unflatten(bbasis)},   // perPhase = 2^bbasis.size()
    {"segment", unflatten(sbasis)},  // maxPhase = 2^sbasis.size()
}, src.getOutDims());
```

然后通过 `buildReps` 拆分 `reps` 维度：

```cpp
// triton/lib/Tools/GenericSwizzling.cpp:102
// 一个基向量成为 "rep" 的条件：
// 1. 在 src 和 dst 的寄存器维度中都存在
// 2. 在 smem 的 segment 维度中
// 这些基向量表示需要多次迭代加载/存储的数据
```

`reps` 维度被独立出来，final scratch buffer 大小为 `total_elements / reps`。

**完整的地址位划分图：**

```
总地址位数 dim = log2(tensor 总 element 数)

地址位分解（从低位到高位）：
┌──────────────────────────────────────────────────────────┐
│  [vector bits]  [bank/perPhase bits]  [segment/maxPhase bits]  │
│  vec = 2^v      perPhase = 2^b       maxPhase = 2^s           │
│  v + b + s = dim                                             │
└──────────────────────────────────────────────────────────┘

bitwidth=16, vec=8 (v=3), perPhase=8 (b=3):   →  s = dim - 6
bitwidth=32, vec=2 (v=1), perPhase=16 (b=4):  →  s = dim - 5
```

##### 路径二：`optimalSwizzling` — 多指令版本（NVIDIA 专用）

当传入 `getSrcDstTiles` 提供的多组指令 tile（ldmatrix、stmatrix、ldmatrix.trans 等）时，Triton 使用入口函数 `optimalSwizzling(src, dst, srcTiles, dstTiles, bitwidth)`。这是 NVIDIA 路径的核心。

**调用链（NVIDIA 路径）：**

```
getNvidiaAllocationAnalysisScratchSizeFn(targetInfo)
  → getNumScratchElemsSwizzledCvt(srcTy, dstTy, targetInfo)
    → toLinearLayout + actionRemoveBroadcastedRegs
    → getSrcDstTiles(targetInfo, bitwidth)    ← 获取架构支持的指令 tile 列表
    → optimalSwizzling(src, dst, srcTiles, dstTiles, bitwidth)
        → 返回 {LinearLayout, (readConflicts, writeConflicts)}
    → 计算 scratch size
```

`optimalSwizzling` 比 `optimalSwizzlingLdSt` 多了一个**指令选择**步骤：

**Step 1 — 枚举可行的指令对：**

```cpp
// triton/lib/Tools/GenericSwizzling.cpp:615-625
for (auto [idxSrc, instrSrc] : enumerate(srcTiles)) {
    for (auto [idxDst, instrDst] : enumerate(dstTiles)) {
        auto maybeTile = optimalSwizzlingTile(
            srcFlat, dstFlat, logRegSrc, logRegDst,
            instrSrc.laneContig, instrDst.laneContig);
        if (maybeTile.has_value())
            instr.push_back({{idxSrc, idxDst}, *maybeTile});
    }
}
```

`optimalSwizzlingTile` 检查一对指令（如 ld.shared → stmatrix）是否兼容——即它们的寄存器映射和 lane 映射能否匹配。只有兼容的对才保留。

可能的指令对举例（NVIDIA GPU）：

| 架构 | 源端指令 | 目标端指令 |
|------|---------|-----------|
| 所有 (sm_70+) | `ld.shared.b32.v4` | `st.shared.b32.v4` |
| sm_75+ (Turing+) | `ldmatrix.v4` | `st.shared.b32.v4` |
| sm_75+ (Turing+) | `ld.shared.b32.v4` | `stmatrix.v4` |
| sm_90+ (Hopper+) | `ldmatrix.v4` | `stmatrix.v4` |

**Step 2 — 为每对指令计算 swizzling 参数：**

```cpp
for (auto [instrs, vbasis, tileSrc, tileDst, leaveReps] : tiles) {
    auto smem = optimalSwizzling(srcFlat, dstFlat, bitwidth,
                                  vbasis, tileSrc, tileDst,
                                  src.getOutDims(), leaveReps);
    auto [read, write] = bankConflicts(tileSrc, tileDst, smem);
    smems.push_back({read + write, smem, {instrs.first, instrs.second}});
}
```

这里调用的 `optimalSwizzling`（带 tile 参数的核心函数，同文件第 388 行）与 `optimalSwizzlingLdSt` 不同：
- **`vbasis`** 不再从 `intersectionBasis` 推导，而是从 `optimalSwizzlingTile` 直接获得（受指令对约束）
- **`tileSrc`/`tileDst`** 包含了特定指令的 lane 约束（如 ldmatrix 的 `laneContig={0,1}` 连续 lane + `laneAddr={2,3,4}` 地址 lane）
- `bankBits` 约束（`lenBbasis`）仍然适用

**Step 3 — 选择 bank conflict 最少的组合：**

```cpp
// triton/lib/Tools/GenericSwizzling.cpp:700-709
// 启发式：最小化总 bank conflict
// 平局时选择 reps 更少的（减少迭代次数）
auto it = llvm::min_element(smems, [](const auto &a, const auto &b) {
    return get<0>(a) < get<0>(b) ||
           (get<0>(a) == get<0>(b) &&
            get<1>(a).getInDimSize(kReps) > get<1>(b).getInDimSize(kReps));
});
return {get<1>(*it), get<2>(*it)};
```

如果没有任何指令对兼容（`tiles.empty()`），则回退到 `optimalSwizzlingLdSt`。

**路径一与路径二的关系：**

```
optimalSwizzling（路径二）                    optimalSwizzlingLdSt（路径一）
────────────────────                         ────────────────────
输入: srcLayout, dstLayout,                 输入: srcLayout, dstLayout,
     srcTiles, dstTiles, bitwidth                 bitwidth

vbasis 来源: optimalSwizzlingTile            vbasis 来源: intersectionBasis
            (指令对专用)                               (通用)

tile 来源: 调用者传入                        tile 来源: 从 lane 基向量自动推导
           (ldmatrix/stmatrix 约束)                    (仅 ld/st.shared)

bankConflicts: 来自 tileSrc/tileDst           bankConflictsLdSt: 从 lane 推导

用途: NVIDIA ConvertLayoutOp                 用途: Generic ConvertLayoutOp
```

##### 路径三：`NVMMASharedEncodingAttr` — Hopper MMA 专用

对于 Hopper 架构的 MMA（矩阵乘累加）操作数，参数通过 `swizzlingByteWidth` 反推，是最直观的计算方式：

```cpp
int getVec()       { return 128 / elementBitWidth; }
int getPerPhase()  { return 128 / swizzlingByteWidth; }
int getMaxPhase()  { return swizzlingByteWidth / 16; }
```

`swizzlingByteWidth` 由连续维度大小决定：

```cpp
auto contigSize = shapePerCTA[order[0]] * eleBitWidth / 8;
if      (contigSize >= 128)  swizzlingByteWidth = 128;  // vec=8(f16), perPhase=1, maxPhase=8
else if (contigSize >= 64)   swizzlingByteWidth = 64;   // vec=8(f16), perPhase=2, maxPhase=4
else if (contigSize >= 32)   swizzlingByteWidth = 32;   // vec=8(f16), perPhase=4, maxPhase=2
else                         swizzlingByteWidth = 0;    // 不 swizzle
```

##### `bankConflicts`：评估 swizzling 质量的指标

`bankConflicts` 是选择最优 swizzling 参数的核心评估函数：

```cpp
// triton/lib/Tools/GenericSwizzling.cpp:243
std::pair<int, int> bankConflicts(ArrayRef<int32_t> tileSrc,
                                  ArrayRef<int32_t> tileDst,
                                  const LinearLayout &smem) {
    auto segmentBases = flatten(smemFlat, "segment");
    int write = 1 << intersectionBasis(segmentBases, tileSrc, rank).size();
    int read  = 1 << intersectionBasis(segmentBases, tileDst, rank).size();
    return {read - 1, write - 1};  // 减 1 是因为至少 1-way（无冲突）
}
```

核心逻辑：计算 src/dst 的 tile 基向量与 smem segment 基向量的交集大小。交集越大 → 该指令的访问模式与 swizzling 的 segment 维度重叠越多 → bank conflict 越严重。

```
例子：segmentBases = [8, 16], tileSrc = [8, 32]
→ intersectionBasis 交集 = {8}，大小为 1
→ write bank conflicts = 2^1 - 1 = 1（即 2-way bank conflict）
```

`optimalSwizzling` 遍历所有可能的指令对，选择 `read + write` 最小的组合。

#### 参数选择的权衡与局限

| 调整方向 | 好处 | 代价 |
|---------|------|------|
| **增大 `vec`** | 更高的向量化效率，内存吞吐更大 | XOR 粒度变粗，无法消除细粒度的 bank conflict；可能需要在 ld/st 指令中做 PRMT 额外重排 |
| **增大 `perPhase`** | 更多行共享同一个 XOR pattern，swizzle 更稳定 | 减少了 bank conflict 消除的粒度范围，对特定 stride 敏感的模式无效 |
| **增大 `maxPhase`** | 更多的 XOR key 可覆盖更大的地址范围 | 引入更多 phase 边界，可能产生新的边界冲突 |

当 `vec * bitwidth` 远大于 32 位时，swizzle 可以更灵活；当 src 和 dst 的寄存器布局完全不一致（`vbasis` 空）时，`vec=1` 退化为无向量化，swizzle 效果受限。编译器通过 `bankConflicts` 函数自动在这些维度间寻找最佳平衡。

### 4.2 Padding（填充）

Padding 是另一种减少 bank conflict 的策略，通过在每个行的末尾添加额外的 padding 元素来实现：

```text
// Without padding:
行 0: [col0] [col1] [col2] ... [colN-1]
行 1: [col0] [col1] [col2] ... [colN-1]
                              ↑ col0 和 colN-1 在同一 bank

// With padding:
行 0: [col0] [col1] [col2] ... [colN-1] [pad]
行 1: [col0] [col1] [col2] ... [colN-1] [pad]
                                          ↑ padding 使行间偏移错开
```

NVIDIA GPU 始终使用 swizzling。

## 五、为什么 NVIDIA 需要单独处理？

### 5.1 核心原因

NVIDIA 需要独立的 `allocate-shared-memory-nv` pass，主要有以下三个原因：

**① TargetInfo 依赖的 swizzling tile 大小**

不同的 NVIDIA GPU 架构（Ampere、Hopper、Blackwell）具有不同的 shared memory 大小、bank 数量和 swizzling 能力。`TargetInfo` 封装了这些架构差异，通过 `computeCapability` 和 `ptxVersion` 参数动态调整 swizzling 的 tile 尺寸。Generic pass 无法知道这些架构细节。

**② 始终使用 swizzling**

NVIDIA 的 CUDA GPU 始终 prefer swizzling 来优化 shared memory 访问（注释明确写着 "In cuda we always swizzle"）。虽然 Generic 实现也支持 swizzling，但 NVIDIA 版本使用了更积极的优化策略（比如 `actionRemoveBroadcastedRegs` 预处理）。

**③ 与 Pipeline 中其他 NVIDIA 特定 pass 的交互**

NVIDIA 的 pipeline 中包含 `allocate-tensor-memory`、`warp-specialize-to-llvm` 等 NVIDIA 特有的 pass，这些 pass 与 shared memory 分配存在交互（例如 tensor memory 的分配会影响 shared memory 的布局）。使用独立的 pass 可以更好地管理这些依赖关系。

### 5.2 Pipeline 对照

```python
# NVIDIA Pipeline
nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)  ← NV 特定
nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)                            ← NV 特定
nvidia.passes.ttnvgpuir.add_check_matmul_two_cta(pm)                              ← NV 特定
nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)                  ← NV 特定
nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)                                     ← NV 特定
nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)                           ← NV 特定
```

### 5.3 NVIDIA 处理 `ConvertLayoutOp` 的 scratch size 细节

在 NVIDIA 的 `getNumScratchElemsSwizzledCvt` 中：
1. 首先对 src/dst layout 执行 `actionRemoveBroadcastedRegs` — 移除广播的寄存器维度，简化 swizzling 分析
2. 通过 `getSrcDstTiles(targetInfo, bitwidth)` 获取架构特定的 tile 大小 — 这取决于 compute capability
3. 调用 `optimalSwizzling` 找到最优的 swizzling 参数
4. 计算最终的 scratch buffer 大小

```text
ConvertLayoutOp (NVIDIA) Scratch Size 计算流程：

src layout ──→ removeBroadcastedRegs ──┐
                                       ├──→ optimalSwizzling ──→ scratch size
dst layout ──→ removeBroadcastedRegs ──┘         ↑
                                               targetInfo
                                            (computeCapability,
                                             ptxVersion)
```

---

## 六、`attachAllocationSizeAndOffsetAttr` 的通用流程

两个实现最终都调用同一个辅助函数 `attachAllocationSizeAndOffsetAttr` 来将分配结果写回 IR：

```cpp
void attachAllocationSizeAndOffsetAttr(ModuleOp mod,
                                       ModuleAllocation &allocation) {
  // 遍历 module 中所有函数
  mod.walk<WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
    auto *funcAllocation = allocation.getFuncData(funcOp);
    funcOp.walk([&](Operation *op) {
      int offset = -1;
      // 查找操作级别的 buffer ID
      auto oBufferId = funcAllocation->getBufferId(op);
      if (oBufferId != Allocation::InvalidBufferId)
        offset = funcAllocation->getOffset(oBufferId);
      // 查找结果值级别的 buffer ID
      else if (op->getNumResults() == 1) {
        Value value = op->getResult(0);
        auto vBufferId = funcAllocation->getBufferId(value);
        if (vBufferId != Allocation::InvalidBufferId)
          offset = funcAllocation->getOffset(vBufferId);
      }
      if (offset == -1) return;
      op->setAttr("allocation.offset", IntegerAttr::get(... , offset));
    });
    return WalkResult::skip();
  });
  // 设置 module 级别的总 shared memory 大小
  mod->setAttr("ttg.shared", IntegerAttr::get(... , allocation.getSharedMemorySize()));
}
```

此函数执行以下操作：
1. **遍历 module 中所有函数**（PreOrder 确保父函数在子函数之前处理）
2. **对每个操作，查找其 buffer ID** — 先查操作级别的分配，再查结果值级别的分配
3. **设置 `allocation.offset` 属性** — 记录该操作在 shared memory 中的偏移量
4. **设置 `ttg.shared` 属性** — 记录整个模块所需的 shared memory 总大小

---

## 七、测试文件分析

### 7.1 Generic 测试

文件：`triton/test/Conversion/allocate_shared_memory.mlir`

```mlir
// RUN: triton-opt %s --allocate-shared-memory | FileCheck %s

// CHECK-SAME: ttg.shared = 131072 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @gather_op(%arg0: tensor<1024x256xi32, #blocked>, ...) {
    // CHECK-NEXT: allocation.offset = 0 : i32
    %0 = tt.gather %arg1[%arg0] ...
    tt.return
  }
}
```



### 7.2 Analysis 级别的测试

文件：`triton/test/Analysis/test-allocation.mlir`

使用 `test-print-allocation` 测试 pass 验证详细的内存分配结果，包括：
- 空函数、循环中的分配、可重用性检查
- 图着色（多轮、多颜色）
- Warp specialization 中的分配
- 各种布局（swizzled_shared、padded_shared、nvmma_shared 等）的对齐计算

---

## 八、关键设计要点

1. **全局分析**：本 pass 在 `ModuleOp` 级别对整个模块进行全局的 shared memory 分配分析，而非逐函数独立分配。这允许跨函数（通过调用图）的内存复用。

2. **图着色算法**：使用寄存器分配风格的图着色算法来最小化 shared memory 总使用量。两个活跃范围不重叠的操作可以共享同一块内存。

3. **架构感知的 swizzling**：NVIDIA 版本通过 `TargetInfo` 将架构特性（compute capability）引入 swizzling tile 大小计算，确保生成的 shared memory 访问模式与具体 GPU 架构匹配。

4. **Backend 可扩展性**：通过 `AllocationAnalysisScratchSizeFn` 回调机制，各 backend 可以定制 scratch buffer 的大小计算逻辑，而无需修改 `ModuleAllocation` 核心算法。

5. **统一属性接口**：所有 backend 最终通过 `attachAllocationSizeAndOffsetAttr` 以统一的方式将 `allocation.offset` 和 `ttg.shared` 属性写入 IR，确保下游 pass 无需关心来自哪个 backend。

---

## 九、`defaultAllocationAnalysisScratchSizeFn` 与 `getNvidiaAllocationAnalysisScratchSizeFn` 深度对比

这是本 pass 最核心的区别所在。两个函数虽然目标相同（计算每个操作所需的 shared memory 大小），但实现策略和对硬件特性的利用程度截然不同。

### 9.1 `defaultAllocationAnalysisScratchSizeFn` — 通用版本

**文件位置**：`triton/lib/Analysis/Allocation.cpp:69-112`

这是 `ModuleAllocation` 构造函数默认使用的回调函数，由所有 backend 共享。

#### 处理的 Operation 类型

```cpp
unsigned defaultAllocationAnalysisScratchSizeFn(Operation *op) {
  if (auto reduceOp = dyn_cast<ReduceOp>(op))
    return ReduceOpHelper(reduceOp).getScratchSizeInBytes();
  if (auto scanOp = dyn_cast<ScanOp>(op))
    return ScanLoweringHelper(scanOp).getScratchSizeInBytes();
  if (auto gatherOp = dyn_cast<GatherOp>(op))
    return GatherLoweringHelper(gatherOp).getScratchSizeInBytes();
  if (auto histogram = dyn_cast<HistogramOp>(op))
    return max(dstTy.getNumElements(), threadsPerWarp) * bitwidth / 8;
  if (auto cvtLayout = dyn_cast<ConvertLayoutOp>(op)) {
    if (!cvtNeedsSharedMemory(srcTy, dstTy)) return 0;
    auto elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy);
    return elems * getBitwidth(srcTy) / 8;
  }
  if (isa<AtomicRMWOp, AtomicCASOp>(op)) {
    auto smemShape = getRepShapeForAtomic(op->getResult(0));
    auto elems = getNumScratchElements(smemShape);
    if (elems == 0) return 0;
    return elems * elemBw / 8;
  }
  if (isa<TensormapCreateOp>(op))
    return 128;  // TMA descriptor
  return 0;
}
```

| Operation | 计算方式 |
|-----------|---------|
| `ReduceOp` | `ReduceOpHelper::getScratchSizeInBytes()` — 跨 warp 归约需要的暂存空间 |
| `ScanOp` | `ScanLoweringHelper::getScratchSizeInBytes()` — 扫描操作需要的暂存空间 |
| `GatherOp` | `GatherLoweringHelper::getScratchSizeInBytes()` — gather 操作需要的暂存空间 |
| `HistogramOp` | `max(numElements, threadsPerWarp) * bitwidth / 8` — 直方图操作，每个线程至少一个 slot |
| `ConvertLayoutOp` | `getNumScratchElemsSwizzledCvt(srcTy, dstTy) * bitwidth / 8` — 布局转换的 swizzled scratch |
| `AtomicRMWOp` / `AtomicCASOp` | 根据结果 tensor 的广播模式计算 rep shape，再乘以元素大小 |
| `TensormapCreateOp` | 固定 128 字节（TMA descriptor 大小） |
| 其他 | 0（不需要 shared memory） |

#### 对 `ConvertLayoutOp` 的处理

Generic 版本调用的是**不带 `targetInfo` 的** `getNumScratchElemsSwizzledCvt`：

```cpp
// triton/lib/Analysis/Allocation.cpp:32
unsigned getNumScratchElemsSwizzledCvt(RankedTensorType srcTy,
                                       RankedTensorType dstTy) {
  auto srcLayout = gpu::toLinearLayout(srcTy);
  auto dstLayout = gpu::toLinearLayout(dstTy);
  srcLayout = actionRemoveBroadcastedRegs(srcLayout).apply(srcLayout);
  dstLayout = actionRemoveBroadcastedRegs(dstLayout).apply(dstLayout);
  int bitwidth = getBitwidth(srcTy);
  auto smem = gpu::optimalSwizzlingLdSt(srcLayout, dstLayout, bitwidth);
  auto reps = smem.getInDimSize(StringAttr::get(ctx, "reps"));
  return smem.getTotalOutDimSize() / reps;
}
```

关键点：使用 `optimalSwizzlingLdSt` — 这是为 **load/store 指令**优化的特定 swizzling 函数，**不包含 ldmatrix/stmatrix** 等特殊指令产生的额外 tile 约束。

### 9.2 `getNvidiaAllocationAnalysisScratchSizeFn` — NVIDIA 版本

**文件位置**：`triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp:61-76`

这是一个**工厂函数**，接受 `TargetInfoBase &` 参数，返回一个 lambda 作为回调：

```cpp
std::function<unsigned(Operation *)>
getNvidiaAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
    if (auto cvtOp = dyn_cast<ConvertLayoutOp>(op)) {
      auto srcTy = cvtOp.getSrc().getType();
      auto dstTy = cvtOp.getType();
      if (!cvtNeedsSharedMemory(srcTy, dstTy)) return 0;
      // "In cuda we always swizzle"
      auto elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy, targetInfo);
      return elems * getBitwidth(srcTy) / 8;
    }
    return defaultAllocationAnalysisScratchSizeFn(op);
  };
  return allocation;
}
```

核心区别：

| 维度 | Generic | NVIDIA |
|------|---------|--------|
| 函数形式 | 普通函数，无参数 | 工厂函数，返回 lambda，携帶 `targetInfo` |
| ConvertLayoutOp | 直接用自己的逻辑 | 用自己的逻辑（见下） |
| 非 ConvertLayoutOp | 不处理（由调用者负责） | **回退到 `defaultAllocationAnalysisScratchSizeFn`** |
| 是否含 `ReduceOp`/`ScanOp` 等 | ✅ 完整实现 | ✅ 通过回退包含 |

#### 对 `ConvertLayoutOp` 的处理 — NVIDIA 版本

NVIDIA 在同一个文件中定义了自己的 `getNumScratchElemsSwizzledCvt`（与 Generic 的同名但不同）：

```cpp
// triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp:45
static unsigned getNumScratchElemsSwizzledCvt(RankedTensorType srcTy,
                                              RankedTensorType dstTy,
                                              TargetInfoBase &targetInfo) {
  auto srcLayout = triton::gpu::toLinearLayout(srcTy);
  auto dstLayout = triton::gpu::toLinearLayout(dstTy);
  srcLayout = actionRemoveBroadcastedRegs(srcLayout).apply(srcLayout);
  dstLayout = actionRemoveBroadcastedRegs(dstLayout).apply(dstLayout);
  int bitwidth = getBitwidth(srcTy);
  auto [srcTiles, dstTiles] = gpu::getSrcDstTiles(targetInfo, bitwidth);
  auto [smem, _] = triton::gpu::optimalSwizzling(srcLayout, dstLayout,
                                                   srcTiles, dstTiles, bitwidth);
  auto reps = smem.getInDimSize(StringAttr::get(ctx, "reps"));
  return smem.getTotalOutDimSize() / reps;
}
```

差异总结：

```
                          Generic                        NVIDIA
                         ────────                      ──────
Swizzling 函数:    optimalSwizzlingLdSt          optimalSwizzling (通用版)
                   (仅 load/store tile)          (支持任意 tile 集)

TargetInfo:        无 targetInfo                 传入 targetInfo

Tile 来源:         内置 tile 规则                 getSrcDstTiles(targetInfo, bw)
                                                  ├─ ld/st.shared (all arch)
                                                  ├─ ldmatrix (cc≥75, 32-bit)
                                                  ├─ stmatrix (cc≥90, 32-bit)
                                                  ├─ ldmatrix.trans (cc≥75, 16-bit)
                                                  └─ stmatrix.trans (cc≥90, 16-bit)
```

### 9.3 `getSrcDstTiles` 如何利用 `TargetInfo`

这是 NVIDIA 版本获得架构感知能力的关键函数。它的实现体现了 NVIDIA GPU 不同代际的指令集演进：

```cpp
// triton/lib/Conversion/TritonGPUToLLVM/Utility.cpp:42
getSrcDstTiles(const TargetInfoBase &targetInfo, int bitwidth) {
  SmallVector<LocalMemOpTile> src, dst;

  // 第 1 层：所有架构都支持的 ld/st.shared
  auto ldstshared = LocalMemOpTile{{}, {0, 1, 2}};
  src.push_back(ldstshared);
  dst.push_back(ldstshared);

  // 第 2 层：ldmatrix/stmatrix (sm_75+ / sm_90+)
  if (targetInfo.supportLdMatrix() || targetInfo.supportStMatrix()) {
    if (bitwidth <= 32) {
      // ldmatrix:  从 smem 加载 32-bit 元素到寄存器
      // stmatrix:  从寄存器存储 32-bit 元素到 smem
      auto ldstmatrix = LocalMemOpTile{{0, 1}, {2, 3, 4}};
      if (targetInfo.supportStMatrix()) src.push_back(ldstmatrix);
      if (targetInfo.supportLdMatrix()) dst.push_back(ldstmatrix);
    }
    if (bitwidth == 16) {
      // ldmatrix.trans:  加载并转置 16-bit 元素
      auto ldstmatrixtrans = LocalMemOpTile{{2, 3, 4}, {0, 1}};
      if (targetInfo.supportStMatrix()) src.push_back(ldstmatrixtrans);
      if (targetInfo.supportLdMatrix()) dst.push_back(ldstmatrixtrans);
    }
  }
  return {src, dst};
}
```

指令集演进的依赖关系：

| 架构 | CC | `supportLdMatrix` | `supportStMatrix` | 新增的 tile |
|------|----|-------------------|-------------------|-------------|
| Volta | sm_70 | ❌ | ❌ | (仅 ld/st.shared) |
| Turing | sm_75 | ✅ | ❌ | ldmatrix, ldmatrix.trans |
| Ampere | sm_80 | ✅ | ❌ | ldmatrix, ldmatrix.trans |
| Ada Lovelace | sm_89 | ✅ | ❌ | ldmatrix, ldmatrix.trans |
| Hopper | sm_90 | ✅ | ✅ | + stmatrix, stmatrix.trans |
| Blackwell | sm_100 | ✅ | ✅ | (同上，额外 B8 支持) |

`optimalSwizzling` 接收这些 tile 描述后，会寻找一个同时满足所有 tile 约束的 shared memory 布局——即每个 tile 都能高效访问（无 bank conflict 或最小化冲突）的 swizzling 参数。

### 9.4 `optimalSwizzlingLdSt` vs `optimalSwizzling` 的区别

此对比已在[第四章 Swizzling 参数的代码计算深度分析](#swizzling-参数的代码计算深度分析)中详细展开，包括完整的调用链、代码实现和 `bankConflicts` 评估机制。这里仅做概要总结：

- `optimalSwizzlingLdSt`：从 src/dst 的寄存器公共基向量推导 `vbasis`，适用于仅使用 `ld/st.shared` 的通用场景
- `optimalSwizzling`（带 tile 版本）：从 `optimalSwizzlingTile` 获取指令对专用的 `vbasis`，遍历多组指令组合（ldmatrix/stmatrix 等），用 `bankConflicts` 选择最优解，是 NVIDIA 路径的核心

### 9.5 为何 NVIDIA 不使用 Generic 版本而必须覆盖？

直接原因：**NVIDIA 必须利用 ldmatrix/stmatrix 指令来获得最优性能**，但这需要：
1. 传入 `targetInfo` 来查询架构是否支持这些指令
2. 将架构特定的 tile 约束传递给 `optimalSwizzling`
3. 在 scratch size 计算中考虑这些特殊指令的数据布局要求

Generic 版本的 `defaultAllocationAnalysisScratchSizeFn` 做不到以上任何一点。

如果 NVIDIA 使用 Generic 版本：
```
Generic 计算的 scratch size:  8192 字节 (仅考虑 ld/st.shared)
实际最优 scratch size:       12288 字节 (包含 ldmatrix tile 对齐)
                              ↑ 差异在于 tile 对齐要求不同
```
轻则性能次优（bank conflict 未被消除），重则代码生成错误（ldmatrix 的 tile 约束未被满足）。

---

*本文档对应源代码：*
- *Generic: [AllocateSharedMemory.cpp](../../triton/lib/Conversion/TritonGPUToLLVM/AllocateSharedMemory.cpp)*
- *NVIDIA: [Allocation.cpp](../../triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp)*
- *Analysis: [Allocation.h](../../triton/include/triton/Analysis/Allocation.h), [Allocation.cpp](../../triton/lib/Analysis/Allocation.cpp)*
