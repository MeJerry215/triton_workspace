# `allocate-shared-memory` Pass 分析

```python
# NVIDIA backend
nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)

# Generic
passes.ttgpuir.add_allocate_shared_memory(pm)
```

---

## 1. 概述

### 1.1 Pass 属性

| 属性 | 值 |
|------|-----|
| **Pass 名称（Generic）** | `allocate-shared-memory` |
| **Pass 名称（NVIDIA）** | `allocate-shared-memory-nv` |
| **操作粒度** | `mlir::ModuleOp` |
| **核心依赖** | `ModuleAllocation` 分析框架 |

### 1.2 输出

| 属性 | 位置 | 含义 |
|------|------|------|
| `allocation.offset` | 需要 shared memory 的 op 上 | 该 op 在 shared memory 中的起始偏移（字节） |
| `ttg.shared` | module 上 | 整个模块所需 shared memory 总大小（字节） |

下游 LLVM 代码生成用这两个属性计算 shared memory 基址和偏移。

### 1.3 Pipeline 位置

```
make_llir pipeline (NVIDIA):
  ...
  gluon-inliner
  allocate-shared-memory / allocate-shared-memory-nv   ← 本 pass
  tritongpu-to-llvmir
  nvgpu-to-llvm / warp-specialize-to-llvm
  canonicalizer + cse + symbol-dce
```

放在内联之后、LLVM 转换之前，确保：
1. 函数已内联，liveness 在单一函数内完成
2. `convert_layout`、`reduce` 等 op 已确定，scratch 需求可准确计算
3. offset/size 属性可供后续 lowering 使用

---

## 2. Pass 入口：Generic 与 NVIDIA

`ModuleAllocation` 类只有一个实现，**Generic 和 NVIDIA 的差异完全在调用方传入的 `scratchSizeGetter`**。

| | Generic | NVIDIA |
|---|---------|--------|
| **文件** | `AllocateSharedMemory.cpp` | `Allocation.cpp`（nvidia） |
| **Pass** | `allocate-shared-memory` | `allocate-shared-memory-nv` |
| **构造** | `ModuleAllocation(mod)` | `ModuleAllocation(mod, getNvidiaAllocationAnalysisScratchSizeFn(targetInfo))` |
| **scratchSizeGetter** | `defaultAllocationAnalysisScratchSizeFn` | 捕获 `TargetInfo` 的 lambda |
| **额外参数** | 无 | `computeCapability` + `ptxVersion` |

```cpp
// Generic — triton/lib/Conversion/TritonGPUToLLVM/AllocateSharedMemory.cpp
void runOnOperation() override {
  ModuleOp mod = getOperation();
  ModuleAllocation allocation(mod);
  attachAllocationSizeAndOffsetAttr(mod, allocation);
}

// NVIDIA — triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp
void runOnOperation() override {
  ModuleOp mod = getOperation();
  TargetInfo targetInfo(computeCapability, ptxVersion);
  ModuleAllocation allocation(
      mod, getNvidiaAllocationAnalysisScratchSizeFn(targetInfo));
  attachAllocationSizeAndOffsetAttr(mod, allocation);
}
```

**唯一实质差异**：NVIDIA 版对 `ConvertLayoutOp` 用架构感知的 `optimalSwizzling`（含 ldmatrix/stmatrix tile），其余 op 回退到 default；liveness、图着色、`attachAllocationSizeAndOffsetAttr` 完全相同。

---

## 3. ModuleAllocation 分析框架

实现位于 `triton/lib/Analysis/Allocation.cpp`，头文件 `triton/include/triton/Analysis/Allocation.h`。

本节按「问题 → 数据结构 → 构造函数那几行费解的代码 → 调用链 → 分析三步」的顺序展开。读者最常卡住的地方是 `try_emplace` / `iter->second.run` 以及 `AllocationAnalysis` 在头文件里找不到——下面会专门拆解。

### 3.1 要解决的问题

Kernel 中多个 op 需要 shared memory，生命周期可能重叠：

| Op | 原因 |
|----|------|
| `ttg.convert_layout` | 布局转换暂存 buffer |
| `tt.reduce` / `tt.scan` / `tt.gather` / `tt.histogram` | 跨线程数据交换 |
| `tt.atomic_rmw` / `tt.atomic_cas` | 原子操作暂存 |
| `ttg.local_alloc` | 用户显式分配 |

**目标**：满足生命周期约束的前提下，最小化 shared memory 总占用。

### 3.2 三个类各自干什么

整个框架涉及三个类，职责分明：

| 类 | 所在文件 | 角色 |
|----|---------|------|
| `CallGraph<Allocation>` | `Utility.h` | 扫描 module，建**调用关系图**（`graph` + `roots`） |
| `Allocation` | `Allocation.h` | **单个 function** 的 smem 账本（buffer 列表、offset、总 size） |
| `ModuleAllocation` | `Allocation.h` | 继承 `CallGraph<Allocation>`，遍历调用图，为每个 func 创建 `Allocation` 并触发分析 |
| `AllocationAnalysis` | `Allocation.cpp`（**仅 .cpp，头文件无类体**） | 真正干活的分析器，结果写入 `Allocation` |

`ModuleAllocation` 继承关系决定了 `funcMap` 里存什么：

```cpp
class ModuleAllocation : public CallGraph<Allocation> { ... };

// 模板参数 T = Allocation → funcMap 的值类型就是 Allocation
using FuncDataMapT = DenseMap<FunctionOpInterface, Allocation>;
```

**每个 function 绑定一个 `Allocation` 对象**，专门存放该函数内的 smem 分析结果——不是 funcOp 本身，而是一个「账本」容器。

`Allocation` 对象内部的字段（`run()` 前全空，`run()` 后被填满）：

| 字段 | `run()` 前 | `run()` 后 |
|------|-----------|-----------|
| `operation` | 指向 `tt.func` | 不变 |
| `bufferSet` | `{}` | 所有 buffer 的 size / offset / kind |
| `opScratch` / `valueBuffer` | `{}` | op/value → buffer 的映射 |
| `sharedMemorySize` | `0` | 图着色后的总字节数 |

### 3.3 `CallGraph::build()` 与 `funcMap`：两张表，两个时机

`ModuleAllocation` 构造时先调 `CallGraph(moduleOp)`，内部执行 `build()`。**`build()` 不填充 `funcMap`**：

| 成员 | `build()` 时？ | 内容 |
|------|--------------|------|
| `graph` | ✅ 填充 | 调用关系：caller → [(callOp, callee), ...] |
| `roots` | ✅ 填充 | 入口函数（从未被 call 的 func，即 kernel） |
| `funcMap` | ❌ **仍为空 `{}`** | 函数 → `Allocation` 账本（walk 时才 lazy 创建） |

```cpp
// Utility.h:352-374
void build() {
  moduleOp.walk([&](Operation *op) {
    if (auto callOp = dyn_cast<CallOpInterface>(op))
      graph[caller].emplace_back(callOp, callee);  // 只建调用边
  });
  // roots = 没被任何人 call 的 func
  // funcMap 完全不碰
}
```

`funcMap` 的填充发生在构造函数紧接着的 `walk` 回调里（见 3.4）。

### 3.4 构造函数源码解读：费解的那几行

`ModuleAllocation` 构造函数完整逻辑：

```cpp
ModuleAllocation(ModuleOp moduleOp,
                 AllocationAnalysisScratchSizeFn scratchSizeGetter = ...)
    : CallGraph<Allocation>(moduleOp)   // ① build() → graph + roots，funcMap 仍空
{
  walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
      [](CallOpInterface, FunctionOpInterface) {},   // 边回调：空操作
      [&](FunctionOpInterface funcOp) {               // 节点回调：后序，每个 func 触发一次
        auto [iter, inserted] = funcMap.try_emplace(funcOp, funcOp);
        if (inserted)
          iter->second.run(funcMap, scratchSizeGetter);
      });
}
```

#### 逐行拆解 `try_emplace(funcOp, funcOp)`

```cpp
auto [iter, inserted] = funcMap.try_emplace(funcOp, funcOp);
//                          key = funcOp（FunctionOpInterface，即 tt.func @kernel）
//                          value 用第二个 funcOp 构造 → Allocation(funcOp)
//                          等价于 funcMap[funcOp] = Allocation(funcOp)

iter->second.run(funcMap, scratchSizeGetter);
//  ↑ second 的类型是 Allocation&，不是 funcOp！
//  调用 Allocation::run()，不是 funcOp 上的任何方法
```

| 符号 | 类型 | 含义 |
|------|------|------|
| `funcOp` | `FunctionOpInterface` | 当前遍历到的 MLIR 函数 |
| `iter->second` | `Allocation` | 该函数的 smem 账本（空容器 → run 后填满） |
| `funcMap` | 整个 module 的 func→Allocation 表 | 传给 `run()`，供 call op 查 callee 已算好的 size |

**常见误解澄清**：

| 误解 | 实际 |
|------|------|
| 「构造函数在算 size」 | 构造函数只做调度；size 在 `run()` 触发的 `AllocationAnalysis` 里算 |
| 「`second` 是 funcOp」 | `second` 是 `Allocation` 对象 |
| 「`build()` 初始化了 funcMap」 | `build()` 只建 `graph`/`roots`；`funcMap` 在 walk 时 lazy 插入 |
| 「`Allocation` 构造时就有分析结果」 | 构造时只有 `operation=funcOp`，必须 `run()` 后才填入 buffer/offset/size |

#### 为何后序遍历？为何传入整个 `funcMap`？

caller 里的 `triton.call` 需要读 callee 已算好的 `sharedMemorySize`：

```cpp
// Allocation.cpp — getScratchValueSize 处理 call op
auto *funcAlloc = &(*funcAllocMap)[calleeFuncOp];
auto bytes = funcAlloc->getSharedMemorySize();  // 依赖 callee 先 run 完
```

所以 walk 必须**后序**（callee 先于 caller），且 `run()` 需要拿到**整个** `funcMap` 的指针。

#### 时间线

```
ModuleAllocation(mod, scratchSizeGetter)
  │
  ├─ CallGraph(mod) → build()     graph + roots 就绪，funcMap = {}
  │
  └─ walk 后序每个 funcOp：
       try_emplace(funcOp, funcOp)           → 插入空 Allocation{operation=funcOp}
       iter->second.run(funcMap, getter)      → AllocationAnalysis 填入结果
  │
  构造结束 → funcMap 每个 func 都有完整 Allocation
  └─ attachAllocationSizeAndOffsetAttr       → 读 funcMap 写 IR 属性
```

### 3.5 从 `run()` 到 `AllocationAnalysis`：调用链

构造函数自身不算 size，但它触发的 `run()` 会。难点在于 **`AllocationAnalysis` 藏在 `.cpp` 里**，头文件只有前向声明：

```cpp
// Allocation.h:15 — 只有声明，没有类体
namespace triton { class AllocationAnalysis; }

// Allocation.h:73-74 — run() 声明
void run(FuncAllocMapT &, AllocationAnalysisScratchSizeFn);
```

四步跳转（对应源码行号）：

```
① Allocation.h:234
   iter->second.run(funcMap, scratchSizeGetter)
        │  second 是 Allocation → 成员函数派发
        ▼
② Allocation.cpp:602
   AllocationAnalysis(getOperation(), &funcMap, this, scratchSizeGetter)
        │  创建临时对象；this = 当前 Allocation 账本
        ▼
③ Allocation.cpp:122
   AllocationAnalysis 构造函数 { run(); }    ← 构造时自动执行，无显式 analysis.run()
        ▼
④ Allocation.cpp:134-137
   getValuesAndSizes() → resolveLiveness() → computeOffsets()
        结果写入 this->bufferSet / this->sharedMemorySize
```

`Allocation::run()` 的实现就两行：

```cpp
void Allocation::run(FuncAllocMapT &funcAllocMap,
                     AllocationAnalysisScratchSizeFn scratchSizeGetter) {
  triton::AllocationAnalysis(getOperation(), &funcAllocMap, this, scratchSizeGetter);
}
```

| 传入参数 | 值 | 作用 |
|---------|-----|------|
| `getOperation()` | 当前 funcOp | 遍历该函数内所有 op |
| `&funcAllocMap` | 整个 funcMap | call op 查 callee size |
| `this` | 当前 Allocation | 分析结果写入此对象 |
| `scratchSizeGetter` | 回调 | 计算 scratch buffer 字节数（见 §4） |

`AllocationAnalysis` 是 `Allocation` 的 `friend`，能直接写 `allocation->bufferSet` 等私有成员。临时对象析构后，数据留在 `this` 上。

### 3.6 分析流水线：`AllocationAnalysis::run()` 三步

```
getValuesAndSizes()     收集每个 buffer 的字节数
    ↓
resolveLiveness()       每个 buffer 何时活跃
    ↓
computeOffsets()        图着色分配 offset → sharedMemorySize
```

#### 第一步：`getValuesAndSizes()` — buffer size 从哪来？

遍历 funcOp 内每个 op，两类入口：

```cpp
operation->walk([&](Operation *op) {
  getExplicitValueSize(op);   // ttg.local_alloc → 从 tensor type 算，不走回调
  getScratchValueSize(op);    // 其他 op → 见下表
});
```

| Buffer 类型 | Kind | size 来源 | 走 `scratchSizeGetter`？ |
|------------|------|----------|------------------------|
| `ttg.local_alloc` | Explicit | tensor encoding → 元素数 × 字节宽 | ❌ |
| `convert_layout` / `reduce` 等 | Scratch | `scratchSizeGetter(op)` | **✅** |
| `triton.call` | Virtual | callee 的 `getSharedMemorySize()` | ❌ |
| `warp_specialize` 等 | Scratch | op 属性硬编码 | ❌ |

`scratchSizeGetter` 是唯一可插拔的钩子（Generic / NVIDIA 差异见 §4），但只是 size 来源之一。回调的真正调用点：

```cpp
unsigned bytes = scratchSizeGetter(op);
maybeAddScratchBuffer<BufferT::BufferKind::Scratch>(op, bytes, 128);
```

#### 第二步：`resolveLiveness()`

为每个 buffer 计算 CFG 上的活跃区间。Explicit buffer 跟 value 的 use-def；Scratch buffer 的生命周期绑定在所属 op 上。

#### 第三步：`computeOffsets()` — 图着色

1. 活跃区间重叠的 buffer 不能共享同一块内存 → 建干涉图
2. 按 buffer size **降序**做 first-fit 图着色
3. `sharedMemorySize = max(offset + size)` 写入当前 `Allocation`

### 3.7 串起来：一个 function 的完整生命周期

以 module 中有 `@kernel` 和 `@helper` 为例：

```
funcMap = {}

walk 先到 @helper（后序：callee 先）：
  funcMap[@helper] = Allocation{@helper}     // 空账本
  Allocation{@helper}.run(funcMap, getter)
    → AllocationAnalysis 填入 bufferSet, sharedMemorySize=4096

walk 再到 @kernel：
  funcMap[@kernel] = Allocation{@kernel}     // 空账本
  Allocation{@kernel}.run(funcMap, getter)
    → kernel 内有 triton.call @helper
    → getScratchValueSize(call) 读 funcMap[@helper].sharedMemorySize = 4096
    → 加上 kernel 自身的 cvt/reduce buffer
    → sharedMemorySize = 12288

构造结束 → attachAllocationSizeAndOffsetAttr 读 funcMap 写 allocation.offset / ttg.shared
```

---


## 4. `scratchSizeGetter` 详解

`scratchSizeGetter` 是本 pass **backend 差异化的唯一入口**。类型定义在 `Allocation.h`：

```cpp
/// Callback to allow backends to specify target-specific scratch sizes for some operations.
using AllocationAnalysisScratchSizeFn = std::function<unsigned(Operation *)>;

unsigned defaultAllocationAnalysisScratchSizeFn(Operation *op);
```

`ModuleAllocation` 构造函数将其作为**唯一可定制参数**，默认值为 `defaultAllocationAnalysisScratchSizeFn`：

```cpp
ModuleAllocation(ModuleOp moduleOp,
                 triton::AllocationAnalysisScratchSizeFn scratchSizeGetter =
                     triton::defaultAllocationAnalysisScratchSizeFn)
```

在 `AllocationAnalysis::getScratchValueSize()` 中，遍历 function 内每个 op 时调用：

```cpp
unsigned bytes = scratchSizeGetter(op);   // 输入：一个 Operation*；输出：scratch 字节数
maybeAddScratchBuffer<BufferT::BufferKind::Scratch>(op, bytes, scratchAlignment);
```

```
                    ┌─────────────────────────────────────┐
                    │  ModuleAllocation(mod, scratchGetter) │
                    └──────────────┬──────────────────────┘
                                   │ 传入
                    ┌──────────────▼──────────────────────┐
                    │  AllocationAnalysis::getScratchValueSize │
                    │    scratchSizeGetter(op)  ← 每个 op 调一次 │
                    └──────────────┬──────────────────────┘
                                   │ 返回 unsigned 字节数
                    ┌──────────────▼──────────────────────┐
                    │  addBuffer(Scratch, op, bytes, 128)  │
                    └─────────────────────────────────────┘
```

---

### 4.1 `defaultAllocationAnalysisScratchSizeFn` — Generic 默认实现

**文件**：`triton/lib/Analysis/Allocation.cpp:69-112`

**调用方**：Generic pass 省略第二参数时自动使用；NVIDIA pass 对非 `ConvertLayoutOp` 回退到此函数。

#### 完整源码

```cpp
unsigned defaultAllocationAnalysisScratchSizeFn(Operation *op) {
  if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    ReduceOpHelper helper(reduceOp);
    return helper.getScratchSizeInBytes();
  }
  if (auto scanOp = dyn_cast<ScanOp>(op)) {
    ScanLoweringHelper helper(scanOp);
    return helper.getScratchSizeInBytes();
  }
  if (auto gatherOp = dyn_cast<GatherOp>(op)) {
    GatherLoweringHelper helper(gatherOp);
    return helper.getScratchSizeInBytes();
  }
  if (auto histogram = dyn_cast<HistogramOp>(op)) {
    auto dstTy = histogram.getType();
    int threadsPerWarp = gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    return std::max<int>(dstTy.getNumElements(), threadsPerWarp) *
           getBitwidth(dstTy) / 8;
  }
  if (auto cvtLayout = dyn_cast<gpu::ConvertLayoutOp>(op)) {
    auto srcTy = cvtLayout.getSrc().getType();
    auto dstTy = cvtLayout.getType();
    if (!cvtNeedsSharedMemory(srcTy, dstTy))
      return 0;
    auto elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy);
    return elems * getBitwidth(srcTy) / 8;
  }
  if (isa<AtomicRMWOp, AtomicCASOp>(op)) {
    auto value = op->getOperand(0);
    auto smemShape = getRepShapeForAtomic(op->getResult(0));
    auto elems = getNumScratchElements(smemShape);
    if (elems == 0)
      return 0;
    auto elemTy = getElementTypeOrSelf(getPointeeType(value.getType()));
    return elems * std::max<int>(8, elemTy.getIntOrFloatBitWidth()) / 8;
  }
  if (isa<ttng::TensormapCreateOp>(op)) {
    constexpr int32_t kTMASize = 128;
    return kTMASize;
  }
  return 0;
}
```

#### 逐分支分析

函数结构是**按 op 类型的 if-else 链**，不匹配则返回 0（不需要 scratch）。每个分支的 scratch size 计算逻辑如下展开。

##### 分支 ①：`tt.reduce` — 跨 warp 归约暂存

**为什么需要 scratch？**

`tt.reduce` 沿指定轴将多个线程的值归约为一个。当归约轴涉及**多个 warp** 时，各 warp 的局部归约结果需要通过 shared memory 交换，才能得到最终结果。若只有 1 个 warp 参与（warp-synchronous），则完全在寄存器内用 shuffle 完成，无需 scratch。

**计算过程**（`ReduceOpHelper::getScratchSizeInBytes()` 位于 `Utility.cpp:122-131`）：

```
scratch_bytes = product(smemShape) × Σ(type_byte_width)
                ↑                        ↑
            smem 元素总数          所有参与归约的源类型字节宽之和
```

其中 `smemShape` 由 `getScratchRepShape()` 决定：

```cpp
SmallVector<unsigned> ReduceOpHelper::getScratchRepShape() {
  // ① 提前返回：warp-synchronous 时不需要 scratch
  if (isWarpSynchronous())
    return {0, 0};   // product = 0 → 0 字节
  // ② 以源 tensor shape 为基础，把归约轴替换为"有独立数据的 warp 数"
  smemShape = srcShape;
  smemShape[axis] = getInterWarpSizeWithUniqueData();
  return smemShape;
}
```

逻辑拆解：

| 步骤 | 含义 |
|------|------|
| `isWarpSynchronous()` | 检查沿归约轴的 warp 数是否 ≤ 1。若是，该归约无需跨 warp 通信，返回 0 |
| `getInterWarpSizeWithUniqueData()` | `= WarpsPerCTA[axis]`：沿归约轴有多少个 warp 持有互不重叠的数据分片 |
| `srcShape` 替换归约轴 | 只有归约轴需要跨 warp 通信，其余维度在 scratch 中保持原样 |
| `bytesPerElem` | 遍历 `srcElementTypes`（`ReduceOp` 可同时归约多个 tensor），`sum(ceil(bitwidth, 8))` |

**数据流示例**（假设 `src = tensor<128x64xf32>`，归约 axis=1，`WarpsPerCTA[1] = 4`）：

```
srcShape = [128, 64]
smemShape = [128, 4]        ← axis=1 从 64 替换为 4（独立 warp 数）
elems = 128 × 4 = 512
bytesPerElem = ceil(32,8) = 4  （仅 f32）
scratch = 512 × 4 = 2048 字节
```

**何时返回 0？**

- `isWarpSynchronous()` 为 true：归约轴只有 1 个 warp，shuffle 即可完成。

---

##### 分支 ②：`tt.scan` — 跨 warp 扫描暂存

**为什么需要 scratch？**

`tt.scan`（前缀和/前缀积）沿指定轴做扫描。与 reduce 类似，当沿扫描轴有**多个 warp** 时，各 warp 的局部扫描结果需要通过 shared memory 做跨 warp 归并。

**计算过程**（`ScanLoweringHelper::getScratchSizeInBytes()` 位于 `Utility.cpp:236-249`）：

```
scratch_bytes = elementSizeInBytes × getScratchSizeInElems()
```

**第一步：提前返回**

```cpp
// ① 仅 BlockedEncodingAttr 支持扫描
if (!isSupported())    return 0;
// ② 沿扫描轴只有 1 个 warp → 无需跨 warp 通信
if (axisNumWarps == 1) return 0;
```

**第二步：`elementSizeInBytes`**

与 reduce 相同：遍历 `srcElementTypes`，累加各类型的字节宽 `ceil(bitwidth, 8)`。`ScanOp` 也可以同时扫描多个 tensor，因此需要求和。

**第三步：`getScratchSizeInElems()`**（`Utility.cpp:227-234`）

```
numWarps × numNonAxisElementsPerWarp × axisNumBlocks × nonAxisNumBlocks
```

每个因子含义：

| 因子 | 计算方式 | 含义 |
|------|---------|------|
| `numWarps` | `product(WarpsPerCTA)` | CTA 内总 warp 数 |
| `numNonAxisElementsPerWarp` | `nonAxisThreadsPerWarp × nonAxisElementsPerThread` | 每个 warp 在非扫描轴上的总元素数 |
| `nonAxisThreadsPerWarp` | `totalThreadsPerWarp / axisThreadsPerWarp` | 非扫描轴上的线程数 |
| `nonAxisElementsPerThread` | 非扫描轴的 `ContigPerThread` 乘积 | 每个线程在非扫描轴上的连续元素数 |
| `axisNumBlocks` | `ceil(shape[axis] / (contig[axis] × threads[axis] × warps[axis]))` | 扫描轴上的 tile 块数 |
| `nonAxisNumBlocks` | 非扫描轴 tile 块数的乘积 | 非扫描轴上的 tile 块数 |

**公式的物理含义**：

scratch 的大小需要容纳所有 warp 交换的局部扫描结果。每个 "块" 代表一组可独立处理的扫描分片，乘以 warp 数和非扫描轴上的元素数，得到总元素数。

**数据流示例**（假设 `src = tensor<128x64xf32>`，axis=1，`WarpsPerCTA=[4,2]`，`ThreadsPerWarp=[16,2]`，`ContigPerThread=[1,4]`）：

```
axisNumWarps = 2             → axis=1 有 2 个 warp
axisNumWarps == 1 ？       → No，继续计算
numWarps = 4 × 2 = 8
numNonAxisElementsPerWarp = (32/(2)) × (1) = 16   （axis=0 相关）
axisNumBlocks  = ceil(64 / (4×2×2)) = ceil(64/16) = 4
nonAxisNumBlocks = ceil(128 / (1×16×4)) = ceil(128/64) = 2
scratchElems = 8 × 16 × 4 × 2 = 1024
scratchBytes = 1024 × 4 = 4096
```

**何时返回 0？**

- `isSupported()` 为 false：encoding 不是 `BlockedEncodingAttr`
- `axisNumWarps == 1`：扫描轴只有 1 个 warp，shuffle 即可

---

##### 分支 ③：`tt.gather` — 跨线程收集暂存

**为什么需要 scratch？**

`tt.gather` 根据索引从源 tensor 中收集元素。当 gather 不是 **warp-local** 时——即源 tensor 沿 gather 轴的数据分布在**多个 warp** 中——需要将整个源 tensor 写入 shared memory，让所有线程都能读到任意位置的数据。

**计算过程**（`GatherLoweringHelper::getScratchSizeInBytes()` 位于 `Utility.cpp:781-792`）：

```cpp
if (isWarpLocal())    return 0;     // 完全在 warp 内完成
// 保守策略：把整个源 tensor 写回 shared memory
RankedTensorType srcType = gatherOp.getSrc().getType();
return product(srcType.getShape()) *
       ceil<unsigned>(srcType.getElementTypeBitWidth(), 8);
```

**逻辑拆解**：

| 条件 | scratch |
|------|---------|
| `isWarpLocal() == true` | 源 tensor 沿 gather 轴的每一列完全属于同一个 warp，无需 shared memory，返回 0 |
| `isWarpLocal() == false` | **整个源 tensor** 溢出到 shared memory，元素数 × 元素字节宽 |

**保守设计**：

代码注释明确写着 *"For now, assume the whole source tensor is written back to shared memory"*。这是一个**保守的兜底策略**——把整个源 tensor 拷贝到 shared memory 虽然开销大，但保证所有线程都能按任意索引访问。未来可以优化为只拷贝被索引到的分片。

**数据流示例**（`src = tensor<256xf32>`）：

```
product(srcShape) = 256
bytesPerElement = ceil(32, 8) = 4
scratch = 256 × 4 = 1024 字节
```

**何时返回 0？**

- `isWarpLocal()` 为 true：gather 的源数据完全在 warp 内部

---

##### 分支 ④：`tt.histogram` — 跨 warp 直方图聚合

**为什么需要 scratch？**

直方图计算分两步：
1. **Warp 级别**：每个 warp 用 ballot + popcount 计算私有直方图，分布在各个线程上
2. **跨 warp 聚合**：通过 shared memory 中的 **atomic add** 将各 warp 的局部直方图累加到一起，然后线程按输出 layout 加载结果

scracth buffer 就是跨 warp 聚合时用来放直方图数据的共享内存。

**计算过程**（`Allocation.cpp:82-88`）：

```cpp
auto dstTy = histogram.getType();
int threadsPerWarp = gpu::TritonGPUDialect::getThreadsPerWarp(
    op->getParentOfType<ModuleOp>());
return std::max<int>(dstTy.getNumElements(), threadsPerWarp) *
       getBitwidth(dstTy) / 8;
```

**为什么取 `max(numElements, threadsPerWarp)`？**

关键在 lowering 代码（`HistogramOpToLLVM.cpp:167`）：

```cpp
int numBins = op.getType().getDimSize(0);
// ...
// Pad out the bins so that we have at least one bin per thread within a warp.
numBins = std::max(numBins, numThreadsPerWarp);
```

Warp 级别直方图算法要求每个线程至少拥有一个 bin：

```cpp
assert(numBins % numThreadPerWarp == 0 &&
       "numBins must be divisible by numThreadPerWarp");
```

因此当用户指定的 bin 数 `< threadsPerWarp` 时，lowering 会**向上填充**到 `threadsPerWarp`，分配也必须按填充后的大小。

| 场景 | bin 数 | 填充后 | scratch 公式 |
|------|--------|--------|-------------|
| bin 数充足 | 128 | 128（无需填充） | `128 × 4 = 512` 字节 |
| bin 数不足 | 8 | 32（填充到 threadsPerWarp） | `max(8, 32) × 4 = 128` 字节 |

**类型说明**：

`tt.histogram` 的输出始终是 `tensor<N x i32>`（Python builder 硬编码 `int32`），所以 `getBitwidth()` 恒为 32。实际公式简化为：

```
scratch = max(numBins, threadsPerWarp) × 4  字节
```

**`getBitwidth` 的内部**（`Utility.cpp:123-126`）：

```cpp
unsigned tt::getBitwidth(RankedTensorType ty) {
  auto isPtr = isa<PointerType>(ty.getElementType());
  return isPtr ? kPtrBitWidth : std::max(ty.getElementTypeBitWidth(), 8u);
}
```

最小值 8 位（1 字节），确保 sub-byte 类型（如 i1）也能正确计算。

---

##### 分支 ⑤：`ttg.convert_layout` — 布局转换暂存

**为什么需要 scratch？**

`ttg.convert_layout` 在两种 GPU tensor layout（如 `BlockedEncoding` ↔ `MmaEncoding`）之间转换数据。当转换不能通过**寄存器重排**或 **warp shuffle** 完成时，需要 shared memory 作为中转：源 layout 写入 smem，再从 smem 按目标 layout 读出。

**判断是否走 smem**（`cvtNeedsSharedMemory`）：

```cpp
bool cvtNeedsSharedMemory(RankedTensorType srcTy, RankedTensorType dstTy) {
  return !cvtReordersRegisters(srcTy, dstTy) &&
         !cvtNeedsWarpShuffle(srcTy, dstTy);
}
```

| 条件 | 含义 | scratch |
|------|------|---------|
| `cvtReordersRegisters == true` | 仅重排寄存器即可完成转换 | 0 |
| `cvtNeedsWarpShuffle == true` | 可用 warp shuffle 指令完成 | 0 |
| 两者均为 false | 必须经过 shared memory 中转 | 计算 swizzling size |

**需要 smem 时——swizzling 优化**：

Generic 路径调用 `getNumScratchElemsSwizzledCvt`（`Allocation.cpp:32-43`）：

```cpp
unsigned getNumScratchElemsSwizzledCvt(RankedTensorType srcTy,
                                       RankedTensorType dstTy) {
  auto *ctx = srcTy.getContext();
  auto srcLayout = gpu::toLinearLayout(srcTy);
  auto dstLayout = gpu::toLinearLayout(dstTy);
  // ① 移除广播维度：广播寄存器不占用显式 shared memory 槽位
  srcLayout = actionRemoveBroadcastedRegs(srcLayout).apply(srcLayout);
  dstLayout = actionRemoveBroadcastedRegs(dstLayout).apply(dstLayout);
  auto bitwidth = getBitwidth(srcTy);
  // ② 用 ld/st.shared 指令约束求解最优 swizzling 布局
  auto smem = gpu::optimalSwizzlingLdSt(srcLayout, dstLayout, bitwidth);
  // ③ totalOutDimSize = swizzle 后 smem 的总地址空间大小
  //    reps = 重复因子，用于分摊 swizzle 开销
  auto reps = smem.getInDimSize(StringAttr::get(ctx, "reps"));
  return smem.getTotalOutDimSize() / reps;
}
```

**最终**：`scratch_bytes = elems × bitwidth / 8`。

Swizzling 通过 XOR 变换物理地址来减少 bank conflict，付出的代价是共享内存略大于原始 tensor 大小（对齐到 swizzle 的粒度）。详见 §5。

---

##### 分支 ⑥：`tt.atomic_rmw` / `tt.atomic_cas` — 原子操作暂存

**为什么需要 scratch？**

Triton 采用**块级编程模型**：一个 kernel 中多个线程协同操作一个 tensor 的同一分区。对于原子操作，即使某些线程不参与，同一分区内所有线程必须能看到相同的值。当原子结果有**广播维度**时，需要 shared memory 暂存来确保广播一致性。

**计算过程**（`Allocation.cpp:98-105`）：

```cpp
auto value = op->getOperand(0);      // ptr 操作数（全局指针）
auto smemShape = getRepShapeForAtomic(op->getResult(0));
auto elems = getNumScratchElements(smemShape);
if (elems == 0)  return 0;
auto elemTy = getElementTypeOrSelf(getPointeeType(value.getType()));
return elems * std::max<int>(8, elemTy.getIntOrFloatBitWidth()) / 8;
```

**第一步：`getRepShapeForAtomic` 确定 shape**（`Allocation.cpp:49-67`）：

```cpp
static SmallVector<unsigned> getRepShapeForAtomic(Value result) {
  SmallVector<unsigned> smemShape;
  if (!result.use_empty()) {
    if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
      auto freeVariableMasks =
          gpu::toLinearLayout(tensorTy).getFreeVariableMasks();
      // 检查是否有广播维度（freeVariableMask != 0 表示该维度有广播）
      if (llvm::any_of(freeVariableMasks,
            [](auto variableMask) { return variableMask.second != 0; })) {
        smemShape = convertType<unsigned>(gpu::getShapePerCTA(tensorTy));
      }
    } else {
      smemShape.push_back(1);   // 标量结果也需要 1 个元素的 scratch
    }
  }
  return smemShape;
}
```

| 结果类型 | 条件 | smemShape | 含义 |
|---------|------|-----------|------|
| tensor | 有广播维度（`freeVariableMask != 0`） | `shapePerCTA` | 按完整 CTA shape 分配暂存 |
| tensor | 无广播维度 | `{}`（空） | 无需 scratch |
| scalar | 结果被使用 | `{1}` | 1 个元素暂存 |
| 任意 | 结果未被使用 | `{}` | 不分配 |

**第二步：从 ptr 操作数提取元素字节宽**

```cpp
// op->getOperand(0) = ptr（AtomicRMWOp/CASOp 的第一个操作数始终是指针）
// getPointeeType 解开指针包装：
//   ptr<f32>     → f32
//   tensor<128xptr<f32>> → tensor<128xf32>
// getElementTypeOrSelf 提取标量元素类型：
//   tensor<128xf32> → f32
//   f32              → f32（保持原样）
auto elemTy = getElementTypeOrSelf(getPointeeType(value.getType()));
```

**第三步：计算字节数**

```
scratch = elems × max(8, elemBitWidth) / 8
```

`max(8, ...)` 保证**至少 1 字节**每个元素（sub-byte 类型如 i1 也能正确对齐）。

**数据流示例**（`tt.atomic_rmw` 在 `tensor<128xf32>` 上，结果有广播维度）：

```
smemShape = shapePerCTA = [128]
elems = 128
elemTy = f32 → bitwidth = 32
scratch = 128 × max(8, 32) / 8 = 128 × 4 = 512 字节
```

---

##### 分支 ⑦：`ttng.tensormap_create` — TMA 描述符

**为什么需要 scratch？**

`ttng.tensormap_create` 在 shared memory 中构造 TMA（Tensor Memory Accelerator）描述符，然后通过 `tensormap.cp_fenceproxy` PTX 指令将其拷贝到 `desc_ptr` 指向的全局内存。TMA 描述符由硬件规范固定为 **128 字节**。

**计算过程**（`Allocation.cpp:107-109`）：

```cpp
constexpr int32_t kTMASize = 128;   // TMA_SIZE_BYTES
return kTMASize;
```

| 常量 | 值 | 来源 |
|------|-----|------|
| `TMA_SIZE_BYTES` | 128 | `TMAUtilities.h:11` |
| `TMA_ALIGN` | 128 | `TMAUtilities.h:12` |

**lowering 如何使用这 128 字节？**（`TMAToLLVM.cpp:243-296`）

```
① getSharedMemoryBase(loc, rewriter, targetInfo, op)
    → 分配 scratch 基址
② zero_fill_tma(smemBase)
    → warp 0 写 32 × i32 = 128 字节清零
③ tensormap_replace_*(smemBase, ...)
    → 通过 `tensormap.replace.tile` PTX 逐字段填入：
      global_address, rank, box_dim, global_dim,
      global_stride, element_stride, elemtype,
      interleave_layout, swizzle_mode, fill_mode
④ tensormap_cp_fenceproxy(descPtr, smemBase)
    → `tensormap.cp_fenceproxy.global.shared::cta...` 指令
      将 128 字节从 shared memory 拷贝到 desc_ptr
```

**为何固定 128？**

TMA 是 NVIDIA 的硬件单元（从 Hopper 架构引入），其描述符格式由硬件 ISA 定义，大小固定为 128 字节。不需要也不可以动态计算。

---

##### 分支 ⑧：默认（其他所有 op）

```cpp
return 0;
```

所有未被以上分支匹配的 op（如 `tt.addptr`、`tt.splat`、`tt.broadcast` 等纯算术/数据流操作）不需要 scratch shared memory，返回 0。

#### `getWarpsPerCTA` 底层原理

分支 ① 和 ② 频繁用到 `getWarpsPerCTA()` 查每个 tensor 维度上有多少 warp。这个查询背后是一套从 `LinearLayout` 的基向量解码出维度分布的统一机制。

##### 调用链路

```
getWarpsPerCTA(layout, shape)                         // Dialect.cpp:107-111
  └→ toLinearEncoding(layout, shape)                    // 将任意 layout 转成 LinearEncodingAttr
      .getWarpsPerCTA()                                // Dialect.cpp:1054-1056
    └→ basesPerDim("warp")
        └→ basesPerDimImpl(ll.getBases(), "warp", rank) // Dialect.cpp:968-995
```

##### LinearLayout 的基向量（basis vector）

`LinearLayout` 是一个线性映射——它把输入坐标（如 `lane_id`、`warp_id` 的二进制位）映射到输出坐标（tensor 各维度的索引）。它的 `bases` 存储每个输入维度的基向量（`Dialect.h:317-324`）：

```cpp
// bases[inDim][i] = L(..., inDim=2^i, ...)
//    ↑ 输入维度名      ↑ 第 i 个基向量（长度为 rank，即输出维数）
//                      inDim=2^i 表示"该输入维度的第 i 个 bit 置 1"
//   每个基向量恰好有 1 个非零元素（或全零——广播情况）
```

对 "warp" 维度来说，warp ID 有 `log2(numWarps)` 个 bit。每个 bit 对应一个基向量，基向量中非零元素的位置决定了这个 bit 影响 tensor 的哪个维度。

##### `basesPerDimImpl` 完整解读

```cpp
static SmallVector<unsigned>
basesPerDimImpl(const BasesT &namedBases, StringAttr dimName,
                size_t rank, bool skipBroadcast) {
  const auto &bases = namedBases.find(dimName)->second;

  // 没有基向量：该维度不参与映射 → 每维只有 1 个
  if (bases.empty())
    return SmallVector<unsigned>(rank, 1);

  SmallVector<unsigned> ret(rank, 1);    // 每维至少 1 个
  int nonZeroIdx = 0;
  for (const auto &basis : bases) {       // 遍历 warp ID 的每个 bit
    auto it = std::find_if(basis.begin(), basis.end(),
                           [](auto val) { return val != 0; });

    if (it != basis.end()) {              // ── case A：该 bit 影响某个输出维度
      nonZeroIdx = it - basis.begin();    //     确定受影响的维度
      ret[nonZeroIdx] *= 2;              //     该维度 warp 数 ×2
    } else if (!skipBroadcast) {          // ── case B：全零（广播），不跳过时
      ret[nonZeroIdx] *= 2;              //     算到上一个维度上
    }                                     // ── case C：全零 + skipBroadcast → 跳过
  }
  return ret;                             // 结果每个元素都是 2 的幂
}
```

**核心逻辑**：warp ID 每个 bit 遍历一次，每次在受影响的维度上 `×2`，所以结果是 2 的幂的向量。下面从原始 `BlockedEncodingAttr` 出发，看它如何被转换为 LinearLayout，再被解码回 `warpsPerCTA`。

##### 从 `BlockedEncodingAttr` 到 LinearLayout 再到 `warpsPerCTA`

`BlockedEncodingAttr::toLinearLayout()`（`LinearLayoutConversions.cpp:851-858`）将 layout 的三组属性分别构造为 LinearLayout，然后相乘：

```cpp
LinearLayout ctaLayout =
    identityStandardND(S("register"), getSizePerThread(), order) *
    identityStandardND(S("lane"), getThreadsPerWarp(), order) *
    identityStandardND(S("warp"), getWarpsPerCTA(), order);
```

其中 `identityStandardND`（`LayoutUtils.cpp:162-178`）按 `order` 指定的维度顺序，为每个维度创建一个 `identity1D`，然后依次相乘：

```cpp
LinearLayout identityStandardND(StringAttr inDimName, ArrayRef<unsigned> shape,
                                ArrayRef<unsigned> order) {
  LinearLayout ret = LinearLayout::empty();
  for (int i = 0; i < shape.size(); i++) {
    int dim = order[i];                           // 从最快变化到最慢变化
    ret *= LinearLayout::identity1D(shape[dim], inDimName, outDimNames[dim]);
  }
  return ret;
}
```

`identity1D(n, inDim, outDim)` 创建线性映射 `L(inDim) = outDim`，产生 `log2(n)` 个基向量，每个基向量对应 inDim 的一个 bit。

##### 3 个实例（从原始 layout 推导到结果）

**例 1：`warpsPerCTA=[2, 2]`，`order=[1, 0]`（4 warps 排成 2×2 网格）**

```
原始 BlockedEncodingAttr（部分属性）：
  { warpsPerCTA=[2, 2], order=[1, 0] }

↓ identityStandardND("warp", [2, 2], [1, 0])：

  迭代 order：
    i=0: dim=order[0]=1  → identity1D(2, "warp", "dim1")
                            └─ log2(2)=1 个基向量：L(warp=1) = (dim1=1)
                               基向量 [0, 1]     ← 在 2D 输出中的表示
    i=1: dim=order[1]=0  → identity1D(2, "warp", "dim0")
                            └─ log2(2)=1 个基向量：L(warp=1) = (dim0=1)
                               基向量 [1, 0]

↓ LinearLayout 中 "warp" 的基向量（共 2 个）：
  bases["warp"][0] = [0, 1]    ← warp bit0 → 影响 dim1
  bases["warp"][1] = [1, 0]    ← warp bit1 → 影响 dim0

↓ basesPerDimImpl 迭代：

  ret = [1, 1]  (初始值)
  basis[0]=[0,1]: nonZeroIdx=1 → ret[1]*=2 → [1, 2]
  basis[1]=[1,0]: nonZeroIdx=0 → ret[0]*=2 → [2, 2]

结果：warpsPerCTA = [2, 2] ✓   ← dim0 有 2 个 warp, dim1 有 2 个 warp
```

**例 2：`warpsPerCTA=[4, 1]`，`order=[1, 0]`（4 warps 全部沿 dim0）**

```
原始 BlockedEncodingAttr：
  { warpsPerCTA=[4, 1], order=[1, 0] }

↓ identityStandardND("warp", [4, 1], [1, 0])：

  迭代 order：
    i=0: dim=order[0]=1  → identity1D(1, "warp", "dim1")
                            └─ log2(1)=0 个基向量（空）
    i=1: dim=order[1]=0  → identity1D(4, "warp", "dim0")
                            └─ log2(4)=2 个基向量：
                               bit0: L(warp=1) = (dim0=1) → 基向量 [1, 0]
                               bit1: L(warp=2) = (dim0=2) → 基向量 [2, 0]

↓ LinearLayout 中 "warp" 的基向量（共 2 个，全部指向 dim0）：
  bases["warp"][0] = [1, 0]    ← warp bit0 → 影响 dim0（stride 1）
  bases["warp"][1] = [2, 0]    ← warp bit1 → 影响 dim0（stride 2）

↓ basesPerDimImpl 迭代：

  ret = [1, 1]  (初始值)
  basis[0]=[1,0]: nonZeroIdx=0 → ret[0]*=2 → [2, 1]
  basis[1]=[2,0]: nonZeroIdx=0 → ret[0]*=2 → [4, 1]

结果：warpsPerCTA = [4, 1] ✓   ← dim0 有 4 个 warp, dim1 只有 1 个
```

**例 3：`DotOperandEncodingAttr` 中的广播 warp**

某些 MMA layout（如 Ampere 的 DotOperand）中，一个 warp 的某些 bit 不影响输出坐标——该 bit 全零，表示广播：

```
原始 DotOperandEncodingAttr（父 layout 的 warpsPerCTA=[2, 1]）：
  有 2 个 warp，但 warp bit1 额外占用了一个"折叠"位

↓ toLinearLayout 中 warp 的基向量：
  bases["warp"][0] = [1, 0]    ← warp bit0 → 影响 dim0
  bases["warp"][1] = [0, 0]    ← 全零！该 bit 不影响任何输出 dim（广播）

↓ basesPerDimImpl：

  skipBroadcast=true（默认）→ 全零基向量跳过
    ret=[1,1] → basis[0]=[1,0]: ret[0]*=2 → [2,1]
                basis[1]=[0,0]: 跳过
    结果：warpsPerCTA = [2, 1]    ← 正确反映了 warp 的布局

  skipBroadcast=false（CTA 计数时）→ 全零基向量算到上一个非零维度
    ret=[1,1] → basis[0]=[1,0]: ret[0]*=2 → [2,1]
                basis[1]=[0,0]: ret[0]*=2 → [4,1]
    结果：warpsPerCTA = [4, 1]    ← 包含了广播 bit 的计数
```

`skipBroadcast` 的用途：对 warp/thread 维度，广播 bit 不影响实际数据布局，跳过即可获得正确的每维度 warp 数；对 CTA/block 维度需要精确计数（广播 bit 对应跨 CTA 的复制），所以 `getCTASplitNum()` 传 `skipBroadcast=false`。

##### 为什么总是 2 的幂？

`basesPerDimImpl` 只做 `×2` 操作，所以 `warpsPerCTA` 每个元素必然都是 2 的幂。`BlockedEncodingAttr::verify()` 中有显式断言：

```cpp
if (llvm::any_of(warpsPerCTA,
                 [](unsigned x) { return !llvm::isPowerOf2_64(x); }))
  return emitError() << "Every element in warpsPerCTA must be a power of two.";
```

---

### 4.2 `getNvidiaAllocationAnalysisScratchSizeFn` — NVIDIA 工厂函数

**文件**：`triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp:61-76`

**调用方**：NVIDIA pass 构造 `ModuleAllocation` 时传入：

```cpp
TargetInfo targetInfo(computeCapability, ptxVersion);
ModuleAllocation allocation(
    mod, getNvidiaAllocationAnalysisScratchSizeFn(targetInfo));
//                          ↑ 工厂函数，返回 std::function
```

#### 为何用工厂函数 + lambda？

`AllocationAnalysisScratchSizeFn` 的类型是 `std::function<unsigned(Operation *)>`——**不接受额外参数**。NVIDIA 需要把 `targetInfo` 传进回调，所以用工厂函数返回一个**捕获了 `targetInfo` 引用的 lambda**：

```cpp
std::function<unsigned(Operation *)>
getNvidiaAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto srcTy = cvtOp.getSrc().getType();
      auto dstTy = cvtOp.getType();
      if (!cvtNeedsSharedMemory(srcTy, dstTy))
        return 0;
      // In cuda we always swizzle
      auto elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy, targetInfo);
      return elems * getBitwidth(srcTy) / 8;
    }
    return defaultAllocationAnalysisScratchSizeFn(op);  // ← 其他 op 完全委托
  };
  return allocation;
}
```

决策逻辑只有两层：

```
scratchSizeGetter(op)
  ├─ ConvertLayoutOp ?
  │    ├─ cvtNeedsSharedMemory?  No → 0
  │    └─ Yes → getNumScratchElemsSwizzledCvt(src, dst, targetInfo)  ← NVIDIA 专属
  └─ 其他所有 op → defaultAllocationAnalysisScratchSizeFn(op)           ← 完全回退
```

**NVIDIA 版只覆盖了 `ConvertLayoutOp` 一个分支**，reduce/scan/gather/atomic 等全部由 default 处理。

#### NVIDIA 版 `getNumScratchElemsSwizzledCvt`

与 Generic 同名但签名不同（`static`，定义在 nvidia `Allocation.cpp:45-59`）：

```cpp
static unsigned getNumScratchElemsSwizzledCvt(RankedTensorType srcTy,
                                              RankedTensorType dstTy,
                                              TargetInfoBase &targetInfo) {
  auto srcLayout = actionRemoveBroadcastedRegs(toLinearLayout(srcTy)).apply(...);
  auto dstLayout = actionRemoveBroadcastedRegs(toLinearLayout(dstTy)).apply(...);
  auto bitwidth = getBitwidth(srcTy);
  auto [srcTiles, dstTiles] = gpu::getSrcDstTiles(targetInfo, bitwidth);  // ← 架构 tile
  auto [smem, _] = triton::gpu::optimalSwizzling(srcLayout, dstLayout,
                                                  srcTiles, dstTiles, bitwidth);
  return smem.getTotalOutDimSize() / smem.getInDimSize("reps");
}
```

`getSrcDstTiles` 根据 `TargetInfo` 的 `support*` 查询决定 tile 集（`Utility.cpp:42-76`）：

| 条件 | 加入的 tile |
|------|-----------|
| 始终 | `ld.shared` / `st.shared`（`laneAddr={0,1,2}`） |
| `supportStMatrix()` && bitwidth ≤ 32 | stmatrix（`laneContig={0,1}, laneAddr={2,3,4}`）加入 src |
| `supportLdMatrix()` && bitwidth ≤ 32 | ldmatrix 加入 dst |
| `support*` && bitwidth == 16 | ldmatrix.trans / stmatrix.trans（`laneContig={2,3,4}, laneAddr={0,1}`） |

---

### 4.3 两者对比：ConvertLayoutOp 是唯一分叉点

```
                          op 进入 scratchSizeGetter
                                    │
                    ┌───────────────┴───────────────┐
                    │         是 ConvertLayoutOp？    │
                    └───────────────┬───────────────┘
                          No        │         Yes
                    ┌───────────────┘         └──────────────────┐
                    ▼                                              ▼
         defaultAllocationAnalysisScratchSizeFn          cvtNeedsSharedMemory?
         （reduce/scan/gather/atomic/...）                      │
                                                    No ──→ 0    │ Yes
                                                    ┌───────────┴───────────┐
                                                    ▼                         ▼
                                          Generic 路径                  NVIDIA 路径
                                    optimalSwizzlingLdSt          optimalSwizzling
                                    （ld/st.shared only）    + getSrcDstTiles(targetInfo)
                                                    │                         │
                                                    └───────────┬─────────────┘
                                                                ▼
                                                    elems × bitwidth / 8
```

| 对比维度 | `defaultAllocationAnalysisScratchSizeFn` | `getNvidiaAllocationAnalysisScratchSizeFn` |
|---------|------------------------------------------|---------------------------------------------|
| **定义形式** | 普通函数，无参 | 工厂函数，返回捕获 `targetInfo` 的 lambda |
| **覆盖范围** | 所有需要 scratch 的 op 类型 | 仅覆盖 `ConvertLayoutOp` |
| **ConvertLayoutOp** | `optimalSwizzlingLdSt` | `optimalSwizzling` + `getSrcDstTiles` |
| **架构感知** | 无 | `TargetInfo.supportLdMatrix()` 等 |
| **其他 op** | 自行处理 | 委托 `defaultAllocationAnalysisScratchSizeFn` |
| **NVIDIA pass 实际行为** | 不直接使用 | lambda 内部对非 cvt op 调 default |

#### 同一条 `ConvertLayoutOp` 的两条路径

| 步骤 | Generic（default 内） | NVIDIA（lambda 内） |
|------|----------------------|-------------------|
| 1. 判断是否需要 smem | `cvtNeedsSharedMemory` | 相同 |
| 2. 提取 layout | `toLinearLayout` + `actionRemoveBroadcastedRegs` | 相同 |
| 3. 获取指令 tile | 内置在 `optimalSwizzlingLdSt` 中（仅 ld/st.shared） | `getSrcDstTiles(targetInfo, bitwidth)` |
| 4. 计算 swizzling | `optimalSwizzlingLdSt` | `optimalSwizzling`（枚举指令对，选 bank conflict 最小） |
| 5. 算元素数 | `totalOutDimSize / reps` | 相同 |
| 6. 转字节 | `elems × bitwidth / 8` | 相同 |

### 4.4 `TargetInfo` 在本 pass 中的角色

`TargetInfo` 是 NVIDIA 后端的架构抽象，但本 pass **只用其 `support*` 能力查询**，不用 `shuffleXor()` / `barrier()` 等 lowering 方法（那些属于 `tritongpu-to-llvmir`）。

| 方法 | 实现 | 本 pass 作用 |
|------|------|-------------|
| `supportLdMatrix()` | `CC >= 75` | 是否加入 ldmatrix tile |
| `supportStMatrix()` | `CC >= 90` | 是否加入 stmatrix tile |
| `supportLdStMatrixB8()` | `CC >= 100` | 8-bit transpose 可行性 |

| 架构 | CC | 可用 tile |
|------|-----|----------|
| Volta | sm_70 | 仅 ld/st.shared |
| Turing+ | sm_75+ | + ldmatrix, ldmatrix.trans |
| Hopper+ | sm_90+ | + stmatrix, stmatrix.trans |

### 4.5 为何 NVIDIA 必须覆盖 Generic？

NVIDIA 要利用 ldmatrix/stmatrix 获得最优性能，需要：
1. 查询架构是否支持这些指令
2. 将 tile 约束传给 `optimalSwizzling`
3. 在 scratch size 中考虑特殊指令的对齐要求

Generic 只用 `optimalSwizzlingLdSt`，无法感知这些约束。若用 Generic 版，轻则 bank conflict 未消除，重则 ldmatrix tile 约束不满足导致代码生成错误。

```
Generic 估算:  8192 字节  (仅 ld/st.shared tile 对齐)
NVIDIA 估算:  12288 字节  (含 ldmatrix tile 对齐要求)
              ↑ tile 约束不同 → scratch size 不同
```

---

## 5. Swizzling 与 Scratch Size 计算

Swizzling 通过 XOR 变换物理地址来减少 bank conflict，是 `ConvertLayoutOp` scratch size 计算的基础。实现位于 `triton/lib/Tools/GenericSwizzling.cpp`。

### 5.1 Bank Conflict 与 Swizzling 原理

NVIDIA shared memory 有 32 个 bank，每 bank 4 字节。同 warp 多线程访问同一 bank 会串行化。

```
no  swizzle: physical_addr = logical_addr
with swizzle: physical_addr = XOR(logical_addr, params)
```

编译器通过 `#ttg.swizzled_shared<{vec, perPhase, maxPhase, order}>` 指定 XOR 参数。

### 5.2 Swizzling 参数

| 参数 | 含义 |
|------|------|
| `vec` | 单线程向量化 load/store 元素数，上限 `128 / bitwidth` |
| `perPhase` | 每个 phase 行数，`perPhase × vec × bitwidth = 1024` bits |
| `maxPhase` | XOR key 种类数 |
| `order` | 维度遍历顺序（行/列优先） |

### 5.3 三条计算路径

#### 路径一：`optimalSwizzlingLdSt`（Generic ConvertLayoutOp）

```
getNumScratchElemsSwizzledCvt(src, dst)
  → toLinearLayout + actionRemoveBroadcastedRegs
  → optimalSwizzlingLdSt(src, dst, bitwidth)
  → scratch size = total_elements / reps
```

核心步骤：
1. **`vec`**：`intersectionBasis(regSrc, regDst)` 求公共寄存器基向量
2. **`perPhase`**：bank 段位数，`bankBits / (vec × bitwidth)`
3. **`maxPhase`**：segment 占剩余地址位
4. **`buildReps`**：拆分 reps 维度

地址位划分：

```
┌──────────────────────────────────────────────┐
│ [vector bits] [bank/perPhase bits] [segment] │
│ vec=2^v       perPhase=2^b        maxPhase=2^s │
└──────────────────────────────────────────────┘
```

#### 路径二：`optimalSwizzling`（NVIDIA ConvertLayoutOp）

```
getNumScratchElemsSwizzledCvt(src, dst, targetInfo)
  → getSrcDstTiles(targetInfo, bitwidth)
  → optimalSwizzling(src, dst, srcTiles, dstTiles, bitwidth)
  → scratch size
```

比路径一多一个**指令选择**步骤：

1. 枚举可行指令对 `(instrSrc, instrDst)`
2. 对每对调用 `optimalSwizzlingTile` 检查兼容性
3. 用 `bankConflicts` 评估，选 `read + write` 最小的组合
4. 无兼容对时回退 `optimalSwizzlingLdSt`

| 维度 | `optimalSwizzlingLdSt` | `optimalSwizzling` |
|------|----------------------|-------------------|
| vbasis 来源 | `intersectionBasis` | `optimalSwizzlingTile`（指令约束） |
| tile 来源 | 从 lane 基向量推导 | `getSrcDstTiles`（ldmatrix/stmatrix） |
| 用途 | Generic | NVIDIA |

#### 路径三：`NVMMASharedEncodingAttr`（Hopper MMA）

```cpp
int getVec()      { return 128 / elementBitWidth; }
int getPerPhase() { return 128 / swizzlingByteWidth; }
int getMaxPhase() { return swizzlingByteWidth / 16; }
```

`swizzlingByteWidth` 由连续维度大小决定（128/64/32/0）。

### 5.4 `bankConflicts` 评估

```cpp
std::pair<int, int> bankConflicts(tileSrc, tileDst, smem) {
  int write = 1 << intersectionBasis(segmentBases, tileSrc, rank).size();
  int read  = 1 << intersectionBasis(segmentBases, tileDst, rank).size();
  return {read - 1, write - 1};
}
```

交集越大 → bank conflict 越严重。`optimalSwizzling` 遍历所有指令对，选总 conflict 最小的。

### 5.5 LinearLayout 前置知识

Swizzling 基于 `LinearLayout` 的基向量（basis vector）：

| `flatten` 调用 | 含义 |
|-------------|------|
| `flatten(src, "register")` | 输出地址哪些位由寄存器索引决定 |
| `flatten(src, "lane")` | 由 lane ID 决定 |
| `flatten(src, "warp")` | 由 warp ID 决定 |

`intersectionBasis(b1, b2, dim)` 求两组基向量的交集，用于确定可安全向量化的维度。

---

## 6. IR 标注：`attachAllocationSizeAndOffsetAttr`

两个 pass 最终都调用此函数（`AllocateSharedMemoryUtility.h`）：

```cpp
void attachAllocationSizeAndOffsetAttr(ModuleOp mod, ModuleAllocation &allocation) {
  mod.walk([&](FunctionOpInterface funcOp) {
    auto *funcAlloc = allocation.getFuncData(funcOp);
    funcOp.walk([&](Operation *op) {
      int offset = -1;
      auto oBufferId = funcAlloc->getBufferId(op);
      if (oBufferId != InvalidBufferId)
        offset = funcAlloc->getOffset(oBufferId);
      else if (op->getNumResults() == 1) {
        auto vBufferId = funcAlloc->getBufferId(op->getResult(0));
        if (vBufferId != InvalidBufferId)
          offset = funcAlloc->getOffset(vBufferId);
      }
      if (offset != -1)
        op->setAttr("allocation.offset", IntegerAttr::get(i32, offset));
    });
  });
  mod->setAttr("ttg.shared", IntegerAttr::get(i32, allocation.getSharedMemorySize()));
}
```

---

## 7. 测试

| 文件 | 内容 |
|------|------|
| `triton/test/Conversion/allocate_shared_memory.mlir` | Generic pass：`ttg.shared` 和 `allocation.offset` 属性 |
| `triton/test/Analysis/test-allocation.mlir` | `test-print-allocation`：图着色、warp specialize、各种 layout 对齐 |

---

## 8. 设计要点

1. **全局分析**：`ModuleOp` 级别，跨 call graph 复用内存
2. **图着色**：活跃范围不重叠的 buffer 共享同一块内存
3. **可插拔 `scratchSizeGetter`**：backend 定制 scratch 大小，不改核心框架
4. **架构感知 swizzling**：NVIDIA 通过 `TargetInfo` + `getSrcDstTiles` 注入 ldmatrix/stmatrix tile
5. **统一 IR 接口**：所有 backend 输出相同的 `allocation.offset` / `ttg.shared` 属性

---

## 附录：源码索引

| 组件 | 路径 |
|------|------|
| Generic pass | `triton/lib/Conversion/TritonGPUToLLVM/AllocateSharedMemory.cpp` |
| NVIDIA pass | `triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp` |
| ModuleAllocation | `triton/include/triton/Analysis/Allocation.h` |
| AllocationAnalysis | `triton/lib/Analysis/Allocation.cpp` |
| getSrcDstTiles | `triton/lib/Conversion/TritonGPUToLLVM/Utility.cpp` |
| Swizzling 算法 | `triton/lib/Tools/GenericSwizzling.cpp` |
| TargetInfo | `triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h` |
| IR 标注 | `triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h` |
| Pipeline 注册 | `triton/third_party/nvidia/backend/compiler.py` |
