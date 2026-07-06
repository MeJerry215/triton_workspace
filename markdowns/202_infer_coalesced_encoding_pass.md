# InferCoalescedEncodings Pass 分析

## 一、概述

`GluonInferCoalescedEncodingsPass` 是 Triton Gluon 方言中的一个 Module 级 Pass，其核心目标是将张量上的**符号化 `CoalescedEncodingAttr`**（一种占位/标记编码）解析为**具体的 GPU `BlockedEncodingAttr`**（基于 AxisInfo 分析的访存连续性推导出的最优布局）。

> **Pass 名称（CLI）**：`gluon-infer-coalesced-encodings`
>
> **依赖方言**：`TritonGPUDialect`
>
> **操作粒度**：`mlir::ModuleOp`

---

## 二、功能详解

### 2.1 整体流程

```
          ModuleOp
             │
             ▼
  ┌─────────────────────────┐
  │  inferCoalescedLayout   │  ← 核心推断：为每个 function 推导具体 layout
  │   (ModuleAxisInfoAnalysis) │
  └─────────┬───────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │  doubleCheckEncodings   │  ← 校验：确保所有符号化 encoding 都被解析
  └─────────────────────────┘
             │
             ▼
       ModuleOp (已解析完毕)
```

#### `runOnOperation()` 实现

```cpp
void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (failed(inferCoalescedLayout(moduleOp)))   // 步骤 1：推断 layout
      return signalPassFailure();

    if (failed(doubleCheckEncodings(moduleOp, isCoalescedEncodingTensorType)))  // 步骤 2：校验
      return signalPassFailure();
}
```

两个步骤是串行的：先推断，后校验。任何一步失败都会导致 pass 失败。

---

### 2.2 核心步骤 1：`inferCoalescedLayout`

| 文件 | `InferCoalescedEncodings.cpp`，第 41-95 行 |
|------|--------------------------------------------|
| 签名 | `LogicalResult inferCoalescedLayout(ModuleOp &mod)` |

#### 算法分解

```
for each function in module:
    │
    ├── Step 1: 查找所有带 CoalescedEncoding 的 load/store 操作
    │            └─ 只处理 element type 为 ptr 的 tensor（即 tensor<ptr<T>>）
    │            └─ 对每个匹配 op，调用 buildCoalescedEncoding 生成具体 BlockedEncodingAttr
    │            └─ 将 (operand_value → concrete_encoding) 收集为 seedEncodings
    │
    └── Step 2: 传播具体 layout（forward + backward）
                 └─ 以 seedEncodings 为种子，调用 inferLayout
                 └─ 沿 SSA 图传播，更新所有匹配张量的类型
```

#### 2.2.1 种子编码生成（Seed Encoding）

**第 55-82 行**：遍历 function 中的所有操作，对每个满足以下条件的操作生成种子编码：

1. **是访存操作**：通过 `getMemAccessPtr(op)` 获取到有效的 ptr（支持 `LoadOp`、`StoreOp`、`AtomicRMWOp`、`AtomicCASOp`、`AsyncCopyGlobalToLocalOp`）
2. **元素类型为指针**：ptr 的类型是 `RankedTensorType`，且其元素类型是 `PointerType`
3. **标记了 CoalescedEncoding**：ptr 的 tensor 编码是 `gluon::CoalescedEncodingAttr`

满足条件后，调用 `buildCoalescedEncoding` 为该操作生成最优的 `BlockedEncodingAttr`：

```cpp
auto layout = ttg::buildCoalescedEncoding(
    mod.getContext(), axisInfoAnalysis, curr, numWarps, threadsPerWarp,
    ctaLayout, shapePerCTA);
```

> **重要**：虽然函数名是 `buildCoalescedEncoding`，但返回的实际是 `BlockedEncodingAttr`（一个具体的 GPU 编码），而非符号化的 `CoalescedEncodingAttr`。

#### 2.2.2 `buildCoalescedEncoding` 的工作原理

| 文件 | `triton/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp`，第 16-94 行 |
|------|--------------------------------------------|

**目标**：为访存操作生成最大化合并访问的 `BlockedEncodingAttr`。

步骤：

1. **确定 axes 优先级（order）**：
   - 使用 `AxisInfoAnalysis` 获取 ptr 各维度的 contiguity（连续访问程度）
   - 按 contiguity 降序排列 axes，得到 `order`
   - contiguity 越高说明该维度在内存中越连续，应优先分配给线程

2. **收集同一 slice 中 shape 和 order 相同的所有访存操作**：
   - 使用 `getSlice` 收集周边操作
   - 筛选出 `memAccessesSameOrder` 的访存操作
   - 目的是在这些操作之间取最大的 `perThread`，以捕捉最宽的向量化机会

3. **计算 sizePerThread**：
   - 对每个操作计算 `getNumElementsPerThread`
   - 取所有操作的最大值作为 `perThread`
   - 上限限制：`perThread <= numElems / numThreads`（至少为 1）
   - 对非 load 操作（store/atomic）：重新限制为该操作自身的 `perThread`，避免写入时的 gap 问题（warp 级 store 宽度限制为 128 位）

4. **组装 `BlockedEncodingAttr`**：
   - `sizePerThread[order[0]] = perThread`
   - 其他维度 `sizePerThread` 设为 1

#### `argSort` 是什么？举例

`argSort` 定义在 `Utility.cpp`，对**下标**做稳定降序排序，而不是对 contiguity 数值本身排序：

```cpp
SmallVector<unsigned, 4> argSort(const SmallVector<int64_t> &arr) {
  SmallVector<unsigned, 4> ret(arr.size());
  std::iota(ret.begin(), ret.end(), 0);           // 先填 [0, 1, 2, ...]
  std::stable_sort(ret.begin(), ret.end(),
                   [&](unsigned x, unsigned y) { return arr[x] > arr[y]; });
  return ret;
}
```

返回值 `order` 的含义：**按 contiguity 从高到低的维度编号排列**，即 `order[0]` 是最值得做向量化/合并访存的那一维。

| contiguity（按 dim 0, 1, …） | `order` 结果 | 含义 |
|-------------------------------|--------------|------|
| `[1, 128]`（2D，行主序 tile） | `[1, 0]` | dim 1 最连续，线程优先铺在 dim 1 |
| `[128, 1]` | `[0, 1]` | dim 0 最连续 |
| `[4, 16, 1]`（3D） | `[1, 0, 2]` | dim 1 > dim 0 > dim 2 |
| `[8, 8]`（相同） | `[0, 1]` | 稳定排序，相等时保持原下标顺序 |

对应 2D kernel 测试（`128×256`，行主序）：`contiguity = [1, 128]` → `order = [1, 0]`，与最终 `#blocked<{order = [1, 0], sizePerThread = [1, 4]}>` 一致。

#### 为什么 `perThread` 用 `max` 而不是 `min`？

`memAccessesSameOrder` 里的多个 load/store **最终会共享同一个 `BlockedEncodingAttr`**（同一 slice、同一 shape、同一 order）。布局只能选一套，所以要满足**所有**相关访存里**最贪心**的那一个：

- **`max`**：取最宽的向量化宽度，让 layout 至少能覆盖 slice 里“最需要”宽向量 load 的那个 op；较窄的 op 仍然合法（只是没有用到全部宽度）。
- **`min`**：会被最保守的 op 拖累，其他本可以 128-bit 向量 load 的 op 会退化成更窄访问，合并访存变差。

但 `max` 不能无脑用于 store，因此后面还有两道 clamp：

```cpp
perThread = std::min(perThread, std::max(numElems / numThreads, 1));  // 线程数上限
if (!dyn_cast<triton::LoadOp>(op))
  perThread = std::min(perThread, getNumElementsPerThread(op, ...));  // store 防 gap
```

对 **store/atomic**：warp 级向量写最宽 128 bit；若 `perThread` 比当前 store 自身算出来的值更大，会导致 warp 内写入出现“空洞”（gap），性能反而更差。load 可以容忍（L1 会掩盖），所以只对非 load 再缩回 `min`。

#### 为什么 `perThread` 是 `unsigned` 标量，而不是多维向量？

1. **类型**：表示“每个线程沿**某一维**连续持有的元素个数”，天然非负，与 `getNumElementsPerThread` 的返回值一致。
2. **只向量化 `order[0]`**：`getNumElementsPerThread` 只看 `order[0]`（contiguity 最高的维）上的 `divisibility`、`contiguity` 和 128-bit 上限，不算其他维。
3. **多维信息在 `sizePerThread` 里展开**：标量 `perThread` 只填最快维，其余维为 1：

```cpp
SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
sizePerThread[order[0]] = perThread;
```

这是 Blocked layout 的常见做法：**合并访存主要发生在内存上最连续的那一维**；其他维由 `threadsPerWarp` / `warpsPerCTA` 的线程网格覆盖，而不是在同一线程内再堆更多元素。

#### 2.2.3 Layout 传播（`inferLayout`）

| 文件 | `triton/lib/Dialect/Gluon/Transforms/InferLayoutUtils.cpp`，第 110-222 行 |
|------|--------------------------------------------|

**签名**：
```cpp
LogicalResult inferLayout(
    FuncOp func,
    llvm::function_ref<bool(Type)> typeCheck,
    const SmallVector<std::pair<Value, Attribute>> &seedEncodings)
```

**算法**：基于 worklist 定点迭代（fixed-point iteration）的 SSA 图传播。

1. **前向传播（Forward Propagation）**：
   - 从种子值出发，沿着 SSA 使用链（use-def chain）向前传播
   - 对每个使用操作，调用 `inferDstEncoding(op, encoding)` 推断目标编码
   - 处理 `scf::ForOp` / `scf::WhileOp` / `scf::YieldOp`：通过 tied block argument 传递

2. **后向传播（Backward Propagation）**：
   - 从种子值出发，向定义操作传播
   - 对定义操作，调用 `inferSrcEncoding(definingOp, encoding)` 推断源编码
   - 同样处理 `scf` 控制流操作的 tied argument

3. **冲突处理**：
   - `LayoutInfo` 结构体包含 `encoding` 和 `mayVary` 标志
   - `JoinOp`、`SplitOp`、`ReshapeOp`、`CatOp`、`TransOp` 等操作会设置 `mayVary = true`
   - 遇到冲突时，优先选择非 `mayVary`（即确定性）的编码
   - 真正的冲突（不兼容编码）会通过 `op->emitOpError` 报错

4. **类型更新**：
   - 传播稳定后，通过 `val.setType(existingTy.cloneWithEncoding(info.encoding))` 更新所有值的类型
   - 对 `arith::ConstantOp`，还同步更新 splat attribute

---

### 2.3 核心步骤 2：`doubleCheckEncodings`

| 文件 | `triton/lib/Dialect/Gluon/Transforms/InferLayoutUtils.cpp`，第 224-249 行 |
|------|--------------------------------------------|

**目的**：校验所有符号化 `CoalescedEncodingAttr` 是否已被完全解析。

实现：

```cpp
LogicalResult doubleCheckEncodings(ModuleOp &mod,
    llvm::function_ref<bool(Type)> typeCheck) {
  // 1. 检查所有操作的返回值类型
  // 2. 检查所有 block argument 的类型
  // 如果仍有类型匹配 typeCheck（即仍带有 CoalescedEncodingAttr），则报错
}
```

具体来说是两次 Walk：

- **操作结果 Walk**：如果任何操作的结果类型仍然含有 `CoalescedEncodingAttr`，报 "Failed to infer return type"
- **Block 参数 Walk**：如果任何 block 的参数类型仍然含有 `CoalescedEncodingAttr`，报 "Failed to infer block argument type"

---

### 2.4 辅助函数 `isCoalescedEncodingTensorType`

```cpp
bool isCoalescedEncodingTensorType(Type ty) {
  auto tensorTy = dyn_cast<RankedTensorType>(ty);
  return tensorTy && isa<gluon::CoalescedEncodingAttr>(tensorTy.getEncoding());
}
```

判断一个 `Type` 是否为带有 `CoalescedEncodingAttr` 的 `RankedTensorType`。用作 `typeCheck` 回调，同时传递给 `inferLayout` 和 `doubleCheckEncodings`，确保这两个步骤操作的是同一类张量。

---

### 2.5 `CoalescedEncodingAttr` 定义

| 文件 | `triton/include/triton/Dialect/Gluon/IR/GluonAttrDefs.td` |
|------|--------------------------------------------|

**Tablegen 定义**：

```tablegen
def Gluon_CoalescedEncodingAttr : AttrDef<Gluon_Dialect, "CoalescedEncoding"> {
  let mnemonic = "coalesced_encoding";
  let description = [{
    An encoding that is optimized for load/store performance.
  }];
}
```

这是一个**符号化/标记性**的 encoding 属性，本身不包含具体的 layout 信息。它告诉后续 pass："这个张量的 layout 应该为了访存合并而优化"，由 `GluonInferCoalescedEncodingsPass` 将其解析为具体的 `BlockedEncodingAttr`。

与之对比的还有 `Gluon_AutoEncodingAttr`（mnemonic: `auto_encoding`），是一个更通用的自动编码标记。

---

### 2.6 `getMemAccessPtr` 辅助函数

| 文件 | `triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp`，第 101-113 行 |
|------|--------------------------------------------|

```cpp
Value getMemAccessPtr(Operation *op) {
  if (auto ld = dyn_cast<triton::LoadOp>(op))   return ld.getPtr();
  if (auto atomic = dyn_cast<triton::AtomicRMWOp>(op)) return atomic.getPtr();
  if (auto atomic = dyn_cast<triton::AtomicCASOp>(op)) return atomic.getPtr();
  if (auto copy = dyn_cast<triton::gpu::AsyncCopyGlobalToLocalOp>(op)) return copy.getSrc();
  if (auto store = dyn_cast<triton::StoreOp>(op))  return store.getPtr();
  return nullptr;
}
```

支持的访存操作类型：

| 操作 | 说明 |
|------|------|
| `triton::LoadOp` | 全局/共享内存加载 |
| `triton::StoreOp` | 全局/共享内存存储 |
| `triton::AtomicRMWOp` | 原子 RMW 操作 |
| `triton::AtomicCASOp` | 原子 CAS 操作 |
| `triton::gpu::AsyncCopyGlobalToLocalOp` | 异步全局到局部拷贝 |

不匹配任何类型时返回 `nullptr`。

---

## 三、Pass 在 Pipeline 中的位置

```
  之前 Pass（如 Inline）
        │
        ▼
  符号化 CoalescedEncodingAttr 标记张量
  由用户或之前 Pass 通过 set_auto_layout 设置
        │
        ▼
  ┌─────────────────────────────────────┐
  │  GluonInferCoalescedEncodingsPass   │  ← 当前 Pass
  │  将 CoalescedEncoding → BlockedEncoding │
  └─────────────────────────────────────┘
        │
        ▼
  具体 BlockedEncodingAttr 已就位
        │
        ▼
  后续 Pass（如 ResolveAutoLayoutPass）
```

此 Pass 处理的是如下的 IR 模式：

```
// Pass 前：gl.set_auto_layout(val, gl.CoalescedLayout())
//                ↓
// Pass 后：gl.set_auto_layout(val, <concrete BlockedEncodingAttr>)
```

其中 `set_auto_layout` 操作相当于一个布局锚点，`inferCoalescedLayout` 会填充具体 layout，后续的 `ResolveAutoLayoutPass` 处理剩下的传播。

---

## 四、IR 示例（来自实际测试 dump）

以下 IR 来自 `ops/pass/202_infer_coalesced_encoding_pass.py` 中 2D kernel 的真实 dump。
测试参数：XBLOCK=128, YBLOCK=256, numWarps=4, threadsPerWarp=32。

### Before（Pass 执行前）

```
// -----// IR Dump Before GluonInferCoalescedEncodingsPass //----- //
```

此时所有张量的 encoding 仍为符号化的 `#gluon.coalesced_encoding`，不包含具体布局信息：

```mlir
#loc = loc(unknown)
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                  ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @coalesced_2d_kernel(%in_ptr: !tt.ptr<f32>,
                                      %out_ptr: !tt.ptr<f32>,
                                      %xnumel: i32, %ynumel: i32,
                                      %xstride_in: i32, %xstride_out: i32) {
    %pid_x = tt.get_program_id x : i32
    %pid_y = tt.get_program_id y : i32
    %indices_x = arith.constant 128 : i32
    %indices_x_0 = arith.muli %pid_x, %indices_x : i32
    %indices_x_1 = tt.make_range {end = 128 : i32, start = 0 : i32}
        : tensor<128xi32, #gluon.coalesced_encoding>
    %indices_x_2 = tt.splat %indices_x_0 : i32
        -> tensor<128xi32, #gluon.coalesced_encoding>
    %indices_x_3 = arith.addi %indices_x_2, %indices_x_1
        : tensor<128xi32, #gluon.coalesced_encoding>

    %indices_y = arith.constant 256 : i32
    %indices_y_4 = arith.muli %pid_y, %indices_y : i32
    %indices_y_5 = tt.make_range {end = 256 : i32, start = 0 : i32}
        : tensor<256xi32, #gluon.coalesced_encoding>
    %indices_y_6 = tt.splat %indices_y_4 : i32
        -> tensor<256xi32, #gluon.coalesced_encoding>
    %indices_y_7 = arith.addi %indices_y_6, %indices_y_5
        : tensor<256xi32, #gluon.coalesced_encoding>

    %in_offsets = tt.expand_dims %indices_x_3 {axis = 1 : i32}
        : tensor<128xi32, #gluon.coalesced_encoding>
      -> tensor<128x1xi32, #gluon.coalesced_encoding>
    // ... 后续所有值都是 #gluon.coalesced_encoding ...

    %value = tt.splat %in_ptr : !tt.ptr<f32>
        -> tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    %value_28 = tt.addptr %value, %in_offsets_13
        : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>,
          tensor<128x256xi32, #gluon.coalesced_encoding>
    %value_29 = tt.load %value_28, %mask_27
        : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    %value_30 = math.sin %value_29
        : tensor<128x256xf32, #gluon.coalesced_encoding>
    %value_32 = arith.maxnumf %value_30, %value_31
        : tensor<128x256xf32, #gluon.coalesced_encoding>
    %1 = tt.addptr %0, %out_offsets_19
        : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>,
          tensor<128x256xi32, #gluon.coalesced_encoding>
    tt.store %1, %value_32, %mask_27
        : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    tt.return
  }
}
```

关键观察：**整个 function 中每个张量的 encoding 都是 `#gluon.coalesced_encoding`**，这是一个纯标记，没有任何线程映射信息。

### After（Pass 执行后）

```
// -----// IR Dump Before GluonResolveAutoEncodingsPass //----- //
```

`GluonInferCoalescedEncodingsPass` 运行后，`#gluon.coalesced_encoding` 已被替换为具体的 `#blocked` 编码：

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 4],
                         threadsPerWarp = [1, 32],
                         warpsPerCTA = [2, 2],
                         order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                  ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @coalesced_2d_kernel(%in_ptr: !tt.ptr<f32>,
                                      %out_ptr: !tt.ptr<f32>,
                                      %xnumel: i32, %ynumel: i32,
                                      %xstride_in: i32, %xstride_out: i32) {
    // 1D 张量推导为 #ttg.slice — 父 layout 的投影
    %indices_x_1 = tt.make_range {end = 128 : i32, start = 0 : i32}
        : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %indices_x_3 = arith.addi %indices_x_2, %indices_x_1
        : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>

    %indices_y_5 = tt.make_range {end = 256 : i32, start = 0 : i32}
        : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %indices_y_7 = arith.addi %indices_y_6, %indices_y_5
        : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>

    // 2D 张量推导为完整的 #blocked
    %in_offsets = tt.expand_dims %indices_x_3 {axis = 1 : i32}
        : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      -> tensor<128x1xi32, #blocked>
    %in_offsets_11 = tt.broadcast %in_offsets_9
        : tensor<128x1xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %in_offsets_13 = arith.addi %in_offsets_11, %in_offsets_12
        : tensor<128x256xi32, #blocked>

    // load/store 全部使用 #blocked
    %value = tt.splat %in_ptr : !tt.ptr<f32>
        -> tensor<128x256x!tt.ptr<f32>, #blocked>
    %value_28 = tt.addptr %value, %in_offsets_13
        : tensor<128x256x!tt.ptr<f32>, #blocked>, tensor<128x256xi32, #blocked>
    %value_29 = tt.load %value_28, %mask_27
        : tensor<128x256x!tt.ptr<f32>, #blocked>
    %value_30 = math.sin %value_29
        : tensor<128x256xf32, #blocked>
    %value_32 = arith.maxnumf %value_30, %value_31
        : tensor<128x256xf32, #blocked>
    tt.store %1, %value_32, %mask_27
        : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
```

### Before/After 对比关键点

| 维度 | Before（符号化） | After（具体化） |
|------|-----------------|-----------------|
| **1D 张量**（arange） | `#gluon.coalesced_encoding` | `#ttg.slice<{dim = 1/0, parent = #blocked}>` |
| **2D 张量**（offsets、mask、value） | `#gluon.coalesced_encoding` | `#blocked` |
| **推导结果** | 纯标记，无布局信息 | `sizePerThread = [1, 4]`，`order = [1, 0]` |
| **线程映射** | 无 | `threadsPerWarp = [1, 32]`，`warpsPerCTA = [2, 2]` |

具体的 `#blocked` 编码含义：
- **`sizePerThread = [1, 4]`**：每个线程在 维度 1（列/最内层）上处理 4 个元素 → 向量化访存宽度为 4
- **`order = [1, 0]`**：列优先（column-major），维度 1（最内层 stride）最先分配 → 确保连续内存访问
- **`warpsPerCTA = [2, 2]`**：4 个 warp 在 2x2 的网格中覆盖 128x256 的 tile
---

> **关于 `ModuleAxisInfoAnalysis`**：本 Pass 的核心分析组件 `ModuleAxisInfoAnalysis` 的详细原理（三个核心属性、highestPowOf2Divisor、数据流传播、运行时连续性计算、`tt.divisibility` 注解等）请参见 [`202.5_moduleaxisinfo_analysis.md`](202.5_moduleaxisinfo_analysis.md)。

## 五、关键数据结构总结

| 数据结构 | 作用 |
|----------|------|
| `AxisInfo` | 每个 SSA 值的轴属性（contiguity / divisibility / constancy 三个 `SmallVector`） |
| `CoalescedEncodingAttr` | 符号化标记编码，表示"需要优化为合并访存 layout" |
| `BlockedEncodingAttr` | 具体的 GPU 线程块布局，指定 `sizePerThread`、`threadsPerWarp`、`warpsPerCTA`、`order` |
| `LayoutInfo` | 传播过程中的中间数据结构，包含 `encoding` 和 `mayVary` 标志 |
| `seedEncodings` | 种子编码集合：`SmallVector<std::pair<Value, Attribute>>`，将 SSA 值映射到具体 layout |

---

## 六、设计要点与约束

1. **单 CTA 限制**：当前 `getDefaultCTALayout` 只支持 `numCTAs == 1`，多 CTA 场景尚未支持（留了 `TODO`）。

2. **仅处理 ptr tensor**：Pass 只处理 element type 为 `PointerType` 的 tensor（即 `tensor<ptr<T>>`），因为只有指针张量的访存 layout 需要 coalesced 优化。普通数据张量（`tensor<f32>`）不受影响。

3. **不跨 `set_auto_layout` 边界传播**：后向传播不会跨越 `set_auto_layout` 边界，即每个 `set_auto_layout` 操作定义了自己的传播域。

4. **定点迭代**：使用 worklist 进行定点迭代传播，确保编码在 SSA 图中收敛到一致状态。

5. **校验严谨**：`doubleCheckEncodings` 确保所有符号化编码都被解析后才声明成功，避免遗漏。

---

## 七、Pass 注册

```tablegen
def GluonInferCoalescedEncodingsPass
    : Pass<"gluon-infer-coalesced-encodings", "mlir::ModuleOp"> {
  let summary = "Infer coalesced encodings based on axis analysis";
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}
```

| 字段 | 值 |
|------|-----|
| Pass 名称 | `gluon-infer-coalesced-encodings` |
| 操作类型 | `mlir::ModuleOp` |
| 依赖方言 | `TritonGPUDialect` |

---

*本文档对应源代码：[InferCoalescedEncodings.cpp](../../triton/lib/Dialect/Gluon/Transforms/InferCoalescedEncodings.cpp)*
