# `gluon-inline` Pass 分析

```python
# triton/third_party/nvidia/backend/compiler.py:320-325
def gluon_to_ttgir(self, src, metadata, options, capability):
    mod = src
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.gluon.add_inliner(pm)           # ── 第 1 个 pass
    # ...
```

---

## 1. 为什么必须先 Inline？

`gluon_to_ttgir` 的核心任务是将 **gluon dialect** 的 IR 降为 **TritonGPU dialect** 的 IR。在这个过程中：

- `CoalescedLayout()` 等 encoding 是 **函数内** 的约定
- 如果 helper 函数没有被内联，后续的 layout 推断会跨函数边界失效
- **因此在管线入口必须先内联所有 `@gluon.jit` 调用**

---

## 2. 关联源码

| 组件 | 路径 | 作用 |
|------|------|------|
| Pipeline 定义 | `triton/third_party/nvidia/backend/compiler.py:320-337` | 注册 `gluon_to_ttgir` pass 序列 |
| Python 绑定 | `triton/python/src/passes.cc:120` | `ADD_PASS_WRAPPER_0("add_inliner", gluon::createGluonInline)` |
| TableGen 定义 | `triton/include/triton/Dialect/Gluon/Transforms/Passes.td:34` | `def GluonInline: Pass<"gluon-inline">` |
| C++ 实现 | `triton/lib/Dialect/Gluon/Transforms/Inline.cpp` | 调用 MLIR 的 `createInlinerPass` + `GluonSimplifyControlFlow` |
| 测试脚本 | `ops/pass/201_inline_pass.py` | 本分析的测试源码 |
| IR dump | `ops/pass/dump.log` | `MLIR_ENABLE_DUMP=1` 产出 |

### 2.1 C++ 实现核心逻辑 (Inline.cpp)

```cpp
void Inline::runOnOperation() {
  mlir::PassManager pm(&getContext());
  pm.addPass(createInlinerPass(/*opPipelines=*/{}, [](OpPassManager &pm) {
    pm.addPass(gluon::createGluonSimplifyControlFlow());
  }));
  if (failed(pm.run(getOperation())))
    return signalPassFailure();
}
```

它使用 MLIR 的标准 `InlinerPass`，并为被内联函数的 body 注册了 `GluonSimplifyControlFlow` 作为后处理 pipeline。

### 2.2 子 pipeline：GluonSimplifyControlFlow (SimplifyControlFlow.cpp)

`GluonSimplifyControlFlow` 是 inline 的**附带清理 pass**，作为 lambda 传给 `createInlinerPass` 的 `opPipelines` 参数，这意味着它仅在被内联的函数体上运行。

```cpp
void SimplifyControlFlow::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(&getContext());

  // Populate `scf` and `cf` canonicalizers.
  ctx->getLoadedDialect<scf::SCFDialect>()->getCanonicalizationPatterns(
      patterns);
  ctx->getLoadedDialect<cf::ControlFlowDialect>()->getCanonicalizationPatterns(
      patterns);
  for (mlir::RegisteredOperationName op :
       ctx->getRegisteredOperationsByDialect(
           scf::SCFDialect::getDialectNamespace()))
    op.getCanonicalizationPatterns(patterns, ctx);
  for (mlir::RegisteredOperationName op :
       ctx->getRegisteredOperationsByDialect(
           cf::ControlFlowDialect::getDialectNamespace()))
    op.getCanonicalizationPatterns(patterns, ctx);
  populateForOpDeadArgumentElimination(patterns);

  GreedyRewriteConfig config;
  // This is intended to run before AutoLayouts are resolved, in which case
  // CSEing constants can lead to additional layout conflicts.
  config.enableConstantCSE(false);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
}
```

#### Canonicalization Pattern 是什么

Canonicalization（规范化）是 MLIR 中的一种通用 IR 化简机制。每个 op 可以注册一组 **RewritePattern**，当 pass 运行时，框架会贪心地（greedily）反复应用这些 pattern，直到 IR 不再变化。

本质上是将 IR 中的局部模式匹配替换为更简单的等价形式：

```
IR before                  Canonicalization                  IR after
──────                     ─────────────────                  ────────
%c0 = arith.constant 0 : i32
%r = arith.addi %x, %c0         →              (直接替换为 %x)
```

常见的 canonicalization 例子：

| 模式 | Before | After |
|------|--------|-------|
| **加 0** | `%r = arith.addi %x, %c0` | 直接替换为 `%x` |
| **乘 1** | `%r = arith.muli %x, %c1` | 直接替换为 `%x` |
| **折叠常量** | `%r = arith.addi %c1, %c2` | `%c3 = arith.constant 3` |
| **恒真分支** | `scf.if %true { ... } else { ... }` | 只保留 then 分支 |
| **死参数消除** | `scf.for %i = ... iter_args(%arg = ...)` (arg 未用) | 去掉该参数 |

`SimplifyControlFlow` 所做的，就是把 **scf** 和 **cf** 两个控制流 dialect 下所有 op 的"化简说明书"（即 canonicalization pattern）收集到一起，批量跑一遍贪心化简。

它做了四件事：

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 加载 scf 和 cf dialect 的 canonicalization 模式 | 化简结构化循环/分支 (`scf.for`, `scf.if`) 和底层分支 (`cf.br`, `cf.cond_br`) |
| 2 | 逐个 op 加载 canonicalization 模式 | 覆盖每个 op 注册到自身的化简规则，粒度更细 |
| 3 | `populateForOpDeadArgumentElimination` | 消除 `scf.for` 中未被循环体使用的迭代参数，减小 IR 体积 |
| 4 | **禁掉 Constant CSE** | 关键约束：此时处于 layout 解析之前，合并常数会导致不同 layout 域的常数冲突 |

注意步骤 1 和步骤 2 的关系：MLIR 的 `Dialect::getCanonicalizationPatterns` 默认实现已经会遍历该 dialect 下所有注册的 op 并收集它们的 canonicalization pattern。这里先调 dialect 级的方法，又手动 `for` 循环一遍，两者在功能上是**重叠的** — 作者可能出于显式安全的意图保留了双重加载。

#### populateForOpDeadArgumentElimination — scf.for 的死参数消除

第 3 步 `populateForOpDeadArgumentElimination(patterns)` 注册的是 `ForOpDeadArgElimination` 这个 pattern（实现在 `triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp:1245-1370`）。它专门处理 `scf.for` 的 **loop-carried values（迭代传递参数）** 中那些从未被使用的死参数。

##### scf.for 的 IR 结构速览

先看一个带迭代参数的 `scf.for` 的文本形式：

```
%result = scf.for %iv = %lb to %ub step %step
    iter_args(%arg = %init) -> i32 {
  %use = arith.addi %arg, %c1        // 使用 %arg
  scf.yield %use                     // 传给下一轮
}
```

- `%iv` — 循环变量（induction variable），每轮迭代自动 + step
- `iter_args(%arg = %init)` — **迭代传递参数**，类似 `[schedule]` 里的 `carry`：初始值是 `%init`，每轮 yield 的新值传给下一轮
- `scf.yield` — 循环体的终结 op，产生的值作为下一轮的 `iter_args` 和最终的循环结果

##### scf.for 的 Block 参数布局（关键）

`scf.for` 内部的 Block 参数索引不是从 0 开始对应 iter_arg 的，因为 **`%iv` 占着 index 0**：

```
%result = scf.for %iv = %lb to %ub step %step iter_args(%a = %init0, %b = %init1) -> (i32, i32) {
  // Block 参数索引：
  //   [0] = %iv           ← 循环变量（induction variable）
  //   [1] = %a            ← 第 0 个 iter_arg
  //   [2] = %b            ← 第 1 个 iter_arg
  scf.yield %y0, %y1 : i32, i32
  // yield operand 索引：
  //   [0] = %y0           ← 对应 iter_arg %a
  //   [1] = %y1           ← 对应 iter_arg %b
}
```

注意 yield operand 和 BlockArgument **不是同一个东西**。yield 的是当前轮的"计算结果"，BlockArgument 是当前轮收到的"输入值"。它们的关系是：

```
上一轮 yield[0] → 本轮 block[1](%a) → 计算 → 本轮 yield[0] → 下轮 block[1](%a) → ...
```

所以同样的"第 0 个 iter_arg"在 yield 和 Block 中的索引相差 1：

```
yield operand index = 0                    ← 从 0 数 iter_arg
BlockArgument index = 0 + 1 = 1           ← Block 参数多了一个 %iv 在头部
                           ↑ 跳过 %iv
```

代码中的体现：

```cpp
// deadArg 存的是 yield operand 的 index（0-based iter_arg 编号）
for (unsigned deadArgIdx : deadArg) {
  // 取对应的 BlockArgument 时要 +1 跳过 %iv
  BlockArgument arg = block.getArgument(deadArgIdx + 1);
  // 把 yield operand 替换为 BlockArgument 自身（变成自引用透传）
  yieldOp.setOperand(deadArgIdx, arg);
}
```

当内联后，有些 `iter_args` 可能变得不再被循环体使用，此时就需要消除它们。

##### 算法核心逻辑

这个 pattern 实现了一个 **liveness analysis（活跃性分析）**，判断每个 `iter_args` 对应的 yield operand 是否为死值：

```
开始：标记所有有外部 use 的 yield operand 为 "live"
  │
  ▼
walk 循环体：标记所有有副作用的 op 的 operand 为 "live"
  │
  ▼
固定点传播：从 live 值出发，沿 def-use 链反向传播
  ├─ 若 def 是 scf.for → 标记其 init arg 和对应 yield operand 为 live
  ├─ 若 def 是 scf.if  → 标记 condition 和 then/else 的 yield operand 为 live
  └─ 若 def 是普通 op → 标记其所有 operand 为 live
  │
  ▼
检查每个 yield operand：
  如果 NOT live → 将 yield operand 替换为 block argument 自身（变成自引用）
  │
  ▼
后续 DCE 会清理无用的 block argument 和对应的迭代传递
```

##### `markLive` 的设计选择：为什么不用 `getDefiningOp()`

```cpp
auto markLive = [&](Value val) {
  // 问：为什么不用 val.getDefiningOp()？
  // 答：因为 val 可能是 BlockArgument（如 %iv、%arg），getDefiningOp() 返回 nullptr
  if (!forOp->isAncestor(val.getParentRegion()->getParentOp()))
    return;
  if (aliveValues.insert(val).second)
    queue.push_back(val);
};
```

`getParentRegion()->getParentOp()` 和 `getDefiningOp()` 都是回溯 Value 的"来源"，但覆盖范围不同：

| Value 类型 | `getDefiningOp()` | `getParentRegion()->getParentOp()` |
|-----------|---------------------|--------------------------------------|
| **OpResult**（如 `addi` 的结果） | ✅ 返回产生它的 `Operation*` | ✅ 返回同样的 `Operation*` |
| **BlockArgument**（如 `%iv`、`%arg`） | ❌ **返回 nullptr** | ✅ 返回**拥有这个 Block 的 Op**（就是 `scf.for` 自己） |

这个 pass 中被 `markLive` 检查的 Value **可能来自 yield 的 operand**，而 yield operand 完全可能是 BlockArgument（比如 `scf.yield %arg`）。如果用 `getDefiningOp()`，BlockArgument 返回 `nullptr` 导致 `isAncestor(nullptr)` 要么 crash 要么错误返回 false，跳过本该处理的参数。

**结论**：处理任何可能包含 BlockArgument 的 Value 时，用 `getParentRegion()->getParentOp()` 做容器回溯是统一安全的选择。

##### 代码逐段解释（对应 Utility.cpp:1245-1370）

| 行号 | 代码 | 含义 |
|------|------|------|
| `1248-1249` | `matchAndRewrite(scf::ForOp forOp, ...)` | 匹配一个 `scf.for` op |
| `1251` | `cast<scf::YieldOp>(block.getTerminator())` | 拿到循环体的终结 op `scf.yield` |
| `1254-1262` | `DenseSet<Value> aliveValues` + `markLive` | 活跃值集合 + 标记函数。`insert().second` 首次见到才入队 |
| `1266-1269` | 遍历 `forOp.getResults()`，若有外部 use → 标记对应的 yield operand 为 live | 这是初始种子：循环结果若被外部使用，对应的 yield operand 必须 live |
| `1274-1279` | `block.walk(...)` 遍历所有有副作用的 op → 标记其 operand 为 live | 防止错误删除有 side effect 但无外部 use 的 op 的 operand |
| `1281-1326` | **固定点传播**：从 queue 中取出 live 值，沿 def 链反向传播 | 见下表 |
| `1327-1356` | 收集所有 dead 的 yield operand 索引 | 那些不在 aliveValues 中的 yield operand |
| `1359-1367` | `modifyOpInPlace`：将 dead yield operand 替换为 block argument 自身 | `block.getArgument(deadArgIdx + 1)` 因为 index 0 是 `%iv` |

**固定点传播中的三种 def 类型：**

```
queue 中取出的 Value v
     │
     ├── v 是 scf.for 的 OpResult
     │   ├── 标记对应的 init arg 为 live
     │   └── 标记该 result 对应的 scf.yield operand 为 live
     │
     ├── v 是 scf.if 的 OpResult
     │   ├── 标记 condition 为 live
     │   └── 标记 then/else 的 yield operand 为 live
     │
     ├── v 是普通 op 的 OpResult → 标记其所有 operand 为 live
     │
     └── v 是 BlockArgument（且 owner 是 scf.for）
         ├── 跳过 induction variable
         └── 标记对应的 init arg 和 yield operand 为 live
```

##### 举例说明

假设内联后产生如下 IR：

```
// Before: 循环体内 %sum 从未被使用
%result:2 = scf.for %iv = %c0 to %c100 step %c1
    iter_args(%unused = %init_val, %used = %c0) -> (i32, i32) {
  %1 = arith.addi %used, %c1
  scf.yield %unused, %1      // %unused 从未被使用
}
```

`ForOpDeadArgElimination` 分析后：

```
// After: dead yield operand 被替换为 block argument 自引用
%result:2 = scf.for %iv = %c0 to %c100 step %c1
    iter_args(%unused = %init_val, %used = %c0) -> (i32, i32) {
  %1 = arith.addi %used, %c1
  scf.yield %unused, %1      // → scf.yield %unused, %1  (yield 不变! 因为是自引用)
}

// 但更常见的情况是 yield 的是其他值：
// 假设 yield %computed, %1 且 %computed 无外部 use
// → yield %unused, %1  替换为 yield %unused, %1
// → 后续 DCE 检测到 %unused 自引用 → 消除整个 iter_arg
```

注意这里 pattern 只做了一步：**将死参数的 yield operand 改回 block argument 自身**。真正的参数消除交给后续的 DCE（死代码消除）来做。这是一种模块化设计：每个 pattern 只做最小变换，让后续通用 pass 接力完成清理。

##### 用到的 MLIR 基础概念

| 概念 | 代码示例 | 说明 |
|------|----------|------|
| `Operation` | `forOp`, `yieldOp` | MLIR 的一切都是 op。op 可以有 0 个或多个 **Region** |
| `Region` | `forOp.getBody()` | op 的嵌套子结构，包含一个或多个 **Block** |
| `Block` | `forOp.getBody()` 返回第一个 block | 一组顺序执行的 op 序列，以 terminator op 结尾 |
| `BlockArgument` | `block.getArgument(idx)` | block 的参数。`scf.for` 的 body block 参数 = induction vars + iter_args |
| `OpResult` | `forOp.getResults()` | op 产生的结果值，也是 SSA Value |
| `Value` | `aliveValues` 集合中的元素 | SSA 值的基类，可以是一个 `OpResult` 或 `BlockArgument` |
| `OpOperand` | `yieldOp.getOperand(idx)` | op 的一个操作数（use），指向一个 `Value` |
| `isAncestor` | `forOp->isAncestor(val.getParentRegion()->getParentOp())` | 检查一个 op 是否是另一个 op 的祖先（沿 Region 嵌套上溯） |
| `walk` | `block.walk([&](Operation *op) { ... })` | 遍历 IR 子树中的所有 op |
| `wouldOpBeTriviallyDead` | `wouldOpBeTriviallyDead(op)` | 检查 op 是否有副作用或有用结果，用于活跃性分析 |
| `PatternRewriter` | `rewriter.modifyOpInPlace(forOp, [&](){...})` | 安全的 IR 修改工具，保证 DRR / DialectConversion 的正确性 |

这些概念构成了 MLIR 的 **IR 骨架**：`Operation` 包含 `Region`，`Region` 包含 `Block`，`Block` 包含 `Operation`（递归）。SSA `Value` 作为 `OpResult`（定义）和 `OpOperand`（使用）在这个树状结构上连接 def-use 链。

##### def-use / use-def 遍历

`ForOpDeadArgElimination` 的核心就是靠 def-use 链传播活跃性。MLIR 提供了两种方向的遍历：

| 方向 | 方法 | 含义 | 代码示例 |
|------|------|------|----------|
| **use → def** | `value.getDefiningOp()` | 一个值被谁定义（反向） | `if (Operation *def = value.getDefiningOp())` |
| **def → uses** | `value.getUsers()` | 一个值被哪些 op 使用（正向） | `for (Operation *user : value.getUsers())` |
| **def → uses** | `value.use_empty()` | 是否有 use（无 use 即可 DCE） | `if (result.value().use_empty()) ...` |
| **def → uses** | `value.getUses()` | 返回 `(OpOperand, idx)` 迭代器 | 遍历每个 use 在所属 op 中的位置 |
| **op → operands** | `op->getOperands()` | op 的所有输入值 | `for (Value operand : def->getOperands())` |
| **op → results** | `op->getResults()` | op 的所有输出值 | `for (auto result : forOp.getResults())` |
| **op → specific** | `op->getOperand(idx)` | 第 idx 个 operand | `yieldOp.getOperand(result.index())` |

**在实际代码中这两种方向是配合使用的** —— 在 `ForOpDeadArgElimination` 的固定点传播里：

```
queue 中取出 Value v（某个 use）
    │
    ▼  use → def
v.getDefiningOp()
    │
    ▼  def → uses（反过来遍历自己的 operand）
def->getOperands()
    │
    ▼
对每个 operand 调用 markLive()，加入 queue（继续下一次 use→def）
```

##### IR 树结构遍历

def-use 遍历的是 SSA 图，而 MLIR 还有另一套遍历：沿 **Operation → Region → Block → Operation** 的嵌套树结构走：

| 方法 | 范围 | 说明 | 代码示例 |
|------|------|------|----------|
| `op->walk(callback)` | 递归子树 | 深度优先，遍历 op 及其所有子孙 op | `block.walk([&](Operation *op) { ... })` |
| `block->walk(callback)` | 递归子树 | 从 block 开始深度优先 | `forOp.getBody()->walk(...)` |
| `block->getOps<T>()` | 本级 | 只遍历当前 block 内的 op，不递归进子 region | `for (auto &op : block)` |
| `block->getTerminator()` | 本级 | block 的最后一条 op（终结 op） | `cast<scf::YieldOp>(block.getTerminator())` |
| `op->getRegions()` | 本级 | op 的所有 region | `forOp->getRegions()` |
| `region->getBlocks()` | 本级 | region 内的所有 block | `forOp.getBody()->getBlocks()` |
| `op->isAncestor(other)` | 沿父链 | 检查 op 是否是 other 的祖先 | `forOp->isAncestor(...)` |

**ForOpDeadArgElimination 同时用了两种遍历：**

```
line 1274: block.walk(...)           ← 树遍历（找有副作用的 op）
line 1306: value.getDefiningOp()     ← def-use 遍历（反向传播）
```

树遍历解决"哪些 op 有副作用"的全局扫描问题，def-use 遍历解决"哪些值 live"的精确传播问题。

#### 为什么在这个时间点运行？

内联后，之前跨函数调用的控制流结构（如 `scf.for`、`scf.if`）可能变得冗余：
- 分支条件变为恒真/恒假 → 死分支可消除
- 循环边界变为常数 → 可折叠
- 循环携带的迭代参数内联后不再使用 → 可消除

这些清理必须在 layout 解析 **之前** 完成，但又要小心不能做 Constant CSE（那会干扰 layout 推断）。这就是 `enableConstantCSE(false)` 的原因。

#### 结合 dump.log 验证

对比 inline 前后的 IR 可以看到效果：helper 中原本包含 i64 溢出检查的复杂控制流模式：

```
%offsets_1 = arith.extsi %pid : i32 to i64
%offsets_2 = arith.extsi %offsets_0 : i32 to i64
%offsets_3 = arith.muli %offsets_1, %offsets_2 : i64
%offsets_6 = arith.cmpi sle, %offsets_3, ...   // 溢出检查
...
```

内联后被 `SimplifyControlFlow` 简化为直接的 i32 运算：

```
%offsets_0 = arith.muli %pid, %offsets : i32   // 直接 i32 乘法，无溢出检查
```

---

## 3. 测试脚本结构

`ops/pass/201_inline_pass.py` 定义了两个函数：

### 3.1 Helper (会被内联)

```python
@gluon.jit
def get_offsets_and_mask(pid, numel, BLOCK: gl.constexpr):
    offsets = pid * BLOCK + gl.arange(0, BLOCK, gl.CoalescedLayout())
    mask = offsets < numel
    return offsets, mask
```

- 用 `gl.CoalescedLayout()` 声明 tile layout（inline 后由后续 pass 解析为具体 blocked layout）
- 两入参 + 一个 `constexpr` → 返回 offsets + mask

### 3.2 主 Kernel (调用 helper)

```python
@gluon.jit
def inline_coalesced_kernel(in_ptr, out_ptr, numel, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    offsets, mask = get_offsets_and_mask(pid, numel, BLOCK)
    value = gl.load(in_ptr + offsets, mask=mask)
    gl.store(out_ptr + offsets, value, mask=mask)
```

- `gl.program_id(0)` → `tt.get_program_id x`
- `get_offsets_and_mask(pid, numel, BLOCK)` → `tt.call`（inline 前）
- `gl.load`/`gl.store` → `tt.load`/`tt.store`

---

## 4. IR Dump — gluon-inline Pass 前后对比

### 4.1 Before `gluon-inline`

```
// -----// IR Dump Before GluonInline (gluon-inline) ('builtin.module' operation) //----- //
```

**关键特征：存在两个独立的 `tt.func`，通过 `tt.call` 连接**

```
tt.func public @inline_coalesced_kernel(...) {
  %pid = tt.get_program_id x : i32
  %0:2 = tt.call @"__main__.get_offsets_and_mask__..."(%pid, %numel)
           : (i32, i32) -> (tensor<1024xi32, #gluon.coalesced_encoding>,
                            tensor<1024xi1, #gluon.coalesced_encoding>)
  %value = tt.splat %in_ptr
  %value_0 = tt.addptr %value, %0#0
  %value_1 = tt.load %value_0, %0#1
  ...
}
```

私有的 helper 函数：

```
tt.func private @"__main__.get_offsets_and_mask__..."(%pid: i32, %numel: i32)
    -> (tensor<1024xi32, #gluon.coalesced_encoding>,
        tensor<1024xi1, #gluon.coalesced_encoding>) {
  %offsets_9 = arith.muli %pid, %offsets_0 : i32          // pid * BLOCK
  %offsets_10 = tt.make_range {end = 1024 : i32, start = 0 : i32}  // arange(0, BLOCK)
  %offsets_11 = tt.splat %offsets_9 : i32
  %offsets_22 = arith.addi %offsets_11, %offsets_10       // offsets = pid*BLOCK + arange
  %mask = tt.splat %numel : i32
  %mask_23 = arith.cmpi slt, %offsets_22, %mask           // mask = offsets < numel
  tt.return %offsets_22, %mask_23
}
```

> helper 函数体中还包含冗余的 i64 溢出检查（`arith.extsi` + `arith.muli` + 边界比较），这是 gluon 前端为安全起见生成的保守代码。

### 4.2 After `gluon-inline`

```
// -----// IR Dump Before GluonInferCoalescedEncodingsPass (gluon-infer-coalesced-encodings) //----- //
```

**关键变化：helper 被内联进主函数，所有逻辑现在在单一函数体内**

```
tt.func public @inline_coalesced_kernel(...) {
  %pid = tt.get_program_id x : i32
  %offsets = arith.constant 1024 : i32                    // BLOCK 常数
  %offsets_0 = arith.muli %pid, %offsets : i32            // pid * BLOCK (i32, 简化了!)
  %offsets_1 = tt.make_range {end = 1024 : i32, start = 0 : i32}  // arange(0, BLOCK)
  %offsets_2 = tt.splat %offsets_0 : i32
  %offsets_3 = arith.addi %offsets_2, %offsets_1          // offsets = pid*BLOCK + arange
  %mask = tt.splat %numel : i32
  %mask_4 = arith.cmpi slt, %offsets_3, %mask             // mask = offsets < numel
  %value = tt.splat %in_ptr
  %value_5 = tt.addptr %value, %offsets_3                 // in_ptr + offsets
  %value_6 = tt.load %value_5, %mask_4                    // load with mask
  %0 = tt.splat %out_ptr
  %1 = tt.addptr %0, %offsets_3                           // out_ptr + offsets
  tt.store %1, %value_6, %mask_4                          // store with mask
}
```

**内联带来的具体变化：**

| 变化项 | Before | After |
|--------|--------|-------|
| 函数个数 | 2 个 (`public` + `private`) | 1 个 (`public`) |
| 调用方式 | `tt.call` 跨函数调用 | 被调用者 body 直接嵌入调用处 |
| helper 签名 | `"__main__.get_offsets_and_mask__i32_i32__(2,)cconstexpr_1024_"` | 已消失 |
| i64 溢出检查 | 存在（`extsi` + `muli` + 边界比较） | 被 `GluonSimplifyControlFlow` 消除 |
| constexpr 传播 | 无（`pid * BLOCK` 在 helper 内部） | `%offsets = arith.constant 1024` 直接可见 |

---

## 5. 运行测试

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate triton
export PYTHONPATH=/root/triton_workspace/triton/python

export TRITON_ALWAYS_COMPILE=1
export MLIR_ENABLE_DUMP=1
export MLIR_DUMP_PATH=/root/triton_workspace/ops/pass/dump.log

python ops/pass/201_inline_pass.py
```

预期输出：

```
============================================================
Gluon Inline Pass 测试
============================================================
[OK] inline_coalesced_kernel: numel=4096, BLOCK=1024
测试通过。
```

---

## 6. 关键要点

1. **Inline 是所有后续 layout 分析的前提** — `CoalescedLayout` 无法跨函数边界传递
2. **测试验证了两种正确性**：功能正确（`torch.testing.assert_close`）+ IR 变换正确（dump 文件 header 中 `IR Dump Before GluonInline` 段的存在）
3. **Inline 附带清理** — `GluonSimplifyControlFlow` 子 pipeline 会消除内联后冗余的 i64 溢出检查
4. **在 dump 文件中找到 `Before GluonInline` 和下一个 `Before` 段之间的 IR 差异**，即可独立观察 gluon-inline 的作用
