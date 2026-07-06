# SCCP (Sparse Conditional Constant Propagation) Pass 分析

```python
# triton/third_party/nvidia/backend/compiler.py
passes.common.add_sccp(pm)
```

## 一、概述

**SCCP（Sparse Conditional Constant Propagation，稀疏条件常量传播）** 是 MLIR 标准 pass 之一（`mlir::createSCCPPass`），其核心目标是通过**乐观数据流分析**（optimistic dataflow analysis）识别出 IR 中所有**恒为常数的值**，将这些值替换为实际的常量操作，并消除因此变为死代码的操作。

| 属性 | 值 |
|------|-----|
| **Pass 名称** | `sccp` |
| **实现文件** | `llvm-project/mlir/lib/Transforms/SCCP.cpp` |
| **分析组件** | `SparseConstantPropagation` + `DeadCodeAnalysis` |
| **策略** | 乐观（optimistic）：假设所有值都是常数，直到被证明不是 |

---

## 二、工作原理

### 2.1 两步流程

```
          ModuleOp
             │
             ▼
  ┌─────────────────────────┐
  │  DataFlowSolver         │  ← 第 1 步：数据流分析
  │  ├── DeadCodeAnalysis   │     加载两个分析器
  │  └── SparseConstantPropagation │
  └─────────┬───────────────┘
             │ (求解完成：每个 SSA 值的 lattice 状态已确定)
             ▼
  ┌─────────────────────────┐
  │  rewrite()              │  ← 第 2 步：IR 重写
  │  └─ 用常量替换已知值    │     遍历所有 block 和 op
  │  └─ 消除死代码          │
  └─────────────────────────┘
             │
             ▼
       ModuleOp (已常量折叠)
```

#### `runOnOperation()` 实现

```cpp
void SCCP::runOnOperation() {
  Operation *op = getOperation();

  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();                 // 加载死代码分析
  solver.load<SparseConstantPropagation>();        // 加载稀疏常量传播
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();
  rewrite(solver, op->getContext(), op->getRegions());
}
```

### 2.2 第 1 步：DataFlowSolver — 乐观数据流分析

`DataFlowSolver` 是 MLIR 的数据流分析框架。它加载两个分析器：

| 分析器 | 作用 |
|--------|------|
| `DeadCodeAnalysis` | 标记不可达的代码块和操作，避免对死代码进行无意义的分析 |
| `SparseConstantPropagation` | 在 SSA 图上跟踪每个值的 lattice（格）状态，判断是否为常数 |

**乐观策略**：假设所有值为常数（顶部 lattice 状态），当遇到使值**不可能是常数**的操作时，将其标记为 `UnknownConstant`（底部状态）。这是"乐观"的——它先假定好消息，只在有确凿证据时才放弃。

**Lattice 状态转换**：

```
  ⊤ (Unknown / 尚未确定)         ← 初始状态（乐观假设）
  │
  ├──→ ConstantValue(c)          ← 分析确定该值为常数 c
  │
  └──→ UnknownConstant           ← 确定不是常数（到达不动点）
                                   值保持原样，不做替换
```

### 2.3 第 2 步：`rewrite()` — IR 重写

```cpp
static void rewrite(DataFlowSolver &solver, MLIRContext *context,
                    MutableArrayRef<Region> initialRegions) {
  // worklist 遍历所有 block
  SmallVector<Block *> worklist;
  // ...
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();
    for (Operation &op : llvm::make_early_inc_range(*block)) {
      // 1. 尝试将 op 的每个 result 替换为常量
      // 2. 如果所有 result 都被替换且 op 无副作用 → erase
      // 3. 将 op 的子 region 加入 worklist
    }
    // 4. 尝试将 block argument 替换为常量
  }
}
```

核心替换逻辑 `replaceWithConstant`：

```cpp
static LogicalResult replaceWithConstant(DataFlowSolver &solver,
                                         OpBuilder &builder,
                                         OperationFolder &folder, Value value) {
  auto *lattice = solver.lookupState<Lattice<ConstantValue>>(value);
  if (!lattice || lattice->getValue().isUninitialized())  return failure();
  const ConstantValue &latticeValue = lattice->getValue();
  if (!latticeValue.getConstantValue())                    return failure();

  // 用 OperationFolder 在适当位置创建常量 op
  Dialect *dialect = latticeValue.getConstantDialect();
  Value constant = folder.getOrCreateConstant(
      builder.getInsertionBlock(), dialect,
      latticeValue.getConstantValue(), value.getType());
  if (!constant) return failure();

  value.replaceAllUsesWith(constant);   // 替换所有 use
  return success();
}
```

---

## 三、能处理哪些场景

### 3.1 基本常量折叠

```
// Before                          // After
%c0 = arith.constant 0 : i32
%c1 = arith.constant 1 : i32
%r  = arith.addi %c0, %c1          %c1 = arith.constant 1 : i32
  : i32                                  // %r 被 %c1 替换
```

### 3.2 条件分支传播

```
// Before                          // After
%cond = arith.cmpi eq, %x, %x     %true = arith.constant true
scf.if %cond {                     // if 分支被折叠
  ...
}
```

### 3.3 通过控制流传播常量

```
// Before — x 的来源有两条路径，但都是同一个常量
%c5 = arith.constant 5 : i32
scf.if %cond {
  scf.yield %c5
} else {
  scf.yield %c5
}
%x = scf.if ...                     // %x 被推导为常量 5

// After
%c5 = arith.constant 5 : i32
// scf.if 被消除，%x 的所有 use 替换为 %c5
```

### 3.4 死代码消除（附带效果）

当一个 op 的所有 result 都被常量替换后，如果该 op 无副作用，它会被自动删除：

```
// Before                          // After
%c0 = arith.constant 0 : i32
%t  = arith.muli %a, %c0          %c0 = arith.constant 0 : i32
                                    // %t 被 %c0 替换，muli 消除
```

### 3.5 Block Argument 常量传播

```
// Before — loop 的 iter_arg 是常量
%c0 = arith.constant 0 : i32
%r  = scf.for %iv = %lb to %ub step %st
        iter_args(%arg = %c0) -> i32 {
  %1 = arith.addi %arg, %c1
  scf.yield %1
}
// 如果 %arg 在循环内可被推导为常量... → %arg 被 %c0 替换
```

---

## 四、与 canonicalize 和 CSE 的配合

在 gluon pipeline 中，SCCP 被放在两轮 gluon-canonicalize 之间：

```
┌─ gluon-canonicalize (第 1 次) ──┐   基础化简（消除冗余 mask、合并 broadcast 等）
├─ common-sccp                  ──┤   ← 本 pass：常量传播+折叠
├─ ttir-loop-aware-cse          ──┤   循环感知 CSE（消除重复计算）
├─ gluon-canonicalize (第 2 次) ──┤   二次化简（常量替换后暴露的新化简机会）
```

这种安排遵循经典的 **"化简→传播→消除→化简"** 模式：

| 步骤 | Pass | 效果 |
|------|------|------|
| ① | `gluon-canonicalize` | 将 IR 化简为标准形式（露出更多常量折叠机会） |
| ② | **`common-sccp`** | **常量传播+折叠（发现并替换常量值）** |
| ③ | `loop-aware-cse` | 消除 SCCP 后重复的公共子表达式 |
| ④ | `gluon-canonicalize` | 对替换后的 IR 做二次化简 |

---

## 五、与 `arith` 常量折叠（constant fold）的区别

MLIR 中有两种常量优化机制：

| 特性 | SCCP | OpFoldResult / constant fold |
|------|------|------------------------------|
| **分析范围** | 跨基本块、跨控制流 | 仅局部（一个 op 的 operand 都是已知常量） |
| **控制流感知** | ✅ 是（通过 `DeadCodeAnalysis`） | ❌ 否 |
| **乐观传播** | ✅ 是（假设所有值都是常量） | ❌ 否（需要所有 operand 都是已知常量） |
| **复杂度** | 高（完整数据流分析） | 低（即时折叠） |
| **典型应用** | 传播循环不变常量、跨分支常量 | 2+3→5 这类简单常量折叠 |

简单来说：**constant fold** 解决的是 `addi(2, 3) → 5` 这样的简单问题，而 **SCCP** 能回答"这个循环 `iter_arg` 在运行时一定是常数 42 吗？"

---

## 六、在 Triton pipeline 中的用途

在 Triton 编译器中，SCCP 主要在以下位置发挥作用：

1. **gluon pipeline**（`gluon_to_ttgir`）：
   - 在 layout 解析后传播常量化的 program_id、步长、边界值等
   - 折叠掉由常量 mask 产生的死分支
   - 为后续的 CSE 和二次 canonicalize 创造更好的化简条件

2. **make_ttgir pipeline**：
   - 同样在 layout 相关 pass 之后做常量传播

---

*本文档对应源代码：[SCCP.cpp](../../llvm-project/mlir/lib/Transforms/SCCP.cpp)*
