# `triton-loop-aware-cse` Pass 分析

```python
# triton/third_party/nvidia/backend/compiler.py
passes.ttir.add_loop_aware_cse(pm)
```

## 一、概述

| 属性 | 值 |
|------|-----|
| **Pass 名称（CLI）** | `triton-loop-aware-cse` |
| **TableGen 定义** | `def TritonLoopAwareCSE : Pass<"triton-loop-aware-cse", "mlir::ModuleOp">` |
| **C++ 实现** | `triton/lib/Dialect/Triton/Transforms/LoopAwareCSE.cpp` |
| **Python 绑定** | `ADD_PASS_WRAPPER_0("add_loop_aware_cse", createTritonLoopAwareCSE)` |
| **操作粒度** | `mlir::ModuleOp` |

**摘要（来自 TableGen）**：

> 在循环体内做 CSE。与普通 CSE（单遍贪心算法）不同，本 pass 能够**递归消除**循环迭代参数和子计算中**恒为相同值**的那些。

**与普通 CSE 的核心区别**：

| | 标准 CSE | LoopAwareCSE |
|--|----------|--------------|
| 范围 | 单次遍历，同一基本块内 | 跨 `scf.for` 迭代理解等价性 |
| 循环参数 | 无法判断 `iter_args[0]` 与 `iter_args[1]` 是否等价 | ✅ 通过递归分析 yield 值判断 |
| 实现 | 基于 DominanceInfo 的贪心算法 | 标准 CSE + LoopCSEDriver 递归分析 |

---

## 二、`runOnOperation()` 四步流程

```cpp
void runOnOperation() override {
    // Step 1: 标准 CSE（预处理）
    // 先把循环外所有等价的 SSA 值变成指针相等
    eliminateCommonSubExpressions(rewriter, domInfo, getOperation());

    // Step 2: 循环感知的 iter_arg 消除（核心）
    // 找出 scf.for 中等价的迭代参数并合并
    getOperation().walk(loopCSE);

    // Step 3: 标准 CSE（后处理）
    // 消除 Step 2 暴露出的新冗余计算
    eliminateCommonSubExpressions(rewriter, domInfo, getOperation());

    // Step 4: scf.for 规范化清理
    // 消除 short-circuit 后的死参数和未用结果
    RewritePatternSet patterns(&getContext());
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
```

```
                    ┌──────────────────────────────────────┐
                    │          ModuleOp                     │
                    └──────────────────────────────────────┘
                                    │
                 ┌──────────────────┼──────────────────────┐
                 ▼                  ▼                       ▼
      ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
      │    Step 1        │  │    Step 2        │  │    Step 3+4      │
      │  标准 CSE 预处理  │→ │  LoopCSEDriver   │→ │  再次 CSE +      │
      │  (外部值指针化)   │  │  (合并 iter_arg) │  │  规范化清理       │
      └──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 三、核心算法：`LoopCSEDriver` — 递归循环感知等价性检测

### 3.1 `areEqualInLoop(a, b)`

这是本 pass 的核心：判断循环体内部两个 SSA 值是否**在循环的每一轮迭代中都相等**。

```cpp
bool LoopCSEDriver::areEqualInLoop(Value a, Value b) {
  // 1. 指针相等 → 肯定相等
  if (a == b) return true;
  // 2. 类型不同 → 肯定不等
  if (a.getType() != b.getType()) return false;
  // 3. 两个值必须都在循环体内部（都不是从外部传入的）
  if (aBlock != loop.getBody() || bBlock != loop.getBody()) return false;
  // 4. 不能是 induction variable（每轮不同）
  if (a == loop.getInductionVar() || b == loop.getInductionVar()) return false;

  if (auto aArg = dyn_cast<BlockArgument>(a)) {
    // 5. 如果是 block argument → 递归检查对应的 iter_arg
    return areIterArgsEqual(aArg.getArgNumber() - 1,
                            bArg.getArgNumber() - 1);
  }

  // 6. 如果是 op result → 递归检查两个 ops 的 operand 是否两两等价
  //    要求：op 无副作用、无子 region、result number 相同
  bool result = OperationEquivalence::isEquivalentTo(
      aDef, bDef,
      [&](Value a, Value b) { return success(areEqualInLoop(a, b)); },
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
  return result;
}
```

### 3.2 `areIterArgsEqual(i, j)` — 递归判断两个 `iter_arg` 是否等价

关键点：**用 worklist 打破递归环**。

```cpp
bool LoopCSEDriver::areIterArgsEqual(int i, int j) {
  // 1. 同一个 arg → 相等
  if (i == j) return true;
  // 2. 初始值不同 → 不可能相等
  if (loop.getInitArgs()[i] != loop.getInitArgs()[j]) return false;
  // 3. 如果已经在等价栈中 → 假设相等（打破递归）
  if (llvm::is_contained(argStack, std::make_pair(i, j))) return true;

  // 4. 递归检查对应的 yield 值是否等价
  argStack.push_back({i, j});
  bool result = areEqualInLoop(
      loop.getYieldedValues()[i], loop.getYieldedValues()[j]);
  argStack.pop_back();
  return result;
}
```

### 3.3 递归等价性检测的直观理解

以一个常见模式为例：

```mlir
// 两个 iter_arg 的初始值相同，yield 的值也相同
%init = arith.constant 0 : i32
%r:2 = scf.for %iv = %lb to %ub step %st
    iter_args(%a = %init, %b = %init) -> (i32, i32) {
  %plus_a = arith.addi %a, %c1     // a++
  %plus_b = arith.addi %b, %c1     // b++
  scf.yield %plus_a, %plus_b       // yield 值结构相同
}
```

`areIterArgsEqual(0, 1)` 的递归过程：

```
areIterArgsEqual(0, 1)
  ├─ init args: %init == %init                   ✅
  └─ areEqualInLoop(yield[0]=%plus_a, yield[1]=%plus_b)
       ├─ 都是 OpResult，defining op 都是 arith.addi
       ├─ result number 都是 0
       ├─ 无副作用、无 region                    ✅
       └─ isEquivalentTo(addi %a %c1, addi %b %c1)
            ├─ areEqualInLoop(%a, %b)
            │    └─ 都是 BlockArgument(iter_arg)
            │         └─ areIterArgsEqual(0, 1)   ↺ 已在 argStack → true
            └─ areEqualInLoop(%c1, %c1)
                 └─ 指针相等 → true               ✅
       → true                                     ✅
  → true                                          ✅
```

→ 结论：`%a` 和 `%b` 等价 → 合并。

### 3.4 合并等价 `iter_arg`

`loopCSE()` 使用 `EquivalenceClasses` 分组，对每组执行：

```cpp
// 选择 leader 作为唯一保留的 arg
BlockArgument unique = loop.getRegionIterArg(eqArgs.front());
// 用 unique 替换所有等价的 other arg
for (int j : drop_begin(eqArgs)) {
  BlockArgument other = loop.getRegionIterArg(j);
  other.replaceAllUsesWith(unique);
  // Short-circuit: yield 值改为 other 自身
  // 后续 canonicalizer 会消除 dead arg
  (*loop.getYieldedValuesMutable())[j].set(other);
  loop.getResult(j).replaceAllUsesWith(uniqueResult);
}
```

**Short-circuit 技巧**：被合并的 yield 值被替换为 block argument 自身（自引用），变成死代码让后续 canonicalizer 消除。

---

## 四、与其他 Pass 的配合

在 `gluon_to_ttgir` pipeline 中的位置：

```
gluon-canonicalize (第 1 次)   化简 IR
common-sccp                    常量传播（可能制造值等价）
▸ triton-loop-aware-cse         ← 本 pass：消除冗余循环参数
gluon-canonicalize (第 2 次)   清理 short-circuit 的死参数
```

在 `make_ttgir` pipeline 中的位置（多次出现）：

```
... (其他 pass)
▸ triton-loop-aware-cse        消除循环中冗余计算
common-canonicalizer
common-cse
...
```

---

## 五、典型优化场景

### 场景 1：冗余的循环迭代参数

```
// Before
%r:2 = scf.for %iv = %c0 to %c100 step %c1
    iter_args(%a = %c0, %b = %c0) -> (i32, i32) {
  %1 = arith.addi %a, %c1
  %2 = arith.addi %b, %c1      // %b 完全等价于 %a
  scf.yield %1, %2
}
// After: %b 被 %a 替换
%r = scf.for %iv = %c0 to %c100 step %c1
    iter_args(%a = %c0) -> i32 {
  %1 = arith.addi %a, %c1
  scf.yield %1
}
```

### 场景 2：循环展开后的子计算等价

TMA 操作或 warp specialization 展开后，循环体中可能出现结构完全相同的子计算，LoopAwareCSE 可以消除它们。

---

## 六、关键设计要点

1. **递归打破**：通过 `argStack` 跟踪正在分析的 `iter_arg` 对，遇到递归时直接假设相等。这等价于求解一个最小固定点（least fixed point），保证算法终止。

2. **两步 CSE 夹心**：Step 1 标准 CSE 确保外部值指针相等（使 `%c1 == %c1` 为 true），Step 2 做循环感知消除，Step 3 再次 CSE 清理。这种"夹心"结构确保最大限度消除冗余。

3. **只处理 `scf.for`**：本 pass 目前只识别 `scf::ForOp`，不处理 `scf.while` 或其他循环结构。

4. **无副作用限制**：只等价性检查无副作用的 op，有副作用的 op（如 store）不会参与合并。

---

*本文档对应源代码：[LoopAwareCSE.cpp](../../triton/lib/Dialect/Triton/Transforms/LoopAwareCSE.cpp)*
