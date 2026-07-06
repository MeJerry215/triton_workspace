# `convert-scf-to-cf` Pass 分析

```python
# triton/third_party/nvidia/backend/compiler.py
passes.convert.add_scf_to_cf(pm)
```

## 一、概述

| 属性 | 值 |
|------|-----|
| **Pass 名称（CLI）** | `convert-scf-to-cf` |
| **TableGen 定义** | `def SCFToControlFlowPass : Pass<"convert-scf-to-cf">` |
| **C++ 实现** | `mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp` |
| **Python 绑定** | `ADD_PASS_WRAPPER_0("add_scf_to_cf", createSCFToControlFlowPass)` |
| **操作粒度** | 任意 operation（通常为 `ModuleOp`） |
| **依赖 Dialect** | `scf`（输入）、`cf`（输出）、`arith`（生成比较指令） |

**摘要（来自 TableGen）**：

> 将 SCF dialect 转换为 ControlFlow dialect，用 CFG (Control Flow Graph) 替换结构化控制流。

### 什么是结构化控制流 (SCF) vs CFG？

这是 MLIR 编译流程中一个关键的**结构化→非结构化**转换步骤。

| 特性 | SCF Dialect（结构化） | CF Dialect（CFG） |
|------|---------------------|-------------------|
| **循环** | `scf.for`、`scf.while`、`scf.parallel` | `cf.br` 回边（back-edge） |
| **条件** | `scf.if` | `cf.cond_br` 条件分支 |
| **多路分支** | `scf.index_switch` | `cf.switch` |
| **区域执行** | `scf.execute_region` | 直接内联 + `cf.br` |
| **并行** | `scf.forall` | 先降为 `scf.parallel` → 再展开为 `scf.for` 嵌套 |
| **语法** | 嵌套 region、结构化终止指令 | 基本块、`cf.br`/`cf.cond_br` |
| **适合后续** | 分析、变换 | 代码生成（LLVM IR） |

### 为什么在 Triton 流程中此时执行？

```
make_llir pipeline:
  │
  ├─ ① tritongpu-combine-tensor-select-and-if  (精简 IR)
  ├─ ② tritongpu-allocate-warp-groups           (warp group 分配)
  ├─ ③ convert-scf-to-cf                        ← 此处：结构化 → 非结构化
  ├─ ④ gluon-inliner                            (内联)
  ├─ ⑤ allocate-shared-memory / allocate-tensor-memory
  ├─ ⑥ tritongpu-to-llvmir                      (TTGPUIR → LLVM IR)
  ├─ ⑦ nvgpu-to-llvm / warp-specialize-to-llvm
  └─ ⑧ canonicalizer + cse + symbol-dce
```

关键原因：**SCF 的结构化控制流（带有 region、yield 的结构）在 LLVM IR 中不存在**。在将 TTGPUIR 转换为 LLVM IR 之前，必须将所有 `scf.*` 操作转换为 `cf.*` 基本块操作。这是 MLIR 跨 dialect 转换的标准模式。

---

## 二、转换总览：7 种 SCF 操作 → CFG

本 pass 对以下 7 种 SCF 操作进行转换：

| SCF 操作 | 转换目标 | 负责 Pattern | 复杂度 |
|----------|---------|-------------|--------|
| `scf.for` | 条件块 + 循环体 + 回边 | `ForLowering` | ⭐⭐⭐ |
| `scf.if` | `cf.cond_br` + then/else 块 | `IfLowering` | ⭐⭐ |
| `scf.while` | before/after 双区域内联 | `WhileLowering` | ⭐⭐⭐ |
| `scf.while`（do-while 优化） | 简化为单区域循环 | `DoWhileLowering` | ⭐⭐ |
| `scf.parallel` | 展开为 `scf.for` 嵌套（间接转换） | `ParallelLowering` | ⭐⭐⭐ |
| `scf.execute_region` | 直接内联 region 为基本块 | `ExecuteRegionLowering` | ⭐ |
| `scf.index_switch` | `cf.switch` | `IndexSwitchLowering` | ⭐⭐ |
| `scf.forall` | 先降为 `scf.parallel` | `ForallLowering` | ⭐ |

转换整体通过 `DialectConversion` 框架执行：

```cpp
void SCFToControlFlowPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateSCFToControlFlowConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addIllegalOp<scf::ForallOp, scf::ForOp, scf::IfOp,
                      scf::IndexSwitchOp, scf::ParallelOp,
                      scf::WhileOp, scf::ExecuteRegionOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
```

**核心策略**：将这 7 种操作设为非法（`addIllegalOp`），未知操作动态合法。`applyPartialConversion` 会将所有非法操作逐一用 pattern 替换。

---

## 三、`scf.for` → CFG（`ForLowering`）

### 3.1 变换效果

```
┌─ 变换前：scf.for（结构化）─────────────────────────────────────────┐
│                                                                     │
│  %r:2 = scf.for %iv = %lb to %ub step %st                          │
│      iter_args(%arg0 = %init0, %arg1 = %init1) -> (i32, f32) {     │
│    %body = arith.addi %arg0, %c1 : i32                              │
│    scf.yield %body, %cst : i32, f32                                 │
│  }                                                                  │
│  // 使用 %r#0, %r#1                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 变换后：CFG 基本块 ────────────────────────────────────────────────┐
│                                                                     │
│  // init block: 初始化 iv, iter_args, 跳转到 cond                   │
│  %iv_init = arith.addi %lb, %c0              ← iv = lowerBound     │
│  cf.br ^cond(%iv_init, %init0, %init1)                             │
│                                                                     │
│  ^cond(%iv: i32, %arg0: i32, %arg1: f32):    ← 条件块              │
│    %cmp = arith.cmpi slt, %iv, %ub           ← iv < upperBound?    │
│    cf.cond_br %cmp, ^body_first, ^end                              │
│                                                                     │
│  ^body_first:                                  ← 循环体第一个块     │
│    // %arg0, %arg1 通过支配关系可见                                 │
│    %body = arith.addi %arg0, %c1 : i32                              │
│    // ... 更多 body 块 ...                                          │
│                                                                     │
│  ^body_last:                                   ← 循环体最后一个块   │
│    %new_iv = arith.addi %iv, %st              ← iv += step          │
│    cf.br ^cond(%new_iv, %body, %cst)          ← 回边到 cond        │
│                                                                     │
│  ^end:                                         ← 延续点             │
│    // %arg0, %arg1 可见（条件块参数）                               │
│    // ... 循环之后的代码 ...                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 CFG 结构图

```text
      +---------------------------------+
      |   <code before the ForOp>       |
      |   <definitions of %init...>     |
      |   <compute initial %iv value>   |
      |   cf.br cond(%iv, %init...)        |
      +---------------------------------+
             |
  -------|   |
  |      v   v
  |   +--------------------------------+
  |   | cond(%iv, %init...):           |
  |   |   <compare %iv to upper bound> |
  |   |   cf.cond_br %r, body, end        |
  |   +--------------------------------+
  |          |               |
  |          |               -------------|
  |          v                            |
  |   +--------------------------------+  |
  |   | body-first:                    |  |
  |   |   <%init visible by dominance> |  |
  |   |   <body contents>              |  |
  |   +--------------------------------+  |
  |                   |                   |
  |                  ...                  |
  |                   |                   |
  |   +--------------------------------+  |
  |   | body-last:                     |  |
  |   |   <body contents>              |  |
  |   |   <operands of yield = %yields>|  |
  |   |   %new_iv = <add step to %iv>  |  |
  |   |   cf.br cond(%new_iv, %yields)    |  |
  |   +--------------------------------+  |
  |          |                            |
  |-----------        |--------------------
                      v
      +--------------------------------+
      | end:                           |
      |   <code after the ForOp>       |
      |   <%init visible by dominance> |
      +--------------------------------+
```

### 3.3 关键维护的不变量

- **单入口单出口**（SESE）：每个循环子图只有一个 entry 和一个 exit
- entry 是 region 的第一个块，exit 是 region 的最后一个块
- 循环携带的值通过**条件块的 block argument** 传递

### 3.4 LLVM 循环注解传播

```cpp
static void propagateLoopAttrs(Operation *scfOp, Operation *brOp) {
  // 将 scf.for 上的 LLVM dialect 属性（如 llvm.loop_annotation）
  // 传播到回边（cf.br）上，供后续 LLVM 使用
  SmallVector<NamedAttribute> llvmAttrs;
  llvm::copy_if(scfOp->getAttrs(), std::back_inserter(llvmAttrs),
                [](auto attr) {
                  return isa<LLVM::LLVMDialect>(attr.getValue().getDialect());
                });
  brOp->setDiscardableAttrs(llvmAttrs);
}
```

---

## 四、`scf.if` → CFG（`IfLowering`）

### 4.1 无返回值的情况

```text
// 变换前                          // 变换后
scf.if %cond {                     cf.cond_br %cond, ^then, ^else
  // then 代码                     ^then:
} else {                             // then 代码
  // else 代码                       cf.br ^continue
}                                  ^else:
                                     // else 代码
                                     cf.br ^continue
                                   ^continue:
```

### 4.2 有返回值的情况

当 `scf.if` 产生结果值时，需要一个**额外的支配块（dom block）**来汇聚两个分支的值：

```text
      +--------------------------------+
      | <code before the IfOp>         |
      | cf.cond_br %cond, ^then, ^else    |
      +--------------------------------+
             |              |
             |              --------------|
             v                            |
      +--------------------------------+  |
      | then:                          |  |
      |   <then contents>              |  |
      |   cf.br ^dom(%args...)            |  |
      +--------------------------------+  |
             |                            |
   |----------               |-------------
   |                         V
   |  +--------------------------------+
   |  | else:                          |
   |  |   <else contents>              |
   |  |   cf.br ^dom(%args...)            |
   |  +--------------------------------+
   |         |
   ------|   |
         v   v
      +--------------------------------+
      | ^dom(%args...):                |
      |   cf.br ^continue                 |
      +--------------------------------+
             |
             v
      +--------------------------------+
      | continue:                      |
      | <code after the IfOp>          |
      +--------------------------------+
```

**为什么需要 `^dom` 块？** 因为 MLIR 的 block argument 只能在创建块时确定，不能在现有块上追加。通过插入一个只有 `cf.br continue` 的中间块，可以把两个分支的 block argument 统一起来，再让 `continue` 块通过参数接收它们。

### 4.3 代码对应

```cpp
// 如果 if 有返回值，在 continue 之前插入一个 dom 块
if (ifOp.getNumResults() == 0) {
  continueBlock = remainingOpsBlock;  // 直接接续
} else {
  continueBlock = rewriter.createBlock(remainingOpsBlock,
      ifOp.getResultTypes(),
      SmallVector<Location>(ifOp.getNumResults(), loc));
  cf::BranchOp::create(rewriter, loc, remainingOpsBlock);
}
// then: 转换为 cf.br ^dom(then_values)
// else: 转换为 cf.br ^dom(else_values)
// 原 if 结果 → 替换为 continueBlock 的 block argument
rewriter.replaceOp(ifOp, continueBlock->getArguments());
```

---

## 五、`scf.while` → CFG（`WhileLowering` + `DoWhileLowering`）

### 5.1 标准 `scf.while` 转换

`scf.while` 有两个 region：`before`（条件判断）和 `after`（循环体）。

```text
// 变换前                           // 变换后
%r = scf.while (%arg = %init)       cf.br ^before(%init)
    : (i32) -> (i32) {
  ^bb0(%barg: i32):                ^before(%barg: i32):
    %cond = ...                    %cond = ...
    scf.condition(%cond)           cf.cond_br %cond, ^after(%vals...), ^cont
        %vals... : i32
  } after {                        ^after(%aargs: i32):
  ^bb1(%aarg: i32):                  // body
    %res = ...                       %res = ...
    scf.yield %res : i32             cf.br ^before(%res)
  }
                                    ^cont:
                                      // 循环后的代码
```

### 5.2 do-while 优化（`DoWhileLowering`）

当 `after` region 仅仅是**将参数原样转发回 before** 时（即 `after` 没有实际 payload），可以简化：

```
scf.while (%arg = %init) : (i32) -> (i32) {
  ^bb0(%barg: i32):
    %cond = ...
    scf.condition(%cond) %barg : i32
  } after {
  ^bb1(%aarg: i32):
    scf.yield %aarg : i32    ← 直接转发，没有修改
  }
}
```

这实际上就是一个 `do-while` 循环。DoWhileLowering 将其简化为单 region：

```text
cf.br ^before(%init)

^before(%barg: i32):
  <body>
  %cond = ...
  cf.cond_br %cond, ^before(%res), ^cont

^cont:
```

- 省去了 `after` region 的内联
- 回边直接从 before 的末尾跳回 before 开头

---

## 六、其他 SCF 操作的转换

### 6.1 `scf.parallel` → `scf.for` 嵌套（`ParallelLowering`）

`scf.parallel` 是一个多维并行循环，支持归约操作。`ParallelLowering` 将其展开为嵌套的 `scf.for` 循环：

```text
// 变换前
scf.parallel (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1)
    step (%st0, %st1) {
  // 并行体
} reductions {
  // 归约操作
}

// 变换后（外循环）
scf.for %i = %lb0 to %ub0 step %st0 {
  // 内循环
  scf.for %j = %lb1 to %ub1 step %st1 {
    // 原并行体（归约已合并）
  }
}
// 后续再由 ForLowering 进一步降低为 CFG
```

关键步骤：
1. 创建 `n` 层嵌套的 `scf.for`（n 为并行维度数）
2. 将归约块内联到最内层循环体中
3. 替换 `scf.yield` 为转发结果的终止指令

### 6.2 `scf.execute_region` → 内联（`ExecuteRegionLowering`）

```text
// 变换前                           // 变换后
%r = scf.execute_region -> (i32) { ^before:
  %v = arith.constant 42 : i32      %v = arith.constant 42 : i32
  scf.yield %v : i32                 cf.br ^cont(%v : i32)
}
                                  ^cont(%r: i32):
```

- 将 region 的块内联到父区域
- `scf.yield` → `cf.br ^cont(yield_vals)`
- `continue` 块获得 block argument，替换原 `execute_region` 的结果

### 6.3 `scf.index_switch` → `cf.switch`（`IndexSwitchLowering`）

```text
// 变换前                           // 变换后
%r = scf.index_switch %arg : i32    %case_val = arith.index_cast %arg : i32
case 0: {                            cf.switch %case_val : i32 [
  %v0 = ...                            ^case0: ^case0,
  scf.yield %v0 : i32                  ^case1: ^case1,
} case 1: {                          ]
  %v1 = ...                         ^case0:
  scf.yield %v1 : i32                  cf.br ^cont(%v0 : i32)
} default: {                         ^case1:
  %vd = ...                            cf.br ^cont(%v1 : i32)
  scf.yield %vd : i32               } ^default:
                                   cf.br ^cont(%vd : i32)
                                   ^cont(%r: i32):
```

- `arith.index_cast` 将 `index` 转为 `i32`（PTX `cfs.switch` 的要求）
- 每个 case region 的 `scf.yield` → `cf.br ^cont(case_val)`

### 6.4 `scf.forall` → `scf.parallel`（`ForallLowering`）

```cpp
LogicalResult ForallLowering::matchAndRewrite(...) {
  return scf::forallToParallelLoop(rewriter, forallOp);
}
```

`scf.forall` 是较新的并行循环操作，先降为 `scf.parallel`，然后由 `ParallelLowering` 进一步处理。注意：**有 shared outputs 的 forall 必须先 bufferization**。

---

## 七、总结：SCF → CF 转换的意义

| 维度 | SCF（结构化） | CF（CFG） |
|------|-------------|----------|
| **表示能力** | 有 region 边界，结构化嵌套 | 平面基本块，任意跳转 |
| **分析友好** | ✅ 结构清晰、支配关系明确 | ❌ 需要额外分析（DomTree, LoopInfo） |
| **代码生成** | ❌ LLVM IR 不支持 region | ✅ 直接对应 LLVM 的基本块 |
| **pass 数量** | 1 个 `convert-scf-to-cf` 覆盖全部 | - |

**关键设计权衡**：

- **SCF 适合变换阶段**：结构化的 region 使得 pass 可以轻松识别循环边界、if-then-else 结构，便于做循环变换、内存提升等优化。
- **CF 适合代码生成阶段**：在进入 LLVM IR 之前，将所有结构化控制流展开为基本块，是标准 MLIR lowering 流程的最后一步。

这也解释了为什么 `convert-scf-to-cf` 出现在 Triton 的 `make_llir` pipeline 中相对靠前的位置——**在它之后的 pass（内联、内存分配、dialect 转换）都是基于基本块（CFG）工作，不再需要理解 SCF 的结构化语义**。

---

*本文档对应源代码：[SCFToControlFlow.cpp](../../llvm-project/mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp)*
