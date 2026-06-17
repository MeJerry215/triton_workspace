# MLIR 是什么、怎么用、如何学习与开发

<!-- https://mlir.llvm.org/?spm=5176.28103460.0.0.a7e32988aTPe6Q -->

本文集中讲 **MLIR**（Multi-Level IR）：定位、核心概念、日常工具、开发与学习路径。与 **LLVM 工程通用习惯**（ADT、`cast`、CMake 等）重叠处见 [100_llvm_basic.md](100_llvm_basic.md)；**TableGen / ODS 细节**可另配合 [140_mlir_tablegen.md](140_mlir_tablegen.md)（若你已拆分笔记）。

---

## 1. MLIR 是什么

**MLIR** 是 LLVM 子项目里的 **编译器中间表示基础设施**：提供 **可扩展的 IR 模型**、**方言（Dialect）机制**、**Pass 与 PatternRewrite 框架**，以及 **逐步降级（progressive lowering）** 到 **LLVM IR** 或自定义后端的路径。

它**不是**「又一种和 LLVM IR 平级的单一指令集」，而是：

- **同一套 IR 骨架**（Operation / Region / Block / SSA Value）上，挂 **多种方言**；
- 每种方言表达 **某一抽象层次**（算术、张量、仿射、SCF 结构化控制流、LLVM 指令等）；
- 编译流程 = **在方言之间变换、降级**，直到满足你的 **CodeGen** 或 **导出** 需求。

**典型动机**：深度学习编译器、异构硬件、领域专用语言需要 **多层抽象**；若全部挤在 LLVM IR 一层，既难表达高层语义，也难做专用优化。MLIR 把 **「分层 + 可插拔方言 + 统一 Pass 基础设施」** 做成库。

---

## 2. 和 LLVM IR 的关系（必读心智模型）

| | MLIR | LLVM IR |
|---|------|---------|
| 粒度 | **Operation** 可带 **Region**，嵌套结构化 IR | **Function + BasicBlock + Instruction**，扁平 CFG 为主 |
| 扩展 | **Dialect** 自定义 op / 类型 / 属性 | 固定指令集 + intrinsic |
| 终点 | 常经 **LLVM Dialect** → **Translation** → LLVM IR → 现有后端 | 机器码 pipeline 的「中端」 |

很多项目（含部分 **GPU / Triton** 路径）在 **MLIR 完成大部分优化**，再降到 LLVM；**不等于**「可以不懂 LLVM」，但 **日常改 MLIR 方言与 Pass** 是另一套工作流。

---

## 3. 核心概念速览

### 3.1 IR 层次（嵌套）

MLIR 的核心包含关系是一个递归树：

```
MLIR 模块
 └── Operation (如 func.func)
      └── Region (有且仅有一个"父 Op"；可含多个 Region)
           └── Block (Region 中按顺序排列多个 Block)
                ├── Operation (普通 op, 如 arith.addi)
                ├── Operation (普通 op, 如 scf.for —— 它自己又含 Region!)
                │    └── Region
                │         └── Block
                │              ├── Operation (...)
                │              └── Operation (Terminator, 如 scf.yield)
                └── Operation (Terminator, 终结当前 Block)
```

每一层的含义：

| 层级 | C++ 类型 | 说明 |
|------|----------|------|
| **Operation** | `mlir::Operation` | 最基本的 IR 单元，类似"指令"。每个 Op 可携带 0~N 个 **Region** |
| **Region** | `mlir::Region` | Op **内部** 的"包体容器"，一个 Op 可拥有多个 Region（如 `scf.if` 有 then/else 两个 Region）。Region 持有 **零个或多个 Block** |
| **Block** | `mlir::Block` | Region 中的 **基本块**（线性指令序列）。结构化控制流（`scf.for`、`scf.while`）往往只有一个 Block；CFG 风格（`cf.br`）可有多个 Block。Block 的**最后一条 Op 必须是 Terminator** |
| **Terminator** | `mlir::Operation` (有 `Terminator` trait) | Block 的**末尾 Op**，标识控制流去向（`scf.yield` 返回数据给父 Op；`cf.br` 跳转到其他 Block） |

> 注：`func.func` 本身也是 `Operation`，它的 body 是一个 `Region`（内含一个 entry `Block`）。

#### 代码中的遍历模式

```cpp
// 取 scf.for 的 body → 它是个 Region → 解引用取第一个 Block
Block &block = *forOp.getBody();

// 从 Block 取终结者 → 得到 Operation* → cast 到具体 Op 类型
auto yieldOp = cast<scf::YieldOp>(block.getTerminator());
```

分解调用链：

1. **`forOp.getBody()`** — 返回 `Region *`，即 `scf.for` 所拥有的那个 Region（`scf.for` 只有一个 Region）。
2. **`*forOp.getBody()`** — 解引用得 `Region &`。对一个单 Block 的 Region，"取 Region 的内容"就是取它里面的第一个 Block。
3. **`Block &block = ...`** — Region 隐式转换为它内部的第一个 Block（`Region` 提供了 `operator[](unsigned)` 和隐式首 Block 转换）。赋值给 `Block &`。
4. **`block.getTerminator()`** — 遍历 Block 内的操作链表，返回**最后一条 Op**（要求该 Op 有 `Terminator` trait）。
5. **`cast<scf::YieldOp>(...)`** — 把 `Operation*` 向下转型为 `scf::YieldOp`（这是一种 Op 的 C++ 封装类）。

#### MLIR 文本对应

```mlir
%result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %init) -> (i32) {
  // ── 以上 scf.for 本身是一个 Operation ──
  //     ↑         它含一个 Region（花括号内）
  //                Region 内含一个 Block（花括号内的指令序列）

  %sum = arith.addi %arg, %cst : i32     // Block 中的普通 Operation
  scf.yield %sum : i32                    // Block 的 Terminator

  // ── Block 结束，Region 结束，scf.for Op 结束 ──
}
```

#### 要点总结

- **Region** 一定是 **Op 的孩子**，不存在孤立的 Region。
- **Block** 一定是 **Region 的孩子**，不存在孤立的 Block。
- **Terminator** 是 Block 的**最后一个孩子**，类型上也是 `Operation`，但携带 `Terminator` trait。
- 因此 `scf.for`(Op) → Region → Block → `scf.yield`(Terminator Op) 这条链是 MLIR 嵌套模型的**标准模式**，几乎每个结构化 Op 都一样。

#### Region 与 Block 的包含关系详解

**一个 Region 可以有几个 Block？**

| 数量 | 含义 | 常见场景 |
|------|------|----------|
| **0 个 Block** | Region 为空，即 Op 的"体"不存在 | `func.func` 仅声明外部函数（`func.func @foo() -> i32 attributes { sym_visibility = "private" }`）时 body 可以缺失；某些 pass 处理过程中临时掏空 Region |
| **1 个 Block** | 最常见，结构化 Op 的标准形态 | `scf.for`、`scf.while`、`func.func`（含实现时）、`scf.if` 的 then/else 各一个 Region，每个 Region 各 1 个 Block |
| **N 个 Block (N > 1)** | CFG（控制流图）风格 | 见下方多 Block 详解 |

**什么情况下一个 Region 会有多个 Block？**

核心原因：**结构化控制流** 被 **Lowering 为显式 CFG（控制流图）** 后，原本一个 Block 能表达的线性逻辑，需要拆成多个 Basic Block，并通过 `cf.br` / `cf.cond_br` 连接。

典型场景：

1. **结构化 Loop 展开为 CFG**（如 `scf.for` → `cf` dialect）：

```
// 原始：一个 Block 的 scf.for
scf.for %iv = %lb to %ub step %step {
  %val = arith.addi %iv, %cst : index
  scf.yield
}

// Lowering 后：一个 Region 多个 Block
^bb0:                            // Block 0: loop pre-header
  %init = ...                    // 初始化
  cf.br ^bb1(%lb)

^bb1(%iv: index):                // Block 1: loop header / condition check
  %cond = arith.cmpi ult, %iv, %ub : index
  cf.cond_br %cond, ^bb2, ^bb3

^bb2:                            // Block 2: loop body
  %val = arith.addi %iv, %cst : index
  %next = arith.addi %iv, %step : index
  cf.br ^bb1(%next)

^bb3:                            // Block 3: loop exit
  ...
  cf.br ^bb4                     // 回到外部 Region
```

2. **条件分支**（如 `scf.if` 展开）：

```mlir
// 原始
scf.if %cond {
  "then_block"()
} else {
  "else_block"()
}

// 展开后：每个 Region 多个 Block 的 CFG
// then Region:
//   ^bb0: cf.cond_br %cond_sub, ^then_entry, ^then_exit
//   ^then_entry: ...
//   ^then_exit: cf.br ^merge
//
// else Region:
//   ...
```

3. **`UnrealizedConversionCastOp` 临时拆分**：DialectConversion 过程中，为适配类型转换，pass 可能插入临时 Block。

4. **手动构造的 CFG-style Op**：某些方言的 Op 天然就是多 Block Region（如 `gpu.launch` 的 body、`async.execute`）。

**区分 Region 多 Block 和 Op 多 Region**

不要混淆这两个概念：

```
Op 多 Region:                     Region 多 Block:
┌─ scf.if ──────────────┐        ┌─ one Region ─────────────┐
│ Region 0 (then):      │        │ ┌─ Block 0 ────┐         │
│   ┌─ Block ──┐       │        │ │ op           │         │
│   │ op       │        │        │ │ cf.br ^bb1   │         │
│   │ scf.yield│        │        │ └──────────────┘         │
│ Region 1 (else):      │        │ ┌─ Block 1 ────┐         │
│   ┌─ Block ──┐       │        │ │ op           │         │
│   │ op       │        │        │ │ cf.cond_br ..│         │
│   │ scf.yield│        │        │ └──────────────┘         │
└───────────────────────┘        └──────────────────────────┘
```

**代码中如何遍历多 Block Region？**

```cpp
// 遍历 Region 内的所有 Block
Region *region = forOp.getBody();
for (Block &block : *region) {
  // 每个 Block 内的 Operation
  for (Operation &op : block) {
    // ...
  }
  // Block 终结者
  Operation *terminator = block.getTerminator();
}

// 按索引访问特定 Block
Block *firstBlock = &region->front();   // 第 0 个
Block *lastBlock  = &region->back();    // 最后 1 个
Block *thirdBlock = &(*region)[2];      // 第 2 个（0-based）
```

**要点总结**

- **Block 数量取决于 Op 处于哪个抽象层次**：结构化层次（`scf`、`func`）多为 1 Block；CFG 层次（`cf`、`llvm`）多为多 Block。
- **Lowering 过程是将 1 Block → N Block 的主要驱动力**：当你看到 `scf.for` 的 Region 突然冒出多个 Block 时，说明某个 pass 已经做了结构化 → CFG 的转换。
- **`getBody()` 只返回第一个 Region / 第一个 Block**，若 Region 有多个 Block，需要显式遍历或索引。

#### Triton 开发者实战指南：我能假设一个 Region 一个 Block 吗？

| 你在写哪个级别的 Pass | 假设 1 Region 1 Block 安全吗？ | 理由 |
|----------------------|-------------------------------|------|
| **Triton / TritonGPU 方言**（你看到的 Utility.cpp） | ✅ **安全** | Triton 自己的 Op 都是结构化 Op，body Region 必定正好 1 个 Block。所以 `*forOp.getBody()` 取 Block 没问题 |
| **SCF 方言**（`scf.for`、`scf.while`、`scf.if`） | ✅ **安全** | SCF 是结构化控制流方言，每个 Region 固定 1 个 Block |
| **cf 方言**（`cf.br`、`cf.cond_br`） | ❌ **不安全** | CFG 风格，1 个 Region 多个 Block，必须遍历 |
| **llvm 方言** | ❌ **不安全** | 也是 CFG 风格，多 Block |
| **通用 Pass**（不限定 dialect） | ❌ **不安全** | 你无法预知输入是结构化还是 CFG，应使用通用遍历 |

**实用规则**：

1. **在 Triton 自己的 pass 里**（如 `triton/lib/Dialect/TritonGPU/Transforms/`），你看到的 Op 要么是 Triton/TritonGPU 方言，要么是 `scf`/`arith` 等结构化方言，**safe to assume 1 Region 1 Block**。代码 `Block &block = *forOp.getBody()` 是惯例写法。
2. **一旦 pass 名字包含 "to-cfg"、"convert-scf-to-cfg"、"lower-to-llvm"**，该 pass 之后的 IR 就是多 Block 的，要改用 `for (Block &b : *region)` 遍历。
3. **不确定时写通用遍历**：`for (Block &b : forOp.getRegion()) { for (Operation &op : b) { ... } }` 总是正确的。

### 3.2 Value 与 Operation：SSA def-use 链

你选中的代码：

```cpp
DenseSet<Value> aliveValues;      // 存 Value（SSA 值）
SmallVector<Value> queue;          // 待传播的 Value 队列
```

这里 `Value` 和 `Operation` 是 MLIR 中 **def-use（定义-使用）** 关系的两端。

#### 关系总览

```
Operation (定义者)
  │
  ├── getResult(i) ──→ Value (OpResult) ──→ getDefiningOp() ──→ Operation
  │     （Op 的第 i 个返回值）                     ↑
  │                                              │
  │                                              └── 只有 OpResult 才有定值 Op
  │
  └── getOperand(j) ──→ Value (使用) ←── getUsers() ──→ 使用此 Value 的 Operation
        （Op 的第 j 个操作数）
                                     ↑
                                     │
                          Value 有两种来源：
                          ├── OpResult：Op 的计算结果
                          └── BlockArgument：Block 的入口形参（如 scf.for 的迭代变量）
```

#### 两种 Value

| Value 类型 | C++ 子类 | 来源 | `getDefiningOp()` |
|-----------|----------|------|-------------------|
| **OpResult** | `mlir::OpResult` | 某个 Operation 的返回值 | 返回那个 `Operation*` |
| **BlockArgument** | `mlir::BlockArgument` | Block 的形式参数（如 `scf.for` 的 `%iv`、`%arg`） | 返回 `nullptr` |

```mlir
// BlockArgument 示例
%result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %init) -> (i32) {
  //  ↑                ↑                            ↑
  //  │                └── %arg 是 BlockArgument     └── %iv 是 BlockArgument
  //  └── %result 是 scf.for 这个 Op 的 OpResult

  %sum = arith.addi %arg, %cst : i32
  // ↑                    ↑
  // └── OpResult         └── %arg 是 BlockArgument（被使用）
  //     （arith.addi                                   ）
  //      的 OpResult）

  scf.yield %sum : i32
}
```

#### def-use 遍历

```cpp
// 从 Value 找到使用它的所有 Operation（"use"）
Value val = ...;
for (auto &use : val.getUses()) {         // 每个 use 带 owner + operand index
  Operation *userOp = use.getOwner();      // 使用 val 的 Op
}

// 等价便捷写法
for (Operation *userOp : val.getUsers()) { // 直接遍历使用此值的 Op
  // ...
}

// 从 Operation 找到它定义的值
Operation *op = ...;
for (Value result : op->getResults()) {   // Op 的所有返回值
  // ...
}

// 从 Value 获取所在的 Region（回溯容器链）
Region *region = val.getParentRegion();  // 若 val 是 OpResult → 定义 Op 所在的 Region
                                         // 若 val 是 BlockArgument → 所属 Block 所在的 Region
```

#### Utility.cpp 中的实战用法

```cpp
auto markLive = [&](Value val) {
  // 检查 val 是否属于当前 forOp 的子树内
  if (!forOp->isAncestor(val.getParentRegion()->getParentOp()))
    return;
  //            ↑                              ↑
  //            │   val.getParentRegion() → val 所在的 Region
  //            │                              （对 OpResult：定义 Op 所在的 Region）
  //            │                              （对 BlockArgument：所属 Block 的 Region）
  //            └── getParentOp() → 拥有该 Region 的 Operation
  //                                 整个链回溯到 val 所属的"最内层 Op"
  //
  // 如果这个最内层 Op 不是 forOp 的后代，说明 val 定义在 forOp 外部，跳过

  // 标记为存活
  if (aliveValues.insert(val).second)
    queue.push_back(val);
};
```
#### 可视化参考

> 本节的完整关系图以 Canvas 形式提供，可在 IDE 中打开：
> [mlir-value-ops-relation.canvas.tsx](/root/.cursor/projects/root-triton-workspace/canvases/mlir-value-ops-relation.canvas.tsx)

关系图（文字版）：

```text
┌─────────────┐
│  Operation  │  ← producer（生产值）
│  (producer) │
└──────┬──────┘
       │ getResult(i)
       ▼
┌──────────────┐    ┌──────────────┐
│   OpResult   │    │ BlockArgument│  ← Value 的两种出身
│   (a Value)  │    │   (a Value)  │
└──────┬───────┘    └──────┬───────┘
       │ is a / inherits   │ is a / inherits
       ▼                   ▼
┌─────────────────────────────┐
│          Value              │  ← 抽象基类
│  (abstract base)            │
└──────────────┬──────────────┘
               │ getOperand(j)
               ▼
┌─────────────────┐   ← consumer（消费值）
│   Operation     │
│  (consumer)     │
└─────────────────┘

反向 API（虚线）:
  OpResult ──getDefiningOp()──→ Operation (producer)
  Value    ──getUsers()───────→ Operation[] (consumers)
```

#### 要点总结

- **Value 是"数据"的抽象**，**Operation 是"计算"的抽象**。Op 消费 Value（operand），生产 Value（result）。
- **Value 有两种出身**：Op 算出来的（OpResult），或 Block 入口参数（BlockArgument）。只有 OpResult 有 `getDefiningOp()`。
- **def-use 链**是双向的：从 Value → `getUsers()` 找到所有消费者；从 Op → `getResults()` / `getOperand()` 找到所有产/消 Value。
- **`val.getParentRegion()`** 不管 Value 是哪一种，都能回溯到它所在的 Region，进而通过 `getParentOp()` 找到拥有这个 Region 的 Op——这是 SSA 值在 IR 树上的"定位"方式。

### 3.3 方言（Dialect）

命名空间式前缀：`arith.addi`、`func.func`、`scf.for`。每个方言是一组 **协同设计的 op（及可选 type/attribute）**，在 **Context** 里注册后解析、打印。

### 3.4 Trait 与 Interface

- **Trait**：op 上的 **静态** 性质（是否 terminator、是否幂等等）。  
- **Interface**：跨方言 **统一行为**（如 `CallOpInterface`、`CallableOpInterface`），供 **CallGraph、Inliner** 等通用 Pass 使用。详见 [100_llvm_basic.md §5.3](100_llvm_basic.md)。

### 3.5 Type 与 Attribute

- **Type**：值的类型（`i32`、`tensor<...>`、`!llvm.ptr` 等）。  
- **Attribute**：编译期常量信息（维度、内存布局、符号名等），挂在 op 或类型上。

### 3.6 Pass 与 Pattern

- **Pass**：遍历 **Module / Operation** 等锚点，做分析或变换。  
- **DialectConversion / RewritePattern**：声明式 **模式匹配 + 重写**，是 **Lowering** 与 **规范化** 的主力。

---

## 4. 怎么用（命令行与日常操作）

### 4.1 `mlir-opt`

对 `.mlir` 跑 **Pass pipeline**，是学 MLIR 最常用的工具。

```bash
# 查看全部 pass
mlir-opt --help

# 示例：规范化 + 打印
mlir-opt input.mlir -pass-pipeline="builtin.module(func.func(canonicalize))" -o out.mlir

# 调试：打印 IR 变化（需 Debug 构建时常更全）
mlir-opt ... -mlir-print-ir-before-all  # 等 flag 视版本而定，以 --help 为准
```

**习惯**：从 **`mlir/test`** 里抄一条带 `RUN:` 的管线，再改自己的文件试验。

### 4.2 `mlir-translate`

在 **MLIR** 与 **其他格式** 间转换，例如 **LLVM Dialect → LLVM IR**（具体子命令以 `--help` 为准，不同版本选项名可能调整）。

### 4.3 其他实用工具（视构建启用）

- **`mlir-reduce`**：缩小触发 bug 的 IR。  
- **`tblgen-lsp-server`（若启用）**：编辑 `.td` 时补全。

### 4.4 阅读与书写 IR

- **文本 `.mlir`**：人类编辑、测试、FileCheck。  
- **C++ Builder API**：在 Pass 里 **`OpBuilder`** 创建 op。  
- **通用格式 / 调试**：`-mlir-print-op-generic` 等查看 **无方言简写** 的形式，利于理解 **底层 opcode**。

---

## 5. 开发时会碰到什么

### 5.1 写 Pass

- 注册：`PassRegistration` 或 **TableGen `Passes.td`**（生成 `createXxxPass`）。  
- 实现：对 **锚点 operation**（如 `builtin.module`）做 `walk`，或用 **接口** 过滤。  
- 分析：复用 **`AnalysisManager`**（如 **支配树、CallGraph**）。

### 5.2 写方言 Op（ODS）

用 **`.td`** 描述 op，**`mlir-tblgen`** 生成 C++ 类与 **builder**。不要手改 **`.inc`**。流程与反查 `.td` 见 [100_llvm_basic.md §5.4](100_llvm_basic.md) 与 **§7**。

### 5.3 Lowering

- **同一 MLIR 内**：`ConversionTarget` + **模式** 把 `high_dialect.op` 换成 `low_dialect.op`。  
- **到 LLVM**：常先降到 **`llvm` dialect**，再 **translate** 到 LLVM IR。

### 5.4 测试

- **`lit` + `FileCheck`**：`mlir/test` 下大量 `.mlir`，**`// RUN: mlir-opt %s ...`**。学习时 **照抄 RUN 行** 最快。

---

## 6. 源码树里该看哪里（`llvm-project/mlir`）

| 路径 | 内容 |
|------|------|
| `include/mlir/IR` | Operation、Region、Block、Builder、Context 等核心 |
| `include/mlir/Transforms` | 通用 Pass 声明（如 Inliner、Canonicalize） |
| `lib/Transforms` | 通用 Pass 实现 |
| `include/mlir/Dialect/*` | 各方言的 op 定义、Lowering |
| `include/mlir/Interfaces` | Call、ControlFlow、InferTypeOpInterface 等 |
| `docs/` | 官方文档源文件，与网页同步 |

---

## 7. 学习路径建议（可并行）

1. **能读懂 `.mlir` 文本**：`func`、`arith`、`scf` 小例子；理解 **Region / Block**（见 [100_llvm_basic.md](100_llvm_basic.md)）。  
2. **跑通 `mlir-opt` pipeline**：从 `mlir/test` 抄 `RUN`，改输入观察输出。  
3. **读一个简单 RewritePattern**：如某方言里 **单条 op 的 lowering**。  
4. **读一个 TableGen 定义的 op**：`.td` → 生成类 → 在测试里如何 `CHECK`。  
5. **再读通用 Pass**：如 **Canonicalize、Inliner**，理解 **Interface** 与 **CallGraph**。  
6. 若做 **Triton / AI 编译器**：在 MLIR 通识之上，跟你们仓库的 **专用 dialect 与 pipeline**。

官方入口：**MLIR 文档站点**（LLVM 官网下 *MLIR* 章节）中的 *LangRef*、*Dialects*、*Writing a Pass*、*Pattern Rewriter*。

---

## 8. 易混点备忘

- **Lowering** vs **Translation**：前者多在 **MLIR 内部**换 op；后者常指 **MLIR → LLVM IR 文件或内存表示**。  
- **Dialect 名** vs **op 名**：`arith` 是方言，`arith.addi` 是完整 op 名。  
- **`scf` vs `cf`**：`scf` 多为 **结构化** 控制流；`cf` 为 **显式 CFG**（`br` 等），降级过程中常出现。

---

## 9. 小结

**MLIR** = **带方言的可扩展多层 IR + Pass/Pattern 基础设施**。**用**：`mlir-opt` / `mlir-translate` + 测试管线。**学**：先 IR 结构再工具再 Pass/ODS。**开发**：ODS 写 op、Pattern 做 lowering、必要时实现 **DialectInterface** 以接入通用变换。

将本文与 [100_llvm_basic.md](100_llvm_basic.md) 对照阅读，可覆盖 **MLIR + LLVM 仓库协作** 的常见读码需求。
