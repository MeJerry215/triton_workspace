# MLIR `InlinerPass`（`inline`）实现说明

> **命名说明**：本文档分析的是 **MLIR** 中的 `InlinerPass`（TableGen 里注册为 pass 名 `"inline"`，通常 `mlir-opt -inline`）。它与 **LLVM IR** 层 `opt` 里的 `AlwaysInlinerPass` / `InlinerLegacyPass` 是不同栈上的两套机制；若你关心的是 LLVM bitcode 上的内联，需要另看 `llvm/lib/Transforms/InlineSimple.cpp` 等路径。

---

## 1. 总体架构：Pass 很薄，核心在 `Inliner` + `inlineCall`

`InlinerPass` 本身只做四件事：

1. 继承生成的 `impl::InlinerBase<InlinerPass>`，挂上 **pass 选项**（pipeline 字符串、`max-iterations`、`inlining-threshold`、按 op 名的 pipeline 列表等）。
2. 在 `initializeOptions` 里把字符串选项解析进 **`InlinerConfig`**（默认 pipeline、按方言 op 名的 `OpPassManager` 映射、最大 SCC 内迭代次数等）。
3. 在 `runOnOperation` 里取 **`CallGraph` 分析**，校验根 operation 带 **`OpTrait::SymbolTable`**（否则无法解析符号调用）。
4. 构造 **`Inliner`** 对象并调用 **`doInlining()`**；真正算法在 `lib/Transforms/Utils/Inliner.cpp`，单次调用的 **体替换**在 `lib/Transforms/Utils/InliningUtils.cpp` 的 **`mlir::inlineCall`**。

数据流可以概括为：

```text
InlinerPass::runOnOperation
  → Inliner(op, cg, pass, am, runPipelineHelper, config, profitabilityCb)
  → Inliner::doInlining()
       → 按 CallGraph 的 SCC 自底向上遍历
       → 每个 SCC：optimizeSCC（嵌套 pipeline） + inlineCallsInSCC（inlineCall）
  → 最后 eraseDeadCallables()
```

---

## 2. `InlinerPass` 类（`InlinerPass.cpp`）

### 2.1 默认优化 pipeline

无自定义 `default-pipeline` 时，构造函数里使用的默认 pipeline 仅为：

- **`createCanonicalizerPass()`**（`defaultInlinerOptPipeline`）。

即在真正尝试 inline 每个 SCC 里的 callable 之前，会先对「可跑子 pipeline 的」节点做一次规范化，便于后续代价判断和 IR 形状稳定。

### 2.2 `runPipelineHelper`

`Inliner` 驱动在子 operation 上跑嵌套 `OpPassManager` 时，需要调用 `Pass::runPipeline`，而该 API 在 `Pass` 里是 **protected**。因此 `InlinerPass` 提供静态方法 **`runPipelineHelper`**，内部 **`cast<InlinerPass>(pass).runPipeline(pipeline, op)`**，把「当前 pass 实例」桥给 `Inliner` 使用。

### 2.3 收益判断：`isProfitableToInline`

Pass 把 **`inlining-threshold`** 转成一个 lambda 传给 `Inliner`：

- 阈值为 **0**：永远不认为划算（不 inline）。
- 阈值为 **`-1U`（无符号最大值，默认）**：永远认为划算（只要后面 `shouldInline` 等检查通过）。
- 否则：统计 **caller 与 callee 各自 callable `Region` 内的 operation 数量**，计算  
  `ratio = calleeOps * 100 / callerOps`，若 `ratio <= threshold` 则认为可 inline。  
  **空 callee**（caller 侧计数为 0 时的边界）会直接认为可 inline。

这是 **非常粗粒度** 的体积比启发式，与 LLVM 里基于 cost model 的内联不同。

### 2.4 `runOnOperation` 摘要

1. `CallGraph &cg = getAnalysis<CallGraph>();`
2. 检查 `getOperation()->hasTrait<OpTrait::SymbolTable>()`，否则报错并 `signalPassFailure()`。
3. `Inliner inliner(op, cg, *this, getAnalysisManager(), runPipelineHelper, config, profitabilityCb);`
4. `if (failed(inliner.doInlining())) signalPassFailure();`

### 2.5 `initializeOptions`

- 若设置了 **`default-pipeline`** 字符串：在 config 里放一个 lambda，内部 **`parsePassPipeline`** 解析到为该 callable op 名构造的 `OpPassManager`。
- 把 **`op-pipelines`** 列表里非空的 pipeline 按 **anchor op 名**放进 `llvm::StringMap<OpPassManager>`。
- 同步 **`max-iterations`** 到 config。

---

## 3. `InlinerConfig`（`include/mlir/Transforms/Inliner.h`）

承载：

| 字段 | 含义 |
|------|------|
| `defaultPipeline` | 对没有专用 `opPipelines` 条目的 callable，使用的通用嵌套 pipeline 构造器 |
| `opPipelines` | `op 名 → OpPassManager`，按 callable 类型选不同 pipeline |
| `maxInliningIterations` | **单个 SCC 内**「先 optimize 再 inline」循环的上限（默认来自 pass 选项，TableGen 默认 4） |
| `cloneCallback` | 内联时 **克隆还是 splice** callee 的 `Region`；默认实现：`shouldCloneInlinedRegion` 为真则 `cloneInto`，否则把 block 链表 **splice** 到插入点之后 |
| `canHandleMultipleBlocks` | 是否允许把 **多 block callee** 内联进结构上只支持单 block 的 caller（默认 false，需方言/回调配合） |

---

## 4. `Inliner::doInlining` 与 SCC 算法（`Inliner.cpp`）

### 4.1 入口

- 构造 **`Inliner::Impl`**、**`InlinerInterfaceImpl`**（继承 `InlinerInterface`）、**`CGUseList`**（跟踪可丢弃符号节点的「使用」，因为 MLIR 里很多引用是 **`SymbolRefAttr`**，没有 SSA `Use` 链表）。
- 调用 **`runTransformOnCGSCCs`**：对 **`llvm::scc_iterator`** 得到的 **CallGraph 强连通分量（SCC）** 逐个处理。顺序是 **SCC 的后序**，等价于从调用图 **叶到根** 的倾向，使内联决策能自下而上传播（文件头注释所述）。

### 4.2 每个 SCC：`Impl::inlineSCC`

循环直到达到 **`maxInliningIterations`**：

1. **`optimizeSCC`**  
   对当前 SCC 中满足条件的节点（非 external、无子节点、父 op 带 **`IsIsolatedFromAbove`**）并行跑 **`optimizeCallable`**：  
   按 callable 的 op 名在 `opPipelines` 里找 pipeline，找不到则用 **`defaultPipeline`** 填一个 `OpPassManager`，再 **`runPipelineHelper(pass, pm, callableOp)`**。  
   之后 **`useList.recomputeUses`** 更新丢弃型符号的使用计数。

2. **`inlineCallsInSCC`**  
   收集本 SCC 各节点 callable **Region 顶层 block** 里的 **`CallOpInterface`**（**不**深入嵌套 callgraph 节点，嵌套由别的 SCC 处理）。  
   对每个 **`ResolvedCall`（call + sourceNode + targetNode）**：若 **`shouldInline`** 为真，则调用 **`mlir::inlineCall`**；成功则更新 `CGUseList`、erase call；若 **单 use 且 callee 可 discard**，可 **原地移动 region** 并标记节点待删。

若某轮 **`inlineCallsInSCC`** 没有内联任何调用，内层循环提前结束（`break`）。

全图结束后 **`inlinerIface.eraseDeadCallables()`** 删掉标记为 dead 的 callable op。

### 4.3 `shouldInline`（除收益外的合法性 / 安全性）

在 **`isProfitableToInline`** 之前会先排除多种情况，例如：

- call 是 **terminator**（不支持）。
- callee 的 callgraph 上存在指向 **自身** 或 **caller 对应节点** 的边（避免简单自递归 / A→B→A 形态的内联）。
- callee 的 `Region` 是 **call 所在 region 的祖先**（防止把祖先内联进子孙，破坏结构）。
- **多 block callee** 且未配置 `canHandleMultipleBlocks` 时，检查 caller 所在 op 是否 **同名** 或 **不** 带 **`SingleBlock`** trait 等启发式。
- 最后才调用 **`isProfitableToInline`**。

另外还有 **`inlineHistory`**：对因某次内联而新收集到的 call，记录其来自哪条内联链，避免 **同一 callee 在同一条 inline 历史上被反复展开** 导致无限内联。

---

## 5. 真正把「函数体」接到调用点：`mlir::inlineCall`（`InliningUtils.cpp`）

流程概要：

1. 检查 callee **region 非空**，call 与 entry block **参数个数**、**结果个数** 与 callable 类型一致。
2. 用 **`DialectInlinerInterface`** 在 operand/result 类型不一致时尝试 **`materializeCallConversion`** 插入 cast；失败则清理并 `failure()`。
3. **`interface.isLegalToInline(call, callable, shouldBeCloned)`**：由各方言注册的 **内联合法性** 决定（默认方言可全返回 false，需 `func` 等扩展注册）。
4. **`inlineRegionImpl`**：在 call 之后插入点，用 **`cloneCallback`** 克隆或移动 region，并通过方言接口处理 **terminator**（单 block / 多 block 不同 `handleTerminator` 重载），把 **return 类结果** 接到 **原 call 的 result**。

因此：**InlinerPass 不负责「怎么把 return 变成 branch」这类细节**；它依赖 **`InlinerInterface`** / **`DialectInlinerInterface`** 与通用的 **region inliner**。

---

## 6. Pass 选项（`Passes.td` 中 `def Inliner`）

| 选项 | 作用 |
|------|------|
| `default-pipeline` | 默认嵌套 pipeline 字符串，默认 `"canonicalize"` |
| `op-pipelines` | 多个 `dialect.op(pipeline)` 形式，为特定 callable op 指定 pipeline |
| `max-iterations` | SCC 内 optimize+inline 最大轮数，默认 4 |
| `inlining-threshold` | callee/caller **op 数量比** 百分上限；`-1U` 表示不限制 |

---

## 7. 小结表

| 层次 | 职责 |
|------|------|
| **`InlinerPass`** | 选项 → `InlinerConfig`；取 `CallGraph`；构造 `Inliner` 并运行 |
| **`Inliner` / `Impl`** | SCC 顺序、每 SCC 嵌套 canonicalize/自定义 pipeline、收集 call、历史与 use 跟踪、调用 `inlineCall` |
| **`mlir::inlineCall`** | 类型映射、合法性、region 克隆/移动、terminator 处理 |
| **方言 `DialectInlinerInterface`** | 是否合法、terminator 如何改、类型转换 |

若你后续想深入某一块（例如 **`CallGraph` 如何建**、**`func.call` 如何注册 inliner**、或与 **LLVM 后端** 的衔接），可以指定文件或 IR 样例再问。
