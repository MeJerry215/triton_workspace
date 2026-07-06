# `gluon-canonicalize` Pass 分析

```python
# triton/third_party/nvidia/backend/compiler.py:329, 332
def gluon_to_ttgir(self, src, metadata, options, capability):
    mod = src
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    passes.gluon.add_inliner(pm)                         # 第 1 个 pass
    passes.gluon.add_infer_coalesced_encodings(pm)        # 第 2 个 pass
    passes.gluon.add_resolve_auto_encodings(pm)           # 第 3 个 pass
    nvidia.passes.ttnvgpuir.add_tma_lowering(pm)          # 第 4 个 pass
    passes.gluon.add_canonicalizer(pm)                    # ← 第 5 个 pass
    passes.common.add_sccp(pm)                            #
    passes.ttir.add_loop_aware_cse(pm)                    #
    passes.gluon.add_canonicalizer(pm)                    # ← 第 8 个 pass（第二次）
    passes.ttgpuir.add_combine_tensor_select_and_if(pm)   #
```

`gluon-canonicalize` 在 gluon pipeline 中**被调用两次**：
- **第 5 个 pass**：在 `gluon-resolve-auto-encodings` 和 `gluon-tma-lowering` 之后，作为第一轮清理
- **第 8 个 pass**：在 SCCP（稀疏条件常量传播）和 CSE（公共子表达式消除）之后，作为第二轮清理

---

## 1. Pass 概述

| 属性 | 值 |
|------|-----|
| **Pass 名称（CLI）** | `gluon-canonicalize` |
| **TableGen 定义** | `def GluonCanonicalize: Pass<"gluon-canonicalize">` |
| **C++ 实现** | `triton/lib/Dialect/Gluon/Transforms/Canonicalize.cpp` |
| **Python 绑定** | `ADD_PASS_WRAPPER_0("add_canonicalizer", gluon::createGluonCanonicalize)` |
| **依赖方言** | `arith::ArithDialect`, `cf::ControlFlowDialect`, `scf::SCFDialect` |

**与 `passes.common.add_canonicalizer(pm)` 的区别**：

| | `passes.gluon.add_canonicalizer(pm)` | `passes.common.add_canonicalizer(pm)` |
|--|---------------------------------------|---------------------------------------|
| 底层 pass | `GluonCanonicalize` | MLIR 标准 `CanonicalizerPass` |
| TritonGPU layout 相关 pattern | ❌ **排除**（不处理 `ConvertLayoutOp`） | ✅ 包含 |
| 适用范围 | 布局解析阶段的保守清理 | 布局已确定的通用化简 |

关键注释原文：

```cpp
// Populate select Triton canonicalization patterns. The important patterns to
// EXCLUDE are those that modify layouts, especially `ConvertLayoutOp`
// patterns.
```

---

## 2. `runOnOperation()` 完整流程

```cpp
void Canonicalize::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(&getContext());

  // ── 1. Dialect 级 Canonicalization ──
  ctx->getLoadedDialect<arith::ArithDialect>()->getCanonicalizationPatterns(patterns);
  ctx->getLoadedDialect<scf::SCFDialect>()->getCanonicalizationPatterns(patterns);
  ctx->getLoadedDialect<cf::ControlFlowDialect>()->getCanonicalizationPatterns(patterns);

  // ── 2. Operation 级 Canonicalization（逐 op 粒度）──
  for (auto op : ctx->getRegisteredOperationsByDialect("arith"))
    op.getCanonicalizationPatterns(patterns, ctx);
  for (auto op : ctx->getRegisteredOperationsByDialect("scf"))
    op.getCanonicalizationPatterns(patterns, ctx);
  for (auto op : ctx->getRegisteredOperationsByDialect("cf"))
    op.getCanonicalizationPatterns(patterns, ctx);

  // ── 3. scf.for 死参数消除 ──
  populateForOpDeadArgumentElimination(patterns);

  // ── 4. Triton 专用 Canonicalization（精选，不含 layout 变换）──
  LoadOp::getCanonicalizationPatterns(patterns, ctx);
  StoreOp::getCanonicalizationPatterns(patterns, ctx);
  BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
  ExpandDimsOp::getCanonicalizationPatterns(patterns, ctx);
  ttg::WarpSpecializeOp::getCanonicalizationPatterns(patterns, ctx);

  // ── 5. 贪心应用 ──
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
```

### 架构示意图

```
                    ┌──────────────────────────────────────┐
                    │          applyPatternsGreedily        │
                    │  (反复应用所有 pattern，直至 IR 不变) │
                    └──────────────────────────────────────┘
                                    │
         ┌──────────────┬───────────┼───────────┬─────────────────┐
         ▼              ▼           ▼           ▼                 ▼
   ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐  ┌────────────────┐
   │  Arith   │  │   SCF    │  │   CF   │  │  Triton  │  │  scf.for 死参  │
   │  化简    │  │  化简    │  │  化简  │  │  精选    │  │  数消除        │
   │          │  │          │  │        │  │  Load    │  │                │
   │ +0, ×1  │  │ 恒真分支 │  │ 死块   │  │  Store   │  │  ForOpDeadArg  │
   │ 常量折叠 │  │ 死循环   │  │ 简化   │  │  Broadcast│  │  Elimination    │
   │          │  │ 消除     │  │        │  │  ExpandDims│  │                │
   │          │  │          │  │        │  │  WarpSpecialize│             │
   └──────────┘  └──────────┘  └────────┘  └──────────┘  └────────────────┘
```

---

## 3. 各 Canonicalization Pattern 详解

### 3.1 arith 算术化简

算术 op 的标准化简，由 `arith::ArithDialect` 提供：

| Before | After | 说明 |
|--------|-------|------|
| `%r = arith.addi %x, %c0` | `%x` | 加 0 |
| `%r = arith.subi %x, %c0` | `%x` | 减 0 |
| `%r = arith.muli %x, %c1` | `%x` | 乘 1 |
| `%r = arith.divui %x, %c1` | `%x` | 除以 1 |
| `%r = arith.addi %c1, %c2` | `%c3 = arith.constant 3` | 常量折叠 |
| `%r = arith.cmpi eq, %x, %x` | `%true` | 自比较恒真 |
| `%r = arith.select %true, %a, %b` | `%a` | 选择折叠 |

### 3.2 scf 结构化控制流化简

由 `scf::SCFDialect` 提供：

| Before | After | 说明 |
|--------|-------|------|
| `scf.if %true { ... } else { ... }` | 仅 then 分支 | 恒真分支消除 |
| `scf.if %false { ... } else { ... }` | 仅 else 分支 | 恒假分支消除 |
| `scf.for %iv = %c0 to %c0 step %c1` | 消除 | 空循环消除 |
| `scf.while %true { ... }` | 简化 | 恒真条件简化 |

### 3.3 cf 底层控制流化简

由 `cf::ControlFlowDialect` 提供，处理非结构化控制流：

| Before | After | 说明 |
|--------|-------|------|
| `cf.br ^bb1` 且 `^bb1` 只有一个前驱 | 内联 block | 无条件分支折叠 |
| `cf.cond_br %true, ^bb1, ^bb2` | `cf.br ^bb1` | 条件分支恒真 |
| `cf.cond_br %false, ^bb1, ^bb2` | `cf.br ^bb2` | 条件分支恒假 |

### 3.4 `populateForOpDeadArgumentElimination` — scf.for 死参数消除

消除 `scf.for` 中未被循环体使用的迭代参数。详见 [`201_inline_pass.md`](201_inline_pass.md) 的详细分析。

```
Before:                                          After:
%r:2 = scf.for %iv = %lb to %ub step %st         %r:1 = scf.for %iv = %lb to %ub step %st
    iter_args(%dead = %init_dead, %live = %c0)        iter_args(%live = %c0)
    → (i32, i32) {                                     → (i32) {
  %1 = arith.addi %live, %c1                      %1 = arith.addi %live, %c1
  scf.yield %dead, %1       ════════════►          scf.yield %1
}                                                 }
```

### 3.5 `CanonicalizeMaskedLoadPattern` — 恒真/恒假 mask 的 load 化简

**源码位置**：`triton/lib/Dialect/Triton/IR/Ops.cpp:87-123`

```
// load(ptr, splat(true), ...)  →  load(ptr, ...)           [去掉 mask]
// load(ptr, splat(false), other, ...)  →  other            [替换为 other 值]
```

**代码逻辑**：

```cpp
struct CanonicalizeMaskedLoadPattern : public OpRewritePattern<LoadOp> {
  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto mask = loadOp.getMask();
    if (!mask) return failure();                               // 无 mask → 不匹配

    auto constantMask = mask.getDefiningOp<arith::ConstantOp>();
    if (!constantMask) return failure();                       // mask 不是常数 → 不匹配

    auto splatMask = mlir::dyn_cast<SplatElementsAttr>(constantMask.getValue());
    if (!splatMask) return failure();                          // mask 不是 splat → 不匹配

    if (splatMask.getSplatValue<IntegerAttr>().getValue() == true) {
      // mask = splat(true) → 生成无 mask 的新 load
      rewriter.replaceOpWithNewOp<LoadOp>(
          loadOp, loadOp.getType(), loadOp.getPtr(), Value(), Value(),
          loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    } else {
      // mask = splat(false) → 替换为 other 值
      auto otherVal = loadOp.getOther();
      if (!otherVal) return failure();                         // 无 other → 不能随便替换
      rewriter.replaceOp(loadOp, otherVal);
    }
    return success();
  }
};
```

**Before/After 示例**：

```
// Before: mask = splat(true) 的全 1 mask → 去掉 mask
%mask = arith.constant dense<true> : tensor<128xi1>
%val = tt.load %ptr, %mask      →    %val = tt.load %ptr

// Before: mask = splat(false) → 直接返回 other
%false = arith.constant dense<false> : tensor<128xi1>
%fallback = arith.constant dense<0.0> : tensor<128xf32>
%val = tt.load %ptr, %false, %fallback    →    %val = %fallback
```

### 3.6 `CanonicalizeMaskedStorePattern` — 恒真/恒假 mask 的 store 化简

**源码位置**：`triton/lib/Dialect/Triton/IR/Ops.cpp:154-183`

```
// store(ptr, value, splat(true), ...)  →  store(ptr, value, ...)  [去掉 mask]
// store(ptr, value, splat(false), ...) →  [无操作，删除 store]
```

**代码逻辑**：

```cpp
struct CanonicalizeMaskedStorePattern : public OpRewritePattern<StoreOp> {
  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto mask = storeOp.getMask();
    if (!mask) return failure();

    auto constantMask = mask.getDefiningOp<arith::ConstantOp>();
    if (!constantMask) return failure();

    auto splatMask = mlir::dyn_cast<SplatElementsAttr>(constantMask.getValue());
    if (!splatMask) return failure();

    if (splatMask.getSplatValue<IntegerAttr>().getValue() == true) {
      // mask = splat(true) → 去掉 mask
      rewriter.replaceOpWithNewOp<StoreOp>(
          storeOp, storeOp.getPtr(), storeOp.getValue(),
          storeOp.getCache(), storeOp.getEvict());
    } else {
      // mask = splat(false) → 删除整个 store
      rewriter.eraseOp(storeOp);
    }
    return success();
  }
};
```

### 3.7 BroadcastOp 化简

**源码位置**：`triton/lib/Dialect/Triton/IR/Canonicalize.td`

通过 TableGen 定义的 DRR（Declarative Rewrite Rules）模式：

```
// broadcast(splat(x)) → splat(x)
def BroadcastSplatPattern :
    Pat<(TT_BroadcastOp (TT_SplatOp $x)), (TT_SplatOp $x)>;

// broadcast(broadcast(x)) → broadcast(x)
def BroadcastBroadcastPattern :
    Pat<(TT_BroadcastOp (TT_BroadcastOp $x)), (TT_BroadcastOp $x)>;
```

| Before | After | 说明 |
|--------|-------|------|
| `%s = tt.splat %x : i32 → tensor<1xi32>`<br>`%b = tt.broadcast %s : tensor<1xi32> → tensor<128xi32>` | `%b = tt.splat %x : i32 → tensor<128xi32>` | broadcast + splat 合并为 splat |
| `%b1 = tt.broadcast %x`<br>`%b2 = tt.broadcast %b1` | `%b2 = tt.broadcast %x` | 连续 broadcast 合并 |

### 3.8 ExpandDimsOp 化简

**源码位置**：`triton/lib/Dialect/Triton/IR/Ops.cpp:740-786`

```
// expand_dims(splat(x)) → splat(x)                        [类型改为 expand 后的 shape]
// expand_dims(broadcast(x)) → broadcast(expand_dims(x))    [交换顺序以便后续合并]
```

**代码逻辑**：

```cpp
LogicalResult ExpandDimsOp::canonicalize(ExpandDimsOp op,
                                         PatternRewriter &rewriter) {
  auto definingOp = op.getSrc().getDefiningOp();
  if (!definingOp) return failure();

  // expand_dims(splat) → splat
  if (auto splat = dyn_cast<SplatOp>(definingOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(), splat.getSrc());
    return success();
  }

  // expand_dims(broadcast(x)) → broadcast(expand_dims(x))
  // 目的是让 broadcast 成为最外层操作，便于后续 broadcast 合并
  if (auto broadcast = dyn_cast<BroadcastOp>(definingOp)) {
    // ... 创建 expand_dims 再包 broadcast ...
    return success();
  }

  return failure();
}
```

| Before | After | 说明 |
|--------|-------|------|
| `%s = tt.splat %x : i32 → tensor<128xi32>`<br>`%e = tt.expand_dims %s {axis = 1} : tensor<128xi32> → tensor<128x1xi32>` | `%e = tt.splat %x : i32 → tensor<128x1xi32>` | expand_dims + splat 直接合并 |
| `%b = tt.broadcast %x : tensor<128xi32> → tensor<128x256xi32>`<br>`%e = tt.expand_dims %b {axis = 0}` | `%e2 = tt.expand_dims %x`<br>`%b2 = tt.broadcast %e2` | 交换顺序，让 broadcast 在最外层 |

### 3.9 WarpSpecializeOp 化简

由 `ttg::WarpSpecializeOp::getCanonicalizationPatterns` 提供，处理 warp specialization 相关的化简（如空 warp group 消除等）。

---

## 4. 关键设计决策：为什么排除 `ConvertLayoutOp`

`ConvertLayoutOp` 的 canonicalization pattern 会做大量的 layout 变换折叠，例如：

```
// cvt(cvt(x)) → cvt(x)
// cvt(reshape(x)) → reshape(cvt(x))
// cvt(splat(x)) → splat(x)     （!）
```

在 `gluon-canonicalize` 运行时，layout 刚被 `gluon-resolve-auto-encodings` 解析完毕。此时引入 layout 变换会：

1. **打乱已确定的 layout**：splat + cvt 的折叠会改变 splat 产生张量的 encoding
2. **引入不必要的 `ConvertLayoutOp`**：某些折叠模式会产生新的 layout 转换
3. **与后续 pass 冲突**：`ResolveAutoEncodings` 已做的工作可能被部分撤销

**安全原则**：在 layout 解析阶段使用**保守化简**，只做 `arith`/`scf`/`cf` 数学上的等价变换和访存 mask 消除，不碰 layout 相关模式。等到 layout 完全确定后，由 `passes.common.add_canonicalizer` 处理 layout 化简。

---

## 5. Pipeline 上下文

```
gluon_to_ttgir pipeline:
┌──────────────────────────────────────┐
│ gluon-inline                         │  内联所有 @gluon.jit 调用
├──────────────────────────────────────┤
│ gluon-infer-coalesced-encodings      │  CoalescedEncoding → BlockedEncoding
├──────────────────────────────────────┤
│ gluon-resolve-auto-encodings         │  AutoEncoding → 具体 layout
├──────────────────────────────────────┤
│ gluon-tma-lowering                   │  TMA 操作降级
├──────────────────────────────────────┤
│ ▸ gluon-canonicalize (第 1 次)       │  ← 本 pass：layout 解析后第一轮清理
├──────────────────────────────────────┤
│ common-sccp                          │  稀疏条件常量传播
├──────────────────────────────────────┤
│ ttir-loop-aware-cse                  │  循环感知 CSE
├──────────────────────────────────────┤
│ ▸ gluon-canonicalize (第 2 次)       │  ← 本 pass：SCCP/CSE 后第二轮清理
├──────────────────────────────────────┤
│ ttgpuir-combine-tensor-select-and-if │  select + if 合并
└──────────────────────────────────────┘
```

---

## 6. 总结

| 方面 | 说明 |
|------|------|
| **核心作用** | 在 layout 解析阶段做保守的 IR 化简，不碰 layout 变换 |
| **主要化简内容** | arith 算术简化、scf/cf 控制流简化、访存 mask 消除、broadcast/expand_dims 折叠、scf.for 死参数消除 |
| **两次调用的目的** | 第 1 次：layout 解析后的基础清理；第 2 次：SCCP/CSE 后的二次清理 |
| **设计哲学** | 化简+传播交替进行（canonicalize → SCCP → CSE → canonicalize），每次化简为下一次传播创造更简单的 IR |
| **与 common.canonicalizer 的关系** | 互补而非替代。gluon 版保守，common 版激进（含 layout 变换） |

---

*本文档对应源代码：[Canonicalize.cpp](../../triton/lib/Dialect/Gluon/Transforms/Canonicalize.cpp)*
