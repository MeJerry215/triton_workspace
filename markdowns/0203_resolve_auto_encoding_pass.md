# ResolveAutoEncodings Pass 分析

## 一、概述

`GluonResolveAutoEncodingsPass` 是 Triton Gluon 方言中的一个 Module 级 Pass，其核心目标是将张量上残留的**符号化 `AutoEncodingAttr`** 通过 `set_auto_layout` 锚点传播解析为具体编码，然后删除所有 `set_auto_layout` 操作。

> **Pass 名称（CLI）**：`gluon-resolve-auto-encodings`
>
> **依赖方言**：`TritonGPUDialect`
>
> **操作粒度**：`mlir::ModuleOp`

### 与 `InferCoalescedEncodingsPass` 的区别

| 特性 | `InferCoalescedEncodingsPass` | `ResolveAutoEncodingsPass` |
|------|-------------------------------|----------------------------|
| 处理目标 | `CoalescedEncodingAttr` → `BlockedEncodingAttr` | `AutoEncodingAttr` → 任意具体编码 |
| 种子来源 | 分析 ptr 的 contiguity 推导最佳 layout | 从 `set_auto_layout` 的结果类型获取已有编码 |
| 锚点操作 | 对负载/存储生成种子编码 | 以 `set_auto_layout` 为锚点 |
| 副作用 | 更新 `set_auto_layout` 的 result encoding | **删除** `set_auto_layout` 操作 |

---

## 二、功能详解

### 2.1 整体流程

```
          ModuleOp
             │
             ▼
  ┌─────────────────────────┐
  │      inferAutoLayout    │  ← 核心：以 set_auto_layout 为种子传播 layout
  │   (for each function)   │
  └─────────┬───────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │  Cleanup set_auto_layout│  ← 删除所有 set_auto_layout 操作
  └─────────┬───────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │  doubleCheckEncodings   │  ← 校验：确保所有 AutoEncoding 被解析
  └─────────────────────────┘
             │
             ▼
       ModuleOp (已解析完毕)
```

#### `runOnOperation()` 实现

```cpp
void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // 步骤 1：推断 layout
    if (failed(inferAutoLayout(m)))
      return signalPassFailure();

    // 步骤 2：清理 set_auto_layout 操作
    m.walk([&](gluon::SetAutoLayoutOp op) {
      assert(op.getSrc().getType() == op.getType());  // src 应已被解析
      op.getResult().replaceAllUsesWith(op.getSrc());
      op->erase();
    });

    // 步骤 3：校验
    if (failed(doubleCheckEncodings(m, isAutoEncodingTensorType)))
      return signalPassFailure();
}
```

三个步骤是串行的：先传播，再清理，最后校验。任何一步失败都会导致 pass 失败。

---

### 2.2 核心步骤 1：`inferAutoLayout`

| 文件 | `ResolveAutoEncodings.cpp`，第 25-41 行 |
|------|----------------------------------------|
| 签名 | `LogicalResult inferAutoLayout(ModuleOp &mod)` |

#### 算法分解

```
for each function in module:
    │
    ├── Step 1: 收集种子编码
    │            └─ 遍历所有 gluon::SetAutoLayoutOp
    │            └─ 对于每个 op：seed = {op.getSrc(), op.getType().getEncoding()}
    │            └─ src value 当前类型带有 AutoEncodingAttr
    │            └─ result encoding 是具体的编码（如 BlockedEncodingAttr）
    │
    └── Step 2: 调用 inferLayout(func, isAutoEncodingTensorType, seedEncodings)
                 └─ 向后传播：从 seed 向 def chain 传播具体编码
                 └─ 向前传播：从 seed 向 use chain 传播具体编码
                 └─ 一次传播后，所有经过路径的 AutoEncodingAttr 都被替换为具体编码
```

#### 种子收集

```cpp
func.walk([&](gluon::SetAutoLayoutOp op) {
    seedEncodings.push_back({op.getSrc(), op.getType().getEncoding()});
});
```

种子是一个 pair：**（当前还是 auto-encoding 的 value，目标具体编码）**。这个 pair 告诉 `inferLayout`："请把这个具体编码传播给这个 value"。

**种子来源的典型情况**：

```
                         set_auto_layout
  tensor<...x#gluon.auto_encoding> ──────────────────> tensor<...x#ttg.blocked<...>>
       ↑ (src, 种子 value)                                  ↑ (result encoding, 种子 encoding)
```

`inferLayout` 从种子出发：
- **向后传播**：找到 src value 的定义操作，把具体编码应用到 src 的定义处
- **向前传播**：沿着 src value 的使用链扩散

---

### 2.3 核心步骤 2：清理 `set_auto_layout` 操作

```cpp
m.walk([&](gluon::SetAutoLayoutOp op) {
    assert(op.getSrc().getType() == op.getType());   // 传播后 src 和 result 类型应一致
    op.getResult().replaceAllUsesWith(op.getSrc());   // 所有使用 result 的地方改为使用 src
    op->erase();                                      // 删除该 op
});
```

在 layout 传播完成之后，`set_auto_layout` 的 src 和 result 类型应当完全相同（src 已被更新为具体编码）。此时这个操作变成了纯 identity，可以安全删除。

#### Before 和 After 对比

**清理前**（传播已完成为例）：

```
%1 = gluon.set_auto_layout %0 : tensor<128xi32, #ttg.blocked<...>> -> tensor<128xi32, #ttg.blocked<...>>
%2 = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.blocked<...>>
// ... 使用 %1 和 %2 ...
```

**清理后**：

```
%2 = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.blocked<...>>
// ... 直接使用 %0 替代 %1 ...
```

---

### 2.4 核心步骤 3：`doubleCheckEncodings`

| 文件 | `InferLayoutUtils.cpp`，第 224-249 行 |
|------|--------------------------------------|

同 `InferCoalescedEncodingsPass` 中使用的 `doubleCheckEncodings`，用于校验所有符号化编码被解析。

使用 `isAutoEncodingTensorType` 作为 typeCheck 回调：

```cpp
bool isAutoEncodingTensorType(Type ty) {
    auto tensorTy = dyn_cast<RankedTensorType>(ty);
    return tensorTy && isa<gluon::AutoEncodingAttr>(tensorTy.getEncoding());
}
```

校验内容：
1. 所有操作的返回类型中不能有 `AutoEncodingAttr`
2. 所有 block 参数类型中不能有 `AutoEncodingAttr`

如果有任何残留的 `AutoEncodingAttr`，pass 会报错并失败。

---

### 2.5 `SetAutoLayoutOp` 的定义与验证

| 文件 | `triton/include/triton/Dialect/Gluon/IR/GluonOps.td` |
|------|--------------------------------------------------------|

**Tablegen 定义**：

```tablegen
def Gluon_SetAutoLayoutOp : Gluon_Op<"set_auto_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType]> {
  let summary = "set auto encoding to a concrete encoding type";

  let arguments = (ins TT_Tensor:$src);
  let results   = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Attribute":$encoding, "Value":$value)>
  ];

  let hasVerifier = 1;
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}
```

**Verifier 检查**（`Dialect.cpp` 第 126-136 行）：

```cpp
LogicalResult SetAutoLayoutOp::verify() {
  // 1. src 必须具有 AutoEncodingAttr
  if (!isa<gluon::AutoEncodingAttr>(getSrc().getType().getEncoding())) {
    return emitOpError("input tensor must have an auto layout type");
  }
  // 2. result 必须有具体编码（不能是 auto）
  auto dstEncoding = getType().getEncoding();
  if (!dstEncoding)
    return emitOpError("result tensor must have an encoding");
  if (isa<gluon::AutoEncodingAttr>(dstEncoding))
    return emitOpError("result type must not be auto layout");
  return success();
}
```

三种违规情况：
- src 不是 auto encoding → 错误 "input tensor must have an auto layout type"
- result 没有 encoding → 错误 "result tensor must have an encoding"
- result 也是 auto encoding → 错误 "result type must not be auto layout"

---

### 2.6 Layout 传播过程（`inferLayout`）

`inferAutoLayout` 内部调用 `inferLayout`，其传播算法与 `InferCoalescedEncodingsPass` 完全相同。详见 [`202_infer_coalesced_encoding_pass.md`](202_infer_coalesced_encoding_pass.md#22-核心步骤-1infercoalescedlayout) 的 2.2.3 节。

关键点回顾：

| 概念 | 说明 |
|------|------|
| **Worklist 定点迭代** | 从种子出发，反复处理待传播值直到收敛 |
| **前向传播** | 沿 use-def chain 向用户传播，调用 `inferDstEncoding` |
| **后向传播** | 向定义操作传播，调用 `inferSrcEncoding` |
| **冲突处理** | `mayVary` 标志允许冲突；`JoinOp`/`SplitOp`/`ReshapeOp`/`CatOp`/`TransOp` 设为 `mayVary=true` |
| **类型更新** | 收敛后统一调用 `val.setType()` 更新所有值的类型 |

---

## 三、Pass 在 Pipeline 中的位置

### `gluon_to_ttgir` 完整管线（NVIDIA CUDA 后端）

```
  顺序 | Python API                              | MLIR pass 名
  ------+------------------------------------------+-----------------------------
    1   | passes.gluon.add_inliner                 | gluon-inline
    2   | passes.gluon.add_infer_coalesced_encodings | gluon-infer-coalesced-encodings
    3   | passes.gluon.add_resolve_auto_encodings  | gluon-resolve-auto-encodings   ← 当前 Pass
    4   | nvidia.passes.ttnvgpuir.add_tma_lowering | TMA lowering
    5   | passes.gluon.add_canonicalizer           | gluon-canonicalize
    6   | passes.common.add_sccp                   | SCCP
    7   | passes.ttir.add_loop_aware_cse           | loop-aware CSE
    8   | passes.gluon.add_canonicalizer           | gluon-canonicalize
    9   | passes.ttgpuir.add_combine_tensor_select_and_if | combine tensor select and if
```

### 数据流图

```
 用户源码 (@gluon.jit)
      │
      ▼
  AST → Gluon MLIR (带 AutoLayout / CoalescedLayout)
      │
      ▼
  ┌────────────────────────────────────────────────┐
  │              gluon_to_ttgir                     │
  │                                                  │
  │  1. gluon-inline                   inline helpers │
  │  2. gluon-infer-coalesced-encodings  CL → BL     │
  │  3. gluon-resolve-auto-encodings     Auto 传播 +   │  ← 当前 Pass
  │                                      set_auto_layout 消除 │
  │  4. TMA lowering / canonicalize / SCCP / CSE     │
  └──────────────────────────────────────────────────┘
      │
      ▼
  TTGIR (所有 layout 已具体化)
```

管线实现代码（`compiler.py` 第 320-337 行）：

```python
def gluon_to_ttgir(self, src, metadata, options, capability):
    mod = src
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    passes.gluon.add_inliner(pm)
    passes.gluon.add_infer_coalesced_encodings(pm)
    passes.gluon.add_resolve_auto_encodings(pm)      # ← 第 3 个 pass
    nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
    passes.gluon.add_canonicalizer(pm)
    passes.common.add_sccp(pm)
    passes.ttir.add_loop_aware_cse(pm)
    passes.gluon.add_canonicalizer(pm)
    passes.ttgpuir.add_combine_tensor_select_and_if(pm)

    pm.run(mod, 'gluon_to_ttgir')
    return mod
```

---

## 四、Python 前端中的 `AutoLayout` 与 `set_auto_layout`

### 4.1 `AutoLayout` 类

| 文件 | `_layouts.py`，第 24-34 行 |
|------|---------------------------|

```python
@dataclass(frozen=True)
class AutoLayout(DistributedLayout):
    def _to_ir(self, builder):
        return builder.get_auto_layout()
    def mangle(self):
        return "AL"
    @property
    def rank(self):
        raise ValueError("AutoLayout has no rank")
```

`AutoLayout` 是一个符号化 layout，无 rank 信息，最终在 IR 中表现为 `#gluon.auto_encoding`。

### 4.2 `set_auto_layout` 语义函数

| 文件 | `_semantic.py`，第 320-328 行 |
|------|-------------------------------|

```python
def set_auto_layout(self, value, layout):
    src_ty = value.type
    _check(isinstance(layout, DistributedLayout),
           lambda: f"set_auto_layout must set to a distributed layout but got {layout}")
    _check(isinstance(src_ty.layout, AutoLayout),
           lambda: f"set_auto_layout input must have auto layout but got {value.type.layout}")
    handle = self.builder.create_set_auto_layout(layout._to_ir(self.builder), value.handle)
    res_ty = ttgl.distributed_type(src_ty.element_ty, src_ty.shape, layout)
    return self.tensor(handle, res_ty)
```

### 4.3 自动 `set_auto_layout` 插入

当 `AutoLayout` 张量与具体 layout 张量进行 **broadcast** 时，frontend 自动插入 `set_auto_layout`：

```python
# _semantic.py，第 197-204 行
is_lhs_auto = isinstance(lhs_ty.layout, AutoLayout)
is_rhs_auto = isinstance(rhs_ty.layout, AutoLayout)
if is_lhs_auto and not is_rhs_auto:
    lhs = self.set_auto_layout(lhs, rhs_ty.layout)
elif is_rhs_auto and not is_lhs_auto:
    rhs = self.set_auto_layout(rhs, lhs_ty.layout)
```

---

## 五、IR 示例（来自实际测试 dump）

以下 IR 来自 `ops/pass/203_resolve_auto_encoding.py` 中 `auto_layout_2d_kernel` 的真实 dump。
测试参数：XBLOCK=128, YBLOCK=256, numWarps=4, threadsPerWarp=32。

### Before（`GluonResolveAutoEncodingsPass` 运行前）

```
// -----// IR Dump Before GluonResolveAutoEncodingsPass (gluon-resolve-auto-encodings) //----- //
```

此时 `CoalescedEncodingAttr` 已被 `GluonInferCoalescedEncodingsPass` 解析为具体的 `BlockedEncodingAttr`，
但 auto-encoded 张量上仍有 `#gluon.auto_encoding`，`set_auto_layout` 操作仍然存在：

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 4],
                         threadsPerWarp = [1, 32],
                         warpsPerCTA = [2, 2],
                         order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                  ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @auto_layout_2d_kernel(...) {
    // arange 使用 AutoLayout → 仍是 #gluon.auto_encoding
    %indices_x = tt.make_range {end = 128 : i32, start = 0 : i32}
        : tensor<128xi32, #gluon.auto_encoding>
    %indices_y = tt.make_range {end = 256 : i32, start = 0 : i32}
        : tensor<256xi32, #gluon.auto_encoding>

    // expand_dims → 保持 auto_encoding
    %indices_x_2d = tt.expand_dims %indices_x {axis = 1 : i32}
        : tensor<128xi32, #gluon.auto_encoding> -> tensor<128x1xi32, #gluon.auto_encoding>
    %indices_y_2d = tt.expand_dims %indices_y {axis = 0 : i32}
        : tensor<256xi32, #gluon.auto_encoding> -> tensor<1x256xi32, #gluon.auto_encoding>

    // broadcast → 保持 auto_encoding
    %mask = arith.cmpi slt, ...  : tensor<128x256xi1, #gluon.auto_encoding>

    // 指针运算 → auto_encoding
    %ptr = tt.splat %in_ptr : !tt.ptr<f32>
        -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
    %in_ptrs = tt.addptr %ptr, %in_offsets
        : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>

    // set_auto_layout 桥接到具体 layout — 这是 pass 的种子来源
    %in_ptrs_layouted = gluon.set_auto_layout %in_ptrs
        : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
      -> tensor<128x256x!tt.ptr<f32>, #blocked>

    // load 使用具体 layout 的指针
    %value = tt.load %in_ptrs_layouted, %mask
        : tensor<128x256xf32, #blocked>

    // 数学运算 → 继承 #blocked
    %sin_val = math.sin %value : tensor<128x256xf32, #blocked>
    %max_val = arith.maxnumf %sin_val, %cst : tensor<128x256xf32, #blocked>

    // store 也有类似的 set_auto_layout 桥接
    %out_ptrs_layouted = gluon.set_auto_layout %out_ptrs
        : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
      -> tensor<128x256x!tt.ptr<f32>, #blocked>
    %out_mask_layouted = gluon.set_auto_layout %mask
        : tensor<128x256xi1, #gluon.auto_encoding>
      -> tensor<128x256xi1, #blocked>
    tt.store %out_ptrs_layouted, %max_val, %out_mask_layouted
        : tensor<128x256x!tt.ptr<f32>, #blocked>
  }
}
```

关键观察：
1. **`#gluon.auto_encoding` 仍然遍布**：arange、expand_dims、broadcast、addptr 等操作的张量类型都是 auto_encoding
2. **`set_auto_layout` 操作仍存在**：三个 `gluon.set_auto_layout` op 桥接 auto 张量到具体 `#blocked` layout
3. **`#blocked` 已就位**：`#blocked` 已由前序 pass（infer-coalesced-encodings）推导出来

### After（`GluonResolveAutoEncodingsPass` 运行后）

```
// -----// IR Dump After GluonResolveAutoEncodingsPass //----- //
```

所有 `#gluon.auto_encoding` 已被传播为具体编码，`set_auto_layout` 操作已删除：

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 4],
                         threadsPerWarp = [1, 32],
                         warpsPerCTA = [2, 2],
                         order = [1, 0]}>
#slice_d0 = #ttg.slice<{dim = 0, parent = #blocked}>
#slice_d1 = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes { ... } {
  tt.func public @auto_layout_2d_kernel(...) {
    // arange 已被传播为具体 slice layout
    %indices_x = tt.make_range {end = 128 : i32, start = 0 : i32}
        : tensor<128xi32, #slice_d0>
    %indices_y = tt.make_range {end = 256 : i32, start = 0 : i32}
        : tensor<256xi32, #slice_d1>

    // expand_dims 和 broadcast 继承具体 layout
    %indices_x_2d = tt.expand_dims %indices_x {axis = 1 : i32}
        : tensor<128xi32, #slice_d0> -> tensor<128x1xi32, #blocked>
    %indices_y_2d = tt.expand_dims %indices_y {axis = 0 : i32}
        : tensor<256xi32, #slice_d1> -> tensor<1x256xi32, #blocked>

    // set_auto_layout 已被删除 — 指针直接使用具体 layout
    %ptr = tt.splat %in_ptr : !tt.ptr<f32>
        -> tensor<128x256x!tt.ptr<f32>, #blocked>
    %in_ptrs = tt.addptr %ptr, %in_offsets
        : tensor<128x256x!tt.ptr<f32>, #blocked>

    // load 直接使用 auto-encoding-free 的指针
    %value = tt.load %in_ptrs, %mask
        : tensor<128x256xf32, #blocked>
    // ...
  }
}
```

关键观察：
| 方面 | Before | After |
|------|--------|-------|
| **arange** | `#gluon.auto_encoding` | `#slice<dim=0/1, parent=#blocked>` |
| **2D 张量** | `#gluon.auto_encoding` | `#blocked` |
| **`set_auto_layout`** | 3 个 `gluon.set_auto_layout` | **已全部删除** |
| **auto_encoding** | 大量残留 | **零残留** |
| **编码来源** | 用户显式 `set_auto_layout` + 前序 pass | 传播后一致性 |

### 投影关系详解

```
原始 set_auto_layout 关系（Before）:

  %indices_x (1D, 128)        ──expand_dims──>  %indices_x_2d (2D, 128×1)
  #gluon.auto_encoding                           #gluon.auto_encoding
       │                                                       │
       └──────── set_auto_layout(%in_ptrs, #blocked) ──────────┘
                   ↓ 种子: src→#blocked

传播结果（After）:

  %indices_x (1D, 128)        ──expand_dims──>  %indices_x_2d (2D, 128×1)
  #slice<dim=0, parent=#blocked>                #blocked
       ↑ (后向传播)                                     ↑ (前向传播)
       │                                                    │
       └────────── inferLayout 从种子出发 ──────────────────┘
                   #blocked → 第 0 维切片 → #slice
```

从 `%in_ptrs` 作为种子（`#blocked`）出发：
- **前向传播**：`tt.load`、`math.sin`、`arith.maxnumf` → 都得到 `#blocked`
- **后向传播**：`tt.addptr` → 两个 operand 都得到 `#blocked`
- **继续后向**：`tt.expand_dims` → `inferSrcEncoding` 推断出 1D 输入的 slice layout
- **继续后向**：`tt.make_range` → 使用 slice layout

这就是为什么 1D 张量得到 `#slice`（父 layout 的投影），而 2D 张量得到完整的 `#blocked`。

---

## 六、关键数据结构总结

| 数据结构 | 作用 |
|----------|------|
| `AutoEncodingAttr` | 符号化编码标记，表示"尚无具体 layout 决策" |
| `SetAutoLayoutOp` | 桥接操作：输入为 auto-encoding，输出为具体编码 |
| `BlockedEncodingAttr` | 具体的 GPU 线程块布局，指定 `sizePerThread`、`threadsPerWarp`、`warpsPerCTA`、`order` |
| `SliceEncodingAttr` | 1D 切片 layout，引用一个 nD 父 layout 在指定 dim 上的投影 |
| `LayoutInfo` | 传播过程中的中间数据结构，包含 `encoding` 和 `mayVary` 标志 |
| `seedEncodings` | 种子编码集合：`SmallVector<std::pair<Value, Attribute>>` |

---

## 七、设计要点与约束

### 7.1 `set_auto_layout` 的三种来源

| 来源 | 示例 | 说明 |
|------|------|------|
| **用户显式** | `ttgl.set_auto_layout(ptr, ttgl.CoalescedLayout())` | 开发者在 kernel 中手写，最常见 |
| **前端自动插入** | broadcast 时 AutoLayout → concrete | `_semantic.py` 在 binary op 时自动插入 |
| **前序 pass 生成** | InferCoalescedEncodings 更新 result encoding | 前序 pass 修改 `set_auto_layout` 的结果类型 |

### 7.2 传播终止条件

`inferLayout` 不会无限传播。终止条件包括：
- **非 tensor 值**：scalar、pointer 等不参与 layout 传播
- **非 auto-encoding 张量**：`typeCheck` 回调（`isAutoEncodingTensorType`）只选中 auto-encoding 的张量
- **function boundary**：如果 function 参数/返回值有 auto-encoding，pass 会在入口处报错（要求先 inline）
- **Worklist 耗尽**：定点迭代收敛

### 7.3 校验严格性

`doubleCheckEncodings` 检查 **所有** 操作的返回类型和 **所有** block 参数类型。任何残留的 `AutoEncodingAttr` 都会导致 pass 失败。

这意味着：**如果前序 pass 或前端没有正确设置具体编码锚点**，`ResolveAutoEncodingsPass` 会直接失败，不会静默留下未解析的 encoding。

### 7.4 与 InferCoalescedEncodings 的协作关系

```
InferCoalescedEncodingsPass               ResolveAutoEncodingsPass
        │                                          │
        │  处理 CoalescedEncodingAttr               │  处理 AutoEncodingAttr
        │                                          │
        │  set_auto_layout(src, CoalescedLayout)    │  set_auto_layout(src, BlockedLayout) — 已被更新
        │      │                                    │      │
        │      ▼  (pass 更新 result encoding)       │      ▼
        │  set_auto_layout(src, BlockedLayout)      │  (传播 + 删除 set_auto_layout)
        │      │                                    │
        └──────┴────────────────────────────────────┘
               │
               ▼
         所有 layout 具体化
```

两个 pass **互补**：InferCoalescedEncodings 处理 `CoalescedEncodingAttr` → 具体编码；ResolveAutoEncodings 处理 `AutoEncodingAttr` → 具体编码。前者在 `set_auto_layout` 的结果端写入具体编码，后者利用这个具体编码向 src 端传播并最终删除桥接。

---

## 八、Pass 注册

```tablegen
def GluonResolveAutoEncodingsPass
    : Pass<"gluon-resolve-auto-encodings", "mlir::ModuleOp"> {
  let summary = "Resolve auto encodings to concrete encodings";
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}
```

| 字段 | 值 |
|------|-----|
| Pass 名称 | `gluon-resolve-auto-encodings` |
| 操作类型 | `mlir::ModuleOp` |
| 依赖方言 | `TritonGPUDialect` |

### Python 绑定

```cpp
// triton/python/src/passes.cc，第 117-118 行
ADD_PASS_WRAPPER_0("add_resolve_auto_encodings",
                   gluon::createGluonResolveAutoEncodingsPass);
```

使用方式：

```python
passes.gluon.add_resolve_auto_encodings(pm)
```

---

*本文档对应源代码：[ResolveAutoEncodings.cpp](../../triton/lib/Dialect/Gluon/Transforms/ResolveAutoEncodings.cpp)*
