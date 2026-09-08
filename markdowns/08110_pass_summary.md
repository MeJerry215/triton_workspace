# TileLang Pass 总结

> 粗略过一遍 tilelang 编译 pipeline 中用到的 60+ 个 pass 分别是干什么用的。

TileLang 的编译 pipeline 由 **backend 无关** 的基础 pass 和 **backend 特有** 的 pass 组成。编译入口为 `tilelang/engine/lower.py` 中的 `lower()` 函数，它会根据 target 解析出对应的 `PassPipeline`（cuda / hip / metal / c / llvm），然后依次执行一串 pass。

## 为什么有 ~77 个 pass？

如果你用 `TL_ENABLE_DUMP_IR` dump 了 IR，会看到 77 个文件（000-076）。这个数字看起来大，但原因很合理：

| 原因 | 说明 | 占比估算 |
|------|------|----------|
| **Simplify 跑了很多次** | `Simplify` 在 pipeline 不同阶段反复运行（~9 次），每次清除前一个 pass 引入的表达式冗余 | ~12% |
| **条件性/后端特有 pass** | CUDA 有额外的 warp specialied、TMA、mbarrier、LDG/STG、Hopper intrin 等 pass，加起来 ~15 个 | ~20% |
| **TVM 框架内置 pass** | `pass_fn` 是 TVM PassContext 自动注入的框架级 pass（绑定 target、函数规范化等），dump 中看到 000/001/004/018/070 共 5 个 | ~7% |
| **成对出现的 pass** | 一些 pass 需要按不同 scope 分别执行，如 `tirx.Simplify` / `tirx.RemoveNoOp` / `tirx.Filter` 出现 2 次以上 | ~10% |
| **分阶段降低** | TileLang 的 IR 经过多层降低：tile op → buffer op → vectorized loop → intrinsic → PTX，每层都需要多个 pass | ~35% |
| **Host/Device 分离** | `SplitHostDevice` 之后，host 和 device 各有独立的后处理 pass | ~16% |

**实际唯一的 pass 种类约 40-45 个**，但很多 pass 在不同阶段重复执行，所以 dump 文件看起来有 77 个。

下面将所有 pass 按 **功能阶段** 分类说明。

---

## 目录

1. [Frontend / 前期准备](#1-frontend--前期准备)
2. [Pipeline Planning & Layout](#2-pipeline-planning--layout)
3. [Tile Op Lowering & Vectorization](#3-tile-op-lowering--vectorization)
4. [Memory & Buffer Management](#4-memory--buffer-management)
5. [Optimization: 循环展开、Storage Rewrite 等](#5-optimization-循环展开storage-rewrite-等)
6. [Thread-Level Allreduce & Intrinsic](#6-thread-level-allreduce--intrinsic)
7. [Host-Device Split & Codegen 准备](#7-host-device-split--codegen-准备)
8. [CUDA 特有 Pass](#8-cuda-特有-pass)
9. [ROCm / Metal / WebGPU / CPU 特有 Pass](#9-rocm--metal--webgpu--cpu-特有-pass)
10. [Legacy tirx / s_tir Pass](#10-legacy-tirx--s_tir-pass)
11. [Analysis / Debug Pass](#11-analysis--debug-pass)

---

## 1. Frontend / 前期准备

| Pass | 源 | 作用 |
|------|-----|------|
| `BindTarget` | `tirx.transform` | 给 PrimFunc 标注 target 信息，后续 pass 可以据此做 target 相关决策 |
| `MaterializeKernelLaunch` | `tilelang.transform` | 将 `T.Kernel` 发射的 thread_binding For 循环（blockIdx/threadIdx）具体化为后端形式：SIMT 后端转成 `thread_extent` AttrStmt；非 SIMT 后端将 blockIdx 展开为串行 For、threadIdx 忽略 |
| `LetInline` | `tilelang.transform` | 强制内联 Let/Bind 绑定。受 `tl.force_let_inline` 配置控制 |
| `AddWrapperForSingleBufStore` | `tilelang.transform` | 将 fragment buffer 的 `buf[0]` 单元素 store 包裹一层 T.Parallel 循环，确保后续 pass 能正确识别 |
| `LegalizeNegativeIndex` | `tilelang.transform` | 将 BufferLoad / BufferStore 中的负索引规范化为非负形式（加 `extent` 取模），消除边界歧义 |
| `VerifyParallelLoop` | `tilelang.transform` | 检查并行循环（T.Parallel）是否存在数据竞争。受 `tl.disable_data_race_check` 控制 |
| `InjectAssumes` | `tilelang.transform` | 向 TIR 注入 `assume` 调用（如 natural shape 边界条件），帮助 TVM prover 做简化。同时把 `Evaluate(Call(assume, ...))` 转为 AttrNode 形式 |
| `Simplify` | `tilelang.transform` | **核心简化 pass**。基于 TVM 的 Analyzer 做表达式简化、常量折叠、条件传播等。有丰富的子选项（如传递性不等式证明、boolean 转 AND-of-ORs 等） |

### 一次 Simplify 到底做了什么？

`Simplify` 是全程被反复调用的 pass。它使用 TVM 的 `arith::Analyzer`：

- **常量折叠**：`3 + 5 * 2` → `13`
- **约束传播**：已知 `i < 32`，则 `i >= 32` 变为 false
- **重写取模/除法**：`(i // 32) * 32 + i % 32` → `i`
- **边界条件 Assume**：对 `T.assume(n >= 0)` 之后的表达式做上下文相关的简化

---

## 2. Pipeline Planning & Layout

| Pass | 源 | 作用 |
|------|-----|------|
| `LayoutReducer` | `tilelang.transform` | 为 Reducer（如 `tl.sum`, `tl.max` 等）设置 layout，使后续布局推导能正确处理 reduce 操作 |
| `IfStmtBinding` | `tilelang.transform` | 规范化 if-without-else 的 AST 结构。同时可内联可重放的 Bind 赋值，让 copy 和 compute 暴露给 pipeline planning |
| `PipelinePlanning` | `tilelang.transform` | **Pipeline 规划**：分析循环体内的 global→shared copy 和 shared→fragment compute，决定如何分阶段（multi-buffering）、如何分配 shared memory 轮转 buffer |
| `InjectSoftwarePipeline` | `tilelang.transform` | 根据 PipelinePlanning 的结果，实际将循环改写为 software-pipelined 形式（prologue + steady state + epilogue），插入多缓冲、双缓冲 |
| `Simplify` | `tilelang.transform` | 第二次简化，清理 pipeline 引入的冗余 |

### 软件流水线（Software Pipeline）是怎么注入的？

输入一个带 `T.Pipelined(num_stages=N)` 的 for 循环，`PipelinePlanning` 会：

1. 识别 global→shared copy 操作
2. 为 shared buffer 分配 N 个 stage（多缓冲）
3. `InjectSoftwarePipeline` 将单循环体展开为：

```c
// Prologue: 预取前 N-1 个 stage
for s in 0..N-1:
   copy(global → shared[s])
   cp_async.commit()
// Steady state: 每个迭代 = 消费 + 预取下一个
for i in 0..M:
   cp_async.wait(N-1)          // 等之前发的拷贝完成
   compute(shared[(i+0)%N])     // 消费当前 stage
   copy(global → shared[(i+1)%N])  // 预取下一个
   cp_async.commit()
// Epilogue: 消费剩余 stage
for s in 0..N-1:
   cp_async.wait(s)
   compute(shared[?])
```

---

## 3. Tile Op Lowering & Vectorization

| Pass | 源 | 作用 |
|------|-----|------|
| `LayoutInference` | `tilelang.transform` | **Layout 推导**：分析 fragment 的访问模式和各线程的 index 映射，推出每个 fragment buffer 在寄存器中的排布方式（thread → index mapping）。这是 TileLang 的核心抽象之一 |
| `LayoutVisual` | `tilelang.analysis` | 调试用。将 `LayoutInference` 推导出的 layout 信息打印或可视化（PDF/PNG/SVG）。受 `tl.layout_visualization_enable` 控制 |
| `LowerTileOp` | `tilelang.transform` | 将高级 tile 操作（如 `T.gemm`, `T.copy`）lower 为低级的 buffer load/store + intrinsic call。这是 TileLang 前端到后端的分水岭 |
| `LowerL2Persistent` | `tilelang.cuda.transform` | CUDA 特有。Lower L2 persistent 映射 |
| `DecoupleTypeCast` | `tilelang.transform` | 解耦混合精度 vectorization 约束。当 vectorized 循环体内有 `cast`（如 `float32` ↔ `float4_e2m1fn`）时，插入 local buffer 做中转，使计算和内存访问可以各自用最优 vectorize 宽度 |

### DecoupleTypeCast 的例子

**Before：**

```python
for vec in T.vectorized(16):
    b[vec] = T.cast(a_frag[vec], "float4_e2m1fn")
```

**After：**

```python
cast_buf = T.alloc_buffer([16], "float4_e2m1fn", scope="local")
for vec in T.vectorized(16):
    cast_buf[vec] = T.cast(a_frag[vec], "float4_e2m1fn")  # compute
for vec_copy in T.vectorized(16):
    b[vec_copy] = cast_buf[vec_copy]                        # copy to memory
```

这样 `cast` 计算可以用一条宽指令，而 store 也可以用一条宽指令，不会被对方拖窄。

---

## 4. Memory & Buffer Management

| Pass | 源 | 作用 |
|------|-----|------|
| `LegalizeVectorizedLoop` | `tilelang.transform` | 检查 vectorized loop 是否合法（如保证对齐、连续访问），若不合法则降级为串行 |
| `LegalizeSafeMemoryAccess` | `tilelang.transform` | 在越界可能发生的 memory access 前插入边界检查（if guard）。受 `tl.disable_safe_memory_legalize` 控制 |
| `LowerAccessPtr` | `tilelang.transform` | 将 TileLang 前端 `tl.access_ptr` 下降为 TVM 标准 `tvm_access_ptr` builtin |
| `HoistNonRestrictParams` | `tilelang.transform` | 将 root-block 中关于参数非别名（__restrict__）的 annotation 提升到 PrimFunc attr |
| `PlanAndUpdateBufferAllocationLocation` | `tilelang.transform` | 为 buffer 分配规划位置（host side / device side），决定哪些 buffer 在哪儿申请 |
| `HoistGlobalBufferAllocations` | `tilelang.transform` | 将全局 buffer 的 alloc 提升到 block 顶部（host 端），使得后续 codegen 可以统一处理 |
| `LowerOpaqueBlock` | `tilelang.transform` | 将 SBlock（结构块 / opaque block）展开 / 消除，暴露内部的 statement |
| `FlattenBuffer` | `tilelang.transform` | 将多维 buffer access（如 `buf[i][j]`）展平为一维 offset 访问，简化后续 codegen |
| `ConfigIndexBitwidth` | `tilelang.transform` | 配置 index 的位宽（默认 32-bit）。**必须在 FlattenBuffer 之后**，因为 FlattenBuffer 会改写 index 计算 |
| `MergeSharedMemoryAllocations` | `tilelang.transform` | 合并所有 shared memory alloc 为单个大 buffer，减少 shared memory 碎片。支持 aggressive merge（通过 buffer lifetime 复用）。受 `tl.enable_aggressive_shared_memory_merge` 等控制 |

---

## 5. Optimization: 循环展开、Storage Rewrite 等

| Pass | 源 | 作用 |
|------|-----|------|
| `VectorizeLoop` | `tilelang.transform` | 将标记为 `vectorized` 的 for 循环向量化（生成 LLVM vector 或 CUDA 的向量 load/store）。受 `tirx.disable_vectorize` 控制 |
| `StorageRewrite` | `tilelang.transform` | **存储重写**（Memory promotion）：将 local buffer 改写为标量寄存器、将 shared memory buffer 的生命周期合并。是寄存器分配前的关键优化 |
| `LoopUnswitching` | `tilelang.transform` | **循环不变量外提**：将循环内条件不变的 if 提到循环外面，减少分支判断次数。如 `for i: if cond: A else: B` → `if cond: for i: A else: for i: B` |
| `UnrollLoop` | `tilelang.transform` | 根据配置（auto_max_step / auto_max_depth / auto_max_extent）展开循环。对局部访问（local access）总是展开 |
| `RenormalizeSplitPattern` | `s_tir.transform` | 将 `floordiv(floormod(x, M), N)` 的 split pattern 规范化为 `floormod(floordiv(x, N), M//N)` 形式，便于后续分析 |
| `RemoveNoOp` | `tirx.transform` | 删除无实际作用的 statement（如 `Evaluate(0)`, `if (false) { ... }` 等） |
| `HoistIfThenElse` | `s_tir.transform` | 将循环不变量 IfThenElse 外提到循环外（比 LoopUnswitching 更通用，处理任意嵌套结构） |
| `MergeIfStmt` | `tilelang.transform` | 合并相邻的 if 语句，减少分支 |
| `ThreadSync` | `tilelang.transform` | 在 parallel read/write 同一 shared buffer 之间插入 `__syncthreads()`。分别对 `"shared"` 和 `"shared.dyn"` 作用域各执行一次 |

---

## 6. Thread-Level Allreduce & Intrinsic

| Pass | 源 | 作用 |
|------|-----|------|
| `VerifyMemory` | `tirx.transform` | 验证内存访问的合法性（如 offset 不越界） |
| `AnnotateEntryFunc` | `tirx.transform` | 标注 entry function |
| `InferFragment` | `s_tir.transform` | 利用 tensor intrinsic 的 descriptor 推导 fragment 信息（如 MMA 的 A/B 矩阵在寄存器中的排布） |
| `LowerThreadAllreduce` | `tilelang.transform` | 将线程级别的 allreduce（如 `tl.sum` across threads）lower 为 warp shuffle / shared memory 实现 |
| `LowerIntrin` | `tilelang.transform` | 将 high-level intrinsic call lower 为 target-specific intrinsic |
| `LowerHopperIntrin` | `tilelang.cuda.transform` | Lower Hopper 特有 intrinsic（如 TMA、WGMMA 等） |
| `LowerLDGSTG` | `tilelang.cuda.transform` | 将 Ramp-based 的 global BufferLoad/Store lower 为 `ldg`/`stg` intrinsic（支持 32/64/128/256 位宽和 predicated 版本） |

---

## 7. Host-Device Split & Codegen 准备

| Pass | 源 | 作用 |
|------|-----|------|
| `AnnotateDeviceRegions` | `tilelang.transform` | 标注 device function region |
| `SplitHostDevice` | `tilelang.transform` | 将 IRModule 中 host 和 device 函数分离，输出两个 module |
| `AnnotateReadOnlyParams` | `tilelang.transform` | 分析参数中只读的 buffer，在 PrimFunc 上添加 `tl.readonly_param_indices` attr，使得 CUDA codegen 可发出 `const __restrict__` 指针，启用只读 cache |
| `MakePackedAPI` | `tilelang.transform` | 生成 TVM PackedFunc API 的 wrapper 函数（参数 packing/unpacking） |
| `LowerDeviceKernelLaunch` | `tilelang.transform` | 将 device kernel launch 构造（CallingConv.DEVICE_KERNEL_LAUNCH）lower 为 target 特定的 launch IR |
| `PersistThreadblock` | `tilelang.cuda.transform` | 将普通 threadblock 转换为 persistent threadblock 形式（持续消费 grid 直到耗尽） |

此外在 `host_codegen()` 阶段还有：

| Pass | 源 | 作用 |
|------|-----|------|
| `FP8StorageLegalize` | `tirx.transform` | 将 FP8 存储类型合法化（如转为 `uint8` 存储 + 类型信息） |
| `BF16StorageLegalize` | `tirx.transform` | 将 BF16 存储类型合法化 |
| `LowerTVMBuiltin` | `tirx.transform` | 将 TVM builtin（如 `tvm_call_packed`）lower 为具体调用 |
| `LowerCustomDatatypes` | `tirx.transform` | 下降自定义数据类型 |
| `CombineContextCall` | `tirx.transform` | 合并 context call |

---

## 8. CUDA 特有 Pass

CUDA pipeline 在公用 pass 基础上额外注册了以下 pass：

| Pass | 作用 |
|------|------|
| `ProducerConsumerWarpSpecialized` | **Warp Specialization**：在 tile-op level 将 pipelined loop 分离为 producer warp（负责 global→shared copy）和 consumer warp（负责 compute）。生产者消费者之间用 mbarrier 同步。仅在 Hopper (SM90+) + TMA 启用 |
| `LowerBlackwell2SM` | Lower Blackwell 2SM TCGEN5MMA 操作（一个 MMA 需要两个 SM 协作完成） |
| `LowerSharedTmem` | 将 shared TMEM（Tensor Memory）初始化 slot lower 为具体代码 |
| `LowerSharedBarrier` | 将 `T.alloc_barrier()` 分配的 barrier buffer lower 为硬件 mbarrier 指令。需要 SM90+ |
| `FuseMBarrierArriveExpectTx` | 将独立的 `expect_tx → TMA issue → arrive` 三段合并为 `arrive_and_expect_tx` 一条指令 |
| `MarkCudaSyncCalls` | 标记包含 `pdl_sync` / `pdl_trigger` 调用的函数，使后续 codegen 可以正确处理同步 |
| `InjectFenceProxy` | 注入 TMA 异步代理需要的 fence 指令（`fence_proxy_async`） |
| `InjectTcgen05Fence` | Blackwell (SM100+) 特有：在 tcgen05 同步边界（ThreadSync / mbarrier handoff）插入保守的 `tcgen05.fence` 指令 |
| `AnnotateWarpGroupRegAlloc` | 在 warp-specialized 函数的 producer/consumer 分支中注入 `set_max_nreg` 调用来控制寄存器分配 |
| `LowerHopperIntrin` | Lower Hopper 架构特有的 intrinsic（WGMMA、TMA 等） |
| `LowerLDGSTG` | 将 global memory load/store 下降为 `ldg`/`stg` PTX 指令 |
| `LowerL2Persistent` | Lower L2 persistent map |

---

## 9. ROCm / Metal / WebGPU / CPU 特有 Pass

### ROCm (`hip`)

pipeline 与 CUDA 几乎一致，但**没有**以下 CUDA 特有 pass：
- Warp Specialization
- WGMMA / TMA / Hopper 相关
- LDG/STG、Shared Barrier、mbarrier 等
- L2 Persistent

### Metal

| Pass | 源 | 作用 |
|------|-----|------|
| `MetalFragmentToSimdgroup` | `tilelang.metal.transform` | 在 LayoutInference 之前，将 `local.fragment` 的 GEMM accumulator 改写为 `metal.simdgroup`（Metal 的 SIMD 矩阵类型）。因为 simdgroup 矩阵是 opaque 的，LayoutInference 不应看到它 |

### WebGPU

pipeline 与 CPU 几乎一致，但 `MaterializeKernelLaunch( lower_thread_binding=True )`（SIMT 模式）。

### CPU (`c`, `llvm`)

pipeline 与 ROCm 几乎一致，但 `MaterializeKernelLaunch( lower_thread_binding=False )`（非 SIMT 模式）。

---

## 10. Legacy tirx / s_tir Pass

这些 pass 来自 TVM 的 `tirx.transform` 和 `s_tir.transform`，被 tilelang pipeline 直接调用：

| Pass | 模块 | 作用 |
|------|------|------|
| `BindTarget` | `tirx.transform` | 给 PrimFunc 设置 target |
| `Simplify` | `tirx.transform` | TIR 通用简化（常量折叠、代数化简、条件简化） |
| `NarrowDataType` | `tirx.transform` | 将表达式数据类型收窄到指定位宽（如 32 位），减少寄存器压力 |
| `RemoveNoOp` | `tirx.transform` | 移除无效语句（空 Evaluate、永假分支等） |
| `VerifyMemory` | `tirx.transform` | 验证内存访问正确性 |
| `AnnotateEntryFunc` | `tirx.transform` | 标注 module 的 entry function |
| `FP8StorageLegalize` | `tirx.transform` | 将 FP8 存储类型合法化 |
| `BF16StorageLegalize` | `tirx.transform` | 将 BF16 存储类型合法化 |
| `LowerTVMBuiltin` | `tirx.transform` | 将 TVM builtin 调用降低 |
| `LowerCustomDatatypes` | `tirx.transform` | 将自定义数据类型转换 |
| `CombineContextCall` | `tirx.transform` | 合并多个 context call 调用 |
| `Filter` | `tirx.transform` | 从 IRModule 中过滤出满足条件的函数 |
| `RenormalizeSplitPattern` | `s_tir.transform` | 规范化 `floordiv(floormod)` 为 `floormod(floordiv)` |
| `HoistIfThenElse` | `s_tir.transform` | 将循环不变量 IfThenElse 外提 |
| `InferFragment` | `s_tir.transform` | 推导 tensor intrinsic 的 fragment 信息 |

---

## 11. Analysis / Debug Pass

| Pass | 源 | 作用 |
|------|-----|------|
| `LayoutVisual` | `tilelang.analysis` | 可视化 LayoutInference 推导出的 fragment/register layout。支持 text / PDF / PNG / SVG 输出 |
| `PreLowerSemanticCheck` | `tilelang.engine` | 在 pipeline 执行前做 Python 端语义检查，如检测无效的 buffer 绑定等 |

---

## 附录：CUDA Pipeline 完整 Pass 序列

下面是 CUDA pipeline 实际执行顺序（`CUDAPassPipelineBody`），合计约 **55 个 pass** 调用（不含 host_codegen 阶段的 5 个）：

```
1.  BindTarget                          (tirx)
2.  MaterializeKernelLaunch             (tl)
3.  [LetInline]                         (tl, 可选)
4.  AddWrapperForSingleBufStore         (tl)
5.  LegalizeNegativeIndex               (tl)
6.  [VerifyParallelLoop]                (tl, 可选)
7.  InjectAssumes                       (tl)
8.  Simplify                            (tl)
9.  LayoutReducer                       (tl)
10. [ProducerConsumerWarpSpecialized]   (cuda, 可选)
11. LowerBlackwell2SM                   (cuda)
12. IfStmtBinding                       (tl)
13. PipelinePlanning                    (tl)
14. InjectSoftwarePipeline              (tl)
15. Simplify                            (tl)
16. LayoutInference                     (tl)
17. LayoutVisual                        (analysis)
18. LowerTileOp                         (tl)
19. LowerL2Persistent                   (cuda)
20. DecoupleTypeCast                    (tl)
21. LegalizeVectorizedLoop              (tl)
22. LegalizeSafeMemoryAccess            (tl)
23. LowerAccessPtr                      (tl)
24. Simplify                            (tl)
25. HoistNonRestrictParams              (tl)
---  CUDA-only passes ---
26. LowerSharedTmem                     (cuda)
27. PlanAndUpdateBufferAllocationLoc    (tl)
28. LowerSharedBarrier                  (cuda)
29. [FuseMBarrierArriveExpectTx]        (cuda, 仅 TMA)
--- 通用 backend passes ---
30. HoistGlobalBufferAllocations        (tl)
31. LowerOpaqueBlock                    (tl)
32. Simplify                            (tl)
33. NarrowDataType(32)                  (tirx)
34. FlattenBuffer                       (tl)
35. ConfigIndexBitwidth                 (tl)
36. Simplify                            (tirx)
37. VectorizeLoop                       (tl)
38. StorageRewrite                      (tl)
39. LoopUnswitching                     (tl)
40. UnrollLoop                          (tl)
41. RenormalizeSplitPattern             (s_tir)
42. Simplify                            (tirx)
43. RemoveNoOp                          (tirx)
44. HoistIfThenElse                     (s_tir)
45. VerifyMemory                        (tirx)
46. AnnotateEntryFunc                   (tirx)
47. InferFragment                       (s_tir)
48. LowerThreadAllreduce                (tl)
49. LowerLDGSTG                         (cuda)
50. LowerHopperIntrin                   (cuda)
51. AnnotateDeviceRegions               (tl)
52. SplitHostDevice                     (tl)
53. MarkCudaSyncCalls                   (cuda)
54. AnnotateReadOnlyParams              (tl)
55. MergeSharedMemoryAllocations        (tl)
56. InjectFenceProxy                    (cuda)
57. ThreadSync("shared")                (tl)
58. ThreadSync("shared.dyn")            (tl)
59. InjectTcgen05Fence                  (cuda)
60. MergeIfStmt                         (tl)
61. [AnnotateWarpGroupRegAlloc]         (cuda, 可选)
62. MakePackedAPI                       (tl)
63. Simplify                            (tl)
64. LowerDeviceKernelLaunch             (tl)
65. PersistThreadblock                  (cuda)
```

### Host Codegen 附加 Pass

```
66. FP8StorageLegalize                  (tirx)
67. BF16StorageLegalize                 (tirx)
68. LowerTVMBuiltin                     (tirx)
69. LowerCustomDatatypes                (tirx)
70. [CombineContextCall]                (tirx, 可选)
71. LowerIntrin                         (tl)
```
