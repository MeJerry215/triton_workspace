# Triton 中 `make_ttir`：TTIR 上的优化与规范化 Pass 管线

这份笔记说明 **NVIDIA CUDA 后端**在拿到已由 `ast_to_ttir` 等路径生成的 **TTIR module** 之后，如何通过 `make_ttir` 跑一轮 **TTIR 级**的 pass，再交给后续 `make_ttgir` 等阶段。

<a id="pass-manager-overview"></a>
## Pass Manager（`ir.pass_manager`）：干什么、能做什么、和 LLVM 的关系

Python 里的 `ir.pass_manager` 是对 **MLIR `PassManager`** 的薄绑定，定义在 `triton/python/src/ir.cc`：用给定的 `MLIRContext` 构造一个 pass 管线容器，向其中 **注册** 各个 transformation / analysis pass，最后对 **整个 `ModuleOp`** 调用 `run`，在 **原地**改写 IR（失败则抛异常）。它处理的是 **MLIR 层的 Triton TTIR（及混在其间的 common dialect）**，不是 LLVM IR。

```1849:1954:triton/python/src/ir.cc
py::class_<PassManager>(m, "pass_manager", py::module_local())
      .def(py::init<MLIRContext *>())
      .def("enable_debug",
           [](PassManager &self) -> bool {
             auto *context = self.getContext();
             bool haveDump = ::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
             std::string funcToDump;
             if (!haveDump) {
               funcToDump = triton::tools::getStrEnv("MLIR_ENABLE_DUMP");
               bool isEnvValueBool =
                   triton::tools::isEnvValueBool(funcToDump).has_value();
               if (!funcToDump.empty() && !isEnvValueBool)
                 haveDump = true;
             }
             if (haveDump) {
               context->disableMultithreading();
               auto printingFlags = getOpPrintingFlags();
               auto printAlways = [funcToDump](Pass *, Operation *op) -> bool {
                 if (funcToDump.empty())
                   return true;
                 if (auto mod = dyn_cast<mlir::ModuleOp>(op)) {
                   return mod.lookupSymbol(funcToDump);
                 }
                 if (auto func = dyn_cast<triton::FuncOp>(op)) {
                   return SymbolTable::getSymbolName(func).getValue() ==
                          funcToDump;
                 }

                 return false;
               };
               self.enableIRPrinting(
                   /*shouldPrintBeforePass=*/printAlways,
                   /*shouldPrintAfterPass=*/printAlways,
                   /*printModuleScope=*/true,
                   /*printAfterOnlyOnChange=*/false,
                   /*printAfterOnlyOnFailure*/ true, mlir_dumps_or_dbgs(),
                   printingFlags);
             }
             return haveDump;
           })
      .def("get_pipeline_str",
           [](PassManager &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.printAsTextualPipeline(os);
             return str;
           })
      .def(
          "run",
          [](PassManager &self, ModuleOp &mod, std::string repro_pipeline_tag) {
            // TODO: maybe dump module to file and print error for better
            // diagnostics

            auto *context = mod.getContext();
            if (::triton::tools::getBoolEnv("MLIR_DISABLE_MULTITHREADING"))
              context->disableMultithreading();

            auto reproducerPath =
                triton::tools::getStrEnv("TRITON_REPRODUCER_PATH");
            if (!reproducerPath.empty()) {
              if (reproducerPath != "-") {
                std::string repro_suffix =
                    "." + repro_pipeline_tag + ".repro.mlir";
                reproducerPath += repro_suffix;
              }
              auto anchorName = self.getOpAnchorName();
              auto passes = self.getPasses();
              Operation *op = mod.getOperation();
              // Save a reproducer for the current pass manager invocation
              // immediately.
              makeReproducer(anchorName, passes, op, reproducerPath);
              // But if the pass manager crashes, attempt to generate a local
              // reproducer instead.
              context->disableMultithreading();
              self.enableCrashReproducerGeneration(reproducerPath,
                                                   /*genLocalReproducer=*/true);
            } else {
              self.enableCrashReproducerGeneration(makeConsoleReproducer());
            }

            if (triton::tools::getBoolEnv("TRITON_ENABLE_LLVM_DEBUG")) {
              ::llvm::DebugFlag = true;
            }

            if (auto debugOnly =
                    triton::tools::getStrEnv("TRITON_LLVM_DEBUG_ONLY");
                !debugOnly.empty()) {
              llvm::SmallVector<std::string, 3> storage;
              llvm::SmallVector<const char *, 3> debugTypes =
                  parseCommaSeparatedValues(debugOnly, storage);
              ::llvm::DebugFlag = true;
              using namespace llvm;
              setCurrentDebugTypes(debugTypes.data(), debugTypes.size());
            }

            bool haveTiming = ::triton::tools::getBoolEnv("MLIR_ENABLE_TIMING");
            if (haveTiming) {
              self.enableTiming();
            }

            TritonSourceMgrDiagnosticHandler diagHandler =
                setupTritonDiagnosticHandler(context);
            if (failed(self.run(mod.getOperation())))
              throw std::runtime_error("PassManager::run failed");
          },
          py::call_guard<py::gil_scoped_release>());
```

### 能力概览（与源码一一对应）

| 接口 | 作用 |
|------|------|
| `PassManager(context)` | 绑定 `MLIRContext`，后续注册的 pass 在该上下文里解析/改写 IR。 |
| `enable_debug()` | 若 `MLIR_ENABLE_DUMP` 为真，或非空的非布尔字符串（视为 **要 dump 的函数名**），则关掉 context 多线程、打开 **每个 pass 前后的 IR 打印**（可按符号名过滤）；输出走 `mlir_dumps_or_dbgs()`。返回是否启用了 dump。 |
| `get_pipeline_str()` | 把当前已注册的 pass 序列打成 **可读的文本 pipeline**（调试用）。 |
| `run(mod, tag)` | 在 **释放 GIL** 的情况下执行 `self.run(mod.getOperation())`：可选 `MLIR_DISABLE_MULTITHREADING`；`TRITON_REPRODUCER_PATH` 写 **可复现的 mlir reproducer** 并启用 crash reproducer；`MLIR_ENABLE_TIMING` 打开 pass 计时；安装 Triton 诊断处理器；任一 pass 失败则 `PassManager::run failed`。 |

### 和 LLVM 的关系（避免和两个 “PassManager” 混淆）

1. **同一条工具链家族**：MLIR 与 LLVM 共享同一套底层设施（例如 `llvm::raw_string_ostream`、`SmallVector`、部分调试开关），Triton 的 C++ 侧大量 `#include "mlir/..."` 与 `"llvm/..."` 混用是常态。
2. **这一层是 MLIR PassManager**：这里调度的是 **对 MLIR Operation 图** 的 rewrite（canonicalize、CSE、Triton 专用 pass 等）。TTIR 仍是结构化的高层 IR，尚未进入 LLVM IR。
3. **LLVM IR 是更后一阶段**：内核最终会 lower 到 LLVM IR，再由 **LLVM 自己的优化/CodeGen** 处理；那是另一套 pipeline，不是这个 Python `pass_manager` 对象。
4. **`run()` 里对 `llvm::DebugFlag` / `setCurrentDebugTypes` 的触碰**：当设置 `TRITON_ENABLE_LLVM_DEBUG` 或 `TRITON_LLVM_DEBUG_ONLY` 时，在 **跑 MLIR pass 的同一时刻** 打开 LLVM 的全局调试输出过滤。典型用途是：MLIR  lowering 或后续链接到 LLVM 的代码路径里若带有 `LLVM_DEBUG` 日志，可以与 MLIR dump 同步排查；并不是说当前 `PassManager` 在跑 LLVM IR pass。

简记：**`ir.pass_manager` = MLIR 侧“编译中端”的 pass 调度器**；LLVM 是“后端与运行时”的亲戚和同仓组件，通过环境变量和共享类型在边界上挂钩。

---

入口：`triton/third_party/nvidia/backend/compiler.py` 中 `CUDABackend.make_ttir`。

```229:244:triton/third_party/nvidia/backend/compiler.py
    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        if capability // 10 < 9:
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod
```

---

## 目录

- [Pass Manager（`ir.pass_manager`）](#pass-manager-overview)
1. [入口、`mod` 与 `capability`](#1-入口mod-与-capability)
2. [创建 Pass Manager 与 `enable_debug`](#2-创建-pass-manager-与-enable_debug)
3. [Common：`add_inliner`](#3-commonadd_inliner)
4. [TTIR：`add_rewrite_tensor_pointer`](#4-ttiradd_rewrite_tensor_pointer)
5. [TTIR：`add_rewrite_tensor_descriptor_to_pointer`（按算力条件）](#5-ttiradd_rewrite_tensor_descriptor_to_pointer按算力条件)
6. [Common：`add_canonicalizer`](#6-commonadd_canonicalizer)
7. [TTIR：`add_combine`](#7-ttiradd_combine)
8. [TTIR：`add_reorder_broadcast`](#8-ttiradd_reorder_broadcast)
9. [Common：`add_cse`](#9-commonadd_cse)
10. [Common：`add_symbol_dce`](#10-commonadd_symbol_dce)
11. [TTIR：`add_loop_unroll`](#11-ttiradd_loop_unroll)
12. [`pm.run` 与返回值](#12-pmrun-与返回值)

---

## 1) 入口、`mod` 与 `capability`

- `make_ttir(mod, metadata, opt, capability)` 各参数含义与调用时机（谁在何时调用 `make_ttir`）。
- 此时 `mod` 应处于何种 dialect / 形态（TTIR）。
- `capability` 与 `capability // 10 < 9` 分支的语义（与 SM 代际对应关系）。

*（待补充）*

---

## 2) 创建 Pass Manager 与 `enable_debug`

- `ir.pass_manager(mod.context)` 与 `context` 的关系。
- `pm.enable_debug()`：行为见文档开头 [Pass Manager](#pass-manager-overview) 一节（`MLIR_ENABLE_DUMP`、按函数名过滤、与 `run()` 里其它环境变量的分工可在此交叉整理）。

*（待补充）*

---

## 3) Common：`add_inliner`

- `passes.common.add_inliner(pm)` 在 TTIR 上内联什么、典型触发场景与边界。

*（待补充）*

---

## 4) TTIR：`add_rewrite_tensor_pointer`

- `passes.ttir.add_rewrite_tensor_pointer(pm)` 将何种表示 rewrite 成什么，与后续 pass 的依赖关系。

*（待补充）*

---

## 5) TTIR：`add_rewrite_tensor_descriptor_to_pointer`（按算力条件）

- 为何仅在 `capability // 10 < 9` 时注册该 pass。
- `tensor descriptor` → `pointer` 的语义与硬件/路径差异。

*（待补充）*

---

## 6) Common：`add_canonicalizer`

- `passes.common.add_canonicalizer(pm)` 在 TTIR 上做的规范化模式（可与 MLIR canonicalize 概念对齐）。

*（待补充）*

---

## 7) TTIR：`add_combine`

- `passes.ttir.add_combine(pm)` 合并 / 化简了哪些 op 模式。

*（待补充）*

---

## 8) TTIR：`add_reorder_broadcast`

- `passes.ttir.add_reorder_broadcast(pm)` 对 broadcast 与计算顺序的影响、目的（例如为后续 CSE / 融合创造条件）。

*（待补充）*

---

## 9) Common：`add_cse`

- `passes.common.add_cse(pm)` 在 TTIR 上的公共子表达式消除范围与限制。

*（待补充）*

---

## 10) Common：`add_symbol_dce`

- `passes.common.add_symbol_dce(pm)` 删除未引用符号 / dead 符号相关行为。

*（待补充）*

---

## 11) TTIR：`add_loop_unroll`

- `passes.ttir.add_loop_unroll(pm)` 的展开策略、与 `constexpr` / trip count 的关系及风险（代码膨胀等）。

*（待补充）*

---

## 12) `pm.run` 与返回值

- `pm.run(mod, 'make_ttir')`：pipeline 名称、失败时的表现、`mod` 是否原地修改。
- `return mod`：与后续 `make_ttgir` 的衔接。

*（待补充）*
