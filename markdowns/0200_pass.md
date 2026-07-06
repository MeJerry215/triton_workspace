# Pass 级 IR Dump（方法二）

用 `MLIR_ENABLE_DUMP` 查看 **`gluon_to_ttgir`（以及其它 stage）每个 pass 前后** 的 MLIR 变化。

与 stage 级 dump（`TRITON_KERNEL_DUMP=1`，对比 `*.source` vs `*.ttgir`）不同，本方法粒度更细：每个 pass 运行前都会打印一份 IR snapshot。

---

## 原理

`gluon_to_ttgir` 创建 `PassManager` 后调用 `pm.enable_debug()`：

```python
# triton/third_party/nvidia/backend/compiler.py
def gluon_to_ttgir(self, src, metadata, options, capability):
    mod = src
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    # ...
    pm.run(mod, 'gluon_to_ttgir')
```

`enable_debug()` 在检测到 `MLIR_ENABLE_DUMP` 环境变量时，会为 PassManager 注册 IR 打印回调——**每个 pass 执行前** dump 当前 module（C++ 实现在 `triton/python/src/ir.cc`）。

---

## 环境变量

| 变量 | 作用 |
|------|------|
| `MLIR_ENABLE_DUMP=1` | dump 所有 kernel 的所有 pass |
| `MLIR_ENABLE_DUMP=<kernel_name>` | 只 dump 指定 kernel 函数（推荐，输出量小很多） |
| `MLIR_DUMP_PATH=<file>` | 输出到文件；不设则打到 **stderr** |
| `TRITON_ALWAYS_COMPILE=1` | 强制重新编译，避免命中 cache 跳过 pass pipeline |

---

## 使用步骤

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate triton
export PYTHONPATH=/root/triton_workspace/triton/python

export MLIR_ENABLE_DUMP=1
export MLIR_DUMP_PATH=/root/triton_workspace/dump.log
export TRITON_ALWAYS_COMPILE=1

# 可选：只 dump 某个 kernel
# export MLIR_ENABLE_DUMP=memcpy_kernel

python ops/gluon/00_gluon_sytax.py
```

若没有任何 dump 输出，先清 cache 再重跑：

```bash
rm -rf ~/.triton/cache/*
```

---

## 输出格式

dump 文件按 pass 分段，每段以注释头标识：

```
// -----// IR Dump Before GluonInline (gluon-inline) ('builtin.module' operation) //----- //
module attributes { ... } {
  tt.func public @memcpy_kernel(...) { ... }
}

// -----// IR Dump Before GluonInferCoalescedEncodingsPass (gluon-infer-coalesced-encodings) ...
module attributes { ... } {
  ...
}
```

- **Before**：该 pass **运行前** 的 IR
- 括号内 `(gluon-inline)` 是 MLIR pass 的 command-line 名字
- 类名如 `GluonInferCoalescedEncodingsPass` 对应 `Passes.td` 里的定义

查看某个 pass 前后的差异：在 dump 文件里找到相邻两个 `IR Dump Before` 段，用 `diff` 或编辑器对比即可。

---

## `gluon_to_ttgir` 中的 pass 顺序

NVIDIA CUDA 后端（`compiler.py` → `gluon_to_ttgir`）：

| 顺序 | Python API | MLIR pass 名 |
|------|------------|--------------|
| 1 | `passes.gluon.add_inliner` | `gluon-inline` |
| 2 | `passes.gluon.add_infer_coalesced_encodings` | `gluon-infer-coalesced-encodings` |
| 3 | `passes.gluon.add_resolve_auto_encodings` | `gluon-resolve-auto-encodings` |
| 4 | `nvidia.passes.ttnvgpuir.add_tma_lowering` | TMA lowering |
| 5 | `passes.gluon.add_canonicalizer` | `gluon-canonicalize` |
| 6 | `passes.common.add_sccp` | SCCP |
| 7 | `passes.ttir.add_loop_aware_cse` | loop-aware CSE |
| 8 | `passes.gluon.add_canonicalizer` | `gluon-canonicalize` |
| 9 | `passes.ttgpuir.add_combine_tensor_select_and_if` | combine tensor select and if |

`pm.run(mod, 'gluon_to_ttgir')` 的 pipeline tag 也用于 `TRITON_REPRODUCER_PATH` 生成 reproducer 文件名后缀。

---

## 从 dump 定位 pass 实现

1. 在 dump 头里看到 pass 类名，例如 `GluonInferCoalescedEncodingsPass`
2. 搜 `def GluonInferCoalescedEncodingsPass` → `triton/include/triton/Dialect/Gluon/Transforms/Passes.td`
3. 搜同名 C++ 类 → `triton/lib/Dialect/Gluon/Transforms/InferCoalescedEncodings.cpp`
4. Python 绑定 → `triton/python/src/passes.cc` 里的 `ADD_PASS_WRAPPER_0`

---

## 常见问题

**Q: dump 为空或只有部分 stage？**  
A: 编译结果被 cache 命中。设置 `TRITON_ALWAYS_COMPILE=1` 或 `rm -rf ~/.triton/cache/*`。

**Q: 输出太多？**  
A: 用 `MLIR_ENABLE_DUMP=<kernel_name>` 过滤；务必设置 `MLIR_DUMP_PATH` 写入文件而非 stderr。

**Q: 和 `TRITON_KERNEL_DUMP=1` 的区别？**  
A: `TRITON_KERNEL_DUMP` 只落盘各 **stage** 边界文件（`.source`、`.ttgir`、`.llir` 等）；`MLIR_ENABLE_DUMP` 落盘每个 **pass** 前的 IR，适合分析单个 pass 的行为。

**Q: 其它 stage 也能用吗？**  
A: 可以。凡是在 pipeline 里调用了 `pm.enable_debug()` 的 stage（如 `make_ttgir`、`make_llir`）都会在 `MLIR_ENABLE_DUMP=1` 时 dump pass 级 IR。

---

## 相关源码

| 组件 | 路径 |
|------|------|
| `enable_debug` 实现 | `triton/python/src/ir.cc` |
| `gluon_to_ttgir` pipeline | `triton/third_party/nvidia/backend/compiler.py` |
| Gluon pass 注册 | `triton/python/src/passes.cc` |
| Pass TableGen 定义 | `triton/include/triton/Dialect/Gluon/Transforms/Passes.td` |
| 环境变量说明 | `triton/README.md`、`triton/python/triton/knobs.py` |
