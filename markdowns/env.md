# Triton README 中的环境变量（中文说明）

以下内容摘自 [triton/README.md](../triton/README.md)，按用途分组。

---

## 一、使用自定义 LLVM 构建

| 变量 | 说明 |
|------|------|
| `LLVM_BUILD_DIR` | 指向你本地 LLVM 构建目录的根路径（例如 `$HOME/llvm-project/build`），需按你的实际路径修改。 |
| `LLVM_INCLUDE_DIRS` | LLVM 头文件目录，通常设为 `$LLVM_BUILD_DIR/include`。 |
| `LLVM_LIBRARY_DIR` | LLVM 库目录，通常设为 `$LLVM_BUILD_DIR/lib`。 |
| `LLVM_SYSPATH` | LLVM 构建的系统路径，通常设为 `$LLVM_BUILD_DIR`，与 `LLVM_INCLUDE_DIRS` / `LLVM_LIBRARY_DIR` 一起传给 `pip install -e .`。 |

---

## 二、编译与安装技巧

| 变量 | 说明 |
|------|------|
| `TRITON_BUILD_WITH_CLANG_LLD` | 设为 `true` 时使用 **clang** 与 **lld** 编译；**lld** 往往能加快链接速度。 |
| `TRITON_BUILD_WITH_CCACHE` | 设为 `true` 时使用 **ccache** 加速重复编译。 |
| `TRITON_HOME` | 指定 **`.triton` 目录**所在位置（缓存与构建过程中的下载等）。默认在用户主目录；可随时改路径。 |
| `MAX_JOBS` | 传给 `pip install -e .` 时限制并行任务数，**内存不足**时可减小该值。 |

---

## 三、调试 / 编译器行为（与 `knobs.py` 对应或仅 C++ 层）

| 变量 | 说明 |
|------|------|
| `MLIR_ENABLE_DUMP` | 设为 `1` 时，对每个 kernel 在每次 MLIR pass 前转储 IR；设为 **kernel 名称** 则只转储该 kernel。若无效可清理 `~/.triton/cache/*`。 |
| `MLIR_DUMP_PATH` | 指定 `MLIR_ENABLE_DUMP` 的输出位置；未设置则输出到 **stderr**。 |
| `LLVM_IR_ENABLE_DUMP` | 设为 `1` 时，在每次对 LLVM IR 的 pass 前转储 IR。 |
| `TRITON_REPRODUCER_PATH` | 设为 `<reproducer_path>` 时，在每个 MLIR 编译阶段前生成 MLIR **复现文件**；若某阶段失败，该路径处会保留失败 pass 前的本地复现器。 |
| `TRITON_INTERPRET` | 设为 `1` 时使用 **Triton 解释器** 而非 GPU 执行，可在 kernel 里下 Python 断点。 |
| `TRITON_ENABLE_LLVM_DEBUG` | 设为 `1` 时向 LLVM 传入 `-debug`，会向 **stdout** 打印大量调试信息；太吵可改用 `TRITON_LLVM_DEBUG_ONLY`。 |
| `TRITON_LLVM_DEBUG_ONLY` | 逗号分隔列表，等价于 LLVM 的 `-debug-only`，只输出指定 pass/组件的调试信息（如 `tritongpu-remove-layout-conversions`）。 |
| `TRITON_ENABLE_ASAN` | 设为 `1` 时启用 LLVM **地址消毒（ASAN）**，用于泄漏与越界检测；**目前仅 AMD 后端**支持，需按 ROCm 文档配置 ASAN 库。 |
| `USE_IR_LOC` | 取值为 `ttir` 或 `ttgir` 时，按对应扩展名的 **IR 文件行号** 重新解析位置信息，便于 IR 与 llir/ptx 对应及性能分析。 |
| `TRITON_PRINT_AUTOTUNING` | 设为 `1` 时，autotuning 结束后打印每个 kernel 的 **最佳配置** 与总耗时。 |
| `DISABLE_LLVM_OPT` | 解析为布尔 `true` 时关闭 `make_llir` / `make_ptx` 的 LLVM 优化；否则可解析为要禁用的优化 **标志列表**（如 `disable-lsr`）。 |
| `TRITON_ALWAYS_COMPILE` | 设为 `1` 时 **强制每次编译 kernel**，忽略缓存命中。 |
| `MLIR_ENABLE_TIMING` | 输出每个 **MLIR pass** 的耗时信息。 |
| `LLVM_ENABLE_TIMING` | 输出每个 **LLVM pass** 的耗时信息。 |
| `TRITON_DEFAULT_FP_FUSION` | 覆盖默认是否允许 **FP 融合**（如 mul+add→fma）。 |
| `MLIR_ENABLE_DIAGNOSTICS` | 逗号分隔：`warnings`、`remarks`、`stacktraces`、`operations` 等，控制 MLIR **诊断输出**；默认多为仅错误。 |
| `MLIR_ENABLE_REMARK` | **已弃用**，请改用 `MLIR_ENABLE_DIAGNOSTICS=remarks`。 |
| `TRITON_KERNEL_DUMP` | 设为 `1` 时转储各编译阶段的 IR 以及最终 **ptx / amdgcn**。 |
| `TRITON_DUMP_DIR` | 在 `TRITON_KERNEL_DUMP=1` 时，指定转储 IR 与 ptx/amdgcn 的 **保存目录**。 |
| `TRITON_KERNEL_OVERRIDE` | 设为 `1` 时允许在每个编译阶段开始用用户提供的 **IR/ptx/amdgcn** 覆盖已编译 kernel。 |
| `TRITON_OVERRIDE_DIR` | 在 `TRITON_KERNEL_OVERRIDE=1` 时，指定 **加载** 覆盖用 IR/ptx/amdgcn 的目录。 |
| `TRITON_F32_DEFAULT` | 设置 `tl.dot` 在 32 位浮点下的默认输入精度：`ieee`、`tf32` 或 `tf32x3`。 |
| `TRITON_FRONT_END_DEBUGGING` | 设为 `1` 时，前端出错 **不包装异常**，便于看到完整 Python 栈。 |
| `TRITON_DISABLE_LINE_INFO` | 设为 `1` 时从模块中 **移除所有行号信息**。 |
| `PTXAS_OPTIONS` | 向 NVIDIA 的 **ptxas** 传递额外命令行参数。 |
| `LLVM_EXTRACT_DI_LOCAL_VARIABLES` | 生成完整调试信息，便于在 **cuda-gdb、rocm-gdb** 等 GPU 调试器中求值。 |
| `TRITON_DEFAULT_BACKEND` | 可选，设置 Triton 构造活动驱动（`triton.runtime.driver.active`）时的 **默认后端**名称。 |

说明：README 注明部分变量在 **`knobs.py` 中没有对应 knob**，仅与 C++ 层相关。

---

## 四、Kernel 覆盖流程中常用组合（README 示例）

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=<dump_dir>
export TRITON_KERNEL_OVERRIDE=1
export TRITON_OVERRIDE_DIR=<override_dir>
```

1. 运行一次 kernel，在 `TRITON_DUMP_DIR` 中转储各阶段 IR 与 ptx/amdgcn。  
2. 将 `TRITON_DUMP_DIR/<kernel_hash>` 复制到 `TRITON_OVERRIDE_DIR`。  
3. 删除不想覆盖的阶段，修改需要覆盖的阶段。  
4. 再次运行 kernel 查看覆盖后的结果。

---

## 五、更多配置

完整 **Python 层 knob** 列表见仓库内 `python/triton/knobs.py`；许多 knob 也可通过环境变量控制，README 只列举了其中一部分。
