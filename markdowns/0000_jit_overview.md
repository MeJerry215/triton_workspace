# Triton `@triton.jit` 全流程总览

这份文档回答两个问题：

1. `triton/python/triton/compiler/compiler.py` 和 `triton/third_party/nvidia/backend/compiler.py` 的关系是什么  
2. 从 `@triton.jit` 到最终 `cubin` 的完整调用链是什么

---

## 1) 三层职责分工（先看全局）

可以把整个链路拆成三层：

- **运行时入口层**：`triton/python/triton/runtime/jit.py`  
  负责装饰器、参数绑定、specialization、设备级缓存、触发编译与 launch。

- **通用编排层**：`triton/python/triton/compiler/compiler.py`  
  负责统一 compile 框架：缓存命中、IR 初始化、stage 循环、产物落盘、返回 `CompiledKernel`。

- **后端实现层（NVIDIA）**：`triton/third_party/nvidia/backend/compiler.py`  
  负责 CUDA 目标相关细节：dialect/pass 组合、`ttir->ttgir->llir->ptx->cubin` 各阶段实现。

一句话：**`jit.py` 发起，`compiler.py` 编排，backend/compiler.py` 真正定义每个阶段怎么编。**

---

## 2) 两个 compiler 文件的关系

### `triton/python/triton/compiler/compiler.py`（通用 orchestrator）

这个文件不偏某个硬件后端，核心是：

- 定义输入抽象：`ASTSource` / `IRSource`
- 定义统一入口：`compile(src, target, options, ...)`
- 调 `make_backend(target)` 选中唯一后端
- 调后端注册 `stages`（`backend.add_stages(...)`）
- 先做 `src.make_ir(...)` 得到初始 module（AST 时会走 `ast_to_ttir`）
- 再循环执行每个 stage：`for ext, compile_ir in ...`

它更像“流水线调度器”，不是每个 stage 的具体算法实现者。

### `triton/third_party/nvidia/backend/compiler.py`（CUDA backend 实现）

这个文件是 `CUDABackend`，核心是：

- `parse_options`：校验/补齐 CUDA 编译选项
- `load_dialects`：加载 NVIDIA 相关 dialect
- `get_codegen_implementation` / `get_module_map`：给前端 lowering 提供 backend 专属函数映射
- `add_stages`：定义阶段顺序和阶段函数
- `make_ttir/make_ttgir/make_llir/make_ptx/make_cubin`：每个阶段的真正 pass/codegen 逻辑

所以两者关系是：

- 通用 `compiler.py` 调用接口；
- NVIDIA `compiler.py` 提供接口实现；
- 二者通过 `BaseBackend` 约定解耦。

---

## 3) `@triton.jit` 触发后的完整时序

下面按“第一次运行（缓存未命中）”展开：

### Step A: 装饰阶段（定义时）

在 `jit.py` 中：

- `@triton.jit` 返回 `JITFunction`
- `JITFunction.__init__` 记录参数信息、constexpr 参数、debug/noinline 等
- 初始化 `device_caches`（按 device 分桶）

此时还没真正编译。

### Step B: 调用阶段（运行时）

用户触发 kernel（例如 `kernel[grid](...)`）后，进入 `JITFunction.run(...)`：

1. 取当前 device/stream  
2. 用 binder 对实参做绑定，得到 `bound_args + specialization + options`
3. 根据 specialization 计算 key，查 `kernel_cache`
4. 未命中则调用 `_do_compile(...)`

### Step C: `JITFunction._do_compile(...)` 触发通用编译入口

`_do_compile` 里会：

- 构造 `ASTSource(self, signature, constexprs, attrs)`
- 调 `self.compile(...)`（即 `triton.compiler.compile`）
- 编译完成后把 kernel 放入 device 级缓存

这一步就是从运行时层进入通用编排层。

### Step D: 通用 `compile(...)` 编排流水线

在 `triton/python/triton/compiler/compiler.py`：

1. `make_backend(target)` 选中后端（CUDA 对应 `CUDABackend`）
2. `backend.parse_options(...)` 处理选项
3. 计算 cache key，检查编译缓存（命中直接返回 `CompiledKernel`）
4. `backend.add_stages(stages, options, src.language)` 注册 stage 顺序
5. `module = src.make_ir(...)`  
   - 对 `ASTSource`：这里调用 `ast_to_ttir`（Python AST -> 初始 TTIR）
6. 执行 stage 循环：
   - `ttir`（`make_ttir`，即对 `ast_to_ttir` 产物做首轮 TTIR 优化/规范化）
   - `ttgir`（`make_ttgir` 或 gluon 路径）
   - `llir`（`make_llir`）
   - `ptx`（`make_ptx`）
   - `cubin`（`make_cubin`）
7. 保存各阶段产物和 metadata，返回 `CompiledKernel`

> 注意：`ast_to_ttir` 不在 `stages` 字典中；它在 `src.make_ir(...)` 这一步先执行。

### Step E: 回到 `JITFunction.run(...)` 并 launch

拿到 `CompiledKernel` 后：

- 规范化 grid
- 组装 launch metadata
- 调 `kernel.run(...)` 真正发射到 GPU stream

---

## 4) 常见“看起来像跳过 ast_to_ttir”的原因

在通用 `compile(...)` 里有：

- `first_stage = list(stages.keys()).index(src.ext)`
- 如果输入是 `IRSource`，还会 `first_stage += 1`

这用于“从某个 IR 文件中间接续编译”的测试/调试场景。  
例如输入是 `.ttgir`，就不会再跑前面的 AST lowering。

但对正常 `@triton.jit`（`ASTSource`, `src.ext="ttir"`）：

- 仍会先执行 `ast_to_ttir`
- 再跑 `ttir/ttgir/llir/ptx/cubin` stages

---

## 5) 一句话记忆

- `jit.py`：**何时编、怎么按实参特化、怎么缓存、怎么 launch**
- `compiler/compiler.py`：**统一编译编排器**
- `third_party/nvidia/backend/compiler.py`：**CUDA 后端每个 stage 的具体实现**

---

## 6) 各 stage 具体做什么（CUDA 路径）

下面按 `add_stages()` 注册顺序解释每个阶段的输入/输出与主要工作。

### 6.1 `ttir` stage（`make_ttir`）

- **输入**：初始 TTIR（AST lowering 刚产出的 module）
- **主要工作**：
  - inliner：展开可内联调用
  - `rewrite_tensor_pointer` / `rewrite_tensor_descriptor_to_pointer`：把部分高层指针/描述符表达改写成更规范形式
  - canonicalizer + CSE + symbol DCE：基础规范化、公共子表达式消除、无用符号清理
  - `combine` / `reorder_broadcast` / `loop_unroll`：做 TTIR 层组合与循环相关优化
- **输出**：更规范、可继续 lowering 的 TTIR

可以理解成“**TTIR 自身清洗与标准化优化阶段**”。
也就是：**`make_ttir` 是 `ast_to_ttir` 的直接后续阶段**（前者偏优化与规范化，后者偏前端 lowering 生成）。

### 6.2 `ttgir` stage（`make_ttgir`）

- **输入**：优化后的 TTIR
- **关键第一步**：`convert_to_ttgpuir`（把 TTIR 转成 GPU 相关的 TTGIR）
- **主要工作**：
  - layout/访存方向：`coalesce`、`remove_layout_conversions`、`prefetch`
  - matmul/dot 方向：`f32_dot_tc`、`accelerate_matmul`、`optimize_dot_operands`
  - 循环与调度：`fuse_nested_loops`、`assign_latencies`、`schedule_loops`、`pipeline`
  - 架构特化：Hopper/Blackwell 路径下的 warp specialization、tmem/tma 相关 pass
  - 结束前再做 canonicalizer/CSE/SCCP/symbol_dce 等收敛
- **输出**：面向目标 GPU 执行模型优化后的 TTGIR

可以理解成“**最重的目标感知优化阶段**”。

### 6.3 `llir` stage（`make_llir`）

- **输入**：TTGIR
- **主要工作（两段）**：
  1. **MLIR 内部转换**  
     - `scf_to_cf`、内存分配（shared/tensor/global scratch）、fence/instrumentation  
     - `to_llvmir`、`nvgpu_to_llvm`、`nvvm_to_llvm`
  2. **LLVM 模块处理**  
     - `llvm.to_module(...)` 得到 LLVM IR（文本）
     - 挂 datalayout/target 特性、外部库链接、`O3` 优化
     - 提取 metadata（shared、tmem、scratch、num_warps 等）
- **输出**：LLVM IR 字符串（仍未变成 PTX）

可以理解成“**从 Triton/MLIR 世界进入 LLVM 世界**”。

### 6.4 `ptx` stage（`make_ptx`）

- **输入**：LLVM IR
- **主要工作**：
  - `llvm.translate_to_asm(...)` 生成 PTX
  - 从 PTX 中提取 kernel entry 名写入 metadata
  - 回填 `.version` / `.target sm_xx`，按配置处理 debug 标志
- **输出**：PTX 文本

可以理解成“**LLVM IR -> NVPTX 汇编文本**”。

### 6.5 `cubin` stage（`make_cubin`）

- **输入**：PTX 文本
- **主要工作**：
  - 调 `ptxas`（携带 arch、优化级别、debug/fmad/额外参数）
  - 捕获并包装 `ptxas` 错误日志
  - 读取 `.o` 二进制作为 cubin 返回
- **输出**：cubin 二进制（最终可加载执行）

可以理解成“**PTX -> 设备可执行二进制**”。

### 6.6 Gluon 的差异（补充）

如果语言是 `Language.GLUON`，`add_stages` 不走 `ttir`，而是：

- `ttgir` 用 `gluon_to_ttgir`
- 然后仍然走 `llir -> ptx -> cubin`

也就是前端入口不同，但后半段后端生成链路一致。

---

## 7) Triton 3.6 常量参数特化与 constant-1 fold

### 7.1 结论速览

| 问题 | Triton 3.6 现状 |
|------|----------------|
| `jit.py` 里有没有 `compute_spec_key` / `AttrsDescriptor.is_equal_to_1`？ | **没有**，已在 #5512 重构中移除 |
| constant-1 fold 还存在吗？ | **存在**，但逻辑下沉到 C++ `native_specialize_impl` |
| `tt.equal_to` MLIR 属性还存在吗？ | **不存在**，改为统一的 `constexpr` 路径 |
| Python 调用签名一致吗？ | **一致**——用户始终传相同参数列表 |
| MLIR / CUDA kernel 签名一致吗？ | **不一致**——`constexpr` 参数从 IR 函数签名中移除 |

### 7.2 旧机制（3.6 之前）vs 新机制（3.6）

#### 旧机制（两套并行逻辑）

```python
# jit.py
def compute_spec_key(v, align):
    if align and hasattr(v, "data_ptr") and (v.data_ptr() % 16 == 0):
        return "D"
    elif isinstance(v, int):
        if align and (v % 16 == 0):
            return "D"
        elif v == 1:
            return "1"   # ← cache key 中的 "1" 标记
    return "N"

# backends/compiler.py — AttrsDescriptor
@staticmethod
def is_equal_to_1(x):
    return isinstance(x, int) and not isinstance(x, bool) and x == 1
```

旧流程对 `int == 1` 做两件事：

1. `compute_spec_key` 返回 `"1"` → 影响 kernel cache key
2. `AttrsDescriptor` 设置 `tt.equal_to` 属性 → 后端 pass 可用此 hint 做优化

#### 新机制（统一 constexpr 路径）

逻辑从 Python 下沉到 `python/src/specialize.cc`：

```cpp
// handle_long_type
if (specialize_value && (val == 1)) {
    return {constexpr_str, arg};  // type="constexpr", value=1
}
```

`AttrsDescriptor` 被移除，对齐特化改为 `BaseBackend.parse_attr("D")` → `[["tt.divisibility", 16]]`。

### 7.3 参数何时被 fold、何时被移除

#### 会被 fold 为 `constexpr`（编译期常量，不进入 MLIR 函数参数）

| 触发条件 | 示例 | 所在代码 |
|----------|------|----------|
| 显式 `tl.constexpr` 注解 | `BLOCK: tl.constexpr` | `create_function_from_signature` |
| 运行时值为 `None` | `mask=None` | `specialize.cc` `Py_IsNone` |
| 传入 `tl.constexpr` 对象 | `tl.constexpr(128)` | `handle_constexpr_type` |
| 嵌套 `@jit` 函数 | 子 kernel 引用 | `handle_jit_callable` |
| **整数隐式特化 `== 1`** | `stride=1` | `handle_long_type` |

#### 不会被 fold，但可能有对齐属性

| 类型 | signature 类型 | 对齐属性 |
|------|----------------|----------|
| 普通 int（≠1） | `i32` / `i64` / `u64` | `tt.divisibility=16`（若 `% 16 == 0`） |
| tensor / pointer | `*fp32` 等 | `tt.divisibility=16`（若 `data_ptr % 16 == 0`） |
| bool | `u1` | 无 |
| float | `fp32` | 无 |

#### fold 后在各层的"移除"行为

```
用户调用 kernel[grid](a, stride=1, N=1024)
         │
         ▼
binder: bound_args = {a, stride, N}          ← 三个参数都在
specialization = [("*fp32", ""), ("constexpr", 1), ("i32", "D")]
         │
         ▼
_pack_args: constexprs = {(1,): 1}          ← stride=1 被提取
           signature = {a: "*fp32", stride: "constexpr", N: "i32"}
         │
         ▼
ASTFunction.serialize:                       ← stride 不在 IR 参数列表
  val_paths = [a, N]  (stride 被 constants 排除)
         │
         ▼
tt.func @kernel(%arg0: !tt.ptr<fp32>, %arg1: i32)  ← 只有两个 MLIR 参数
         │
         ▼
launcher: 仍接收 3 个 Python 参数            ← stride 作为 PyObject 传入但不转发给 CUDA
```

关键代码：

```python
# jit.py _pack_args
constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}

# code_generator.py ASTFunction.serialize
is_val = lambda path, _: path not in self.constants and _ is not None
val_paths = list(find_paths_if(self.arg_types, is_val))  # 排除 constants

# driver.py make_launcher
elif ty != "constexpr":
    internal_args_list.append(f"_arg{i}")  # constexpr 不进入 CUDA kernel params
```

### 7.4 函数签名一致性

| 层级 | stride=1 时 | stride=2 时 | 是否一致 |
|------|-------------|-------------|----------|
| Python 用户调用 | `kernel(a, 1, N)` | `kernel(a, 2, N)` | ✓ 签名相同 |
| `bound_args` 键集合 | `{a, stride, N}` | `{a, stride, N}` | ✓ |
| `ASTSource.signature` | `{a:"*fp32", stride:"constexpr", N:"i32"}` | `{a:"*fp32", stride:"i32", N:"i32"}` | ✗ 类型不同 |
| MLIR `tt.func` 参数数 | 2（stride 被 fold） | 3（stride 是运行时 i32） | ✗ |
| CUDA kernel 实参 | 2 | 3 | ✗ |
| cache key | 不同 specialization | 不同 specialization | ✗ 分别缓存 |

**要点**：Python 层签名对用户始终一致；但编译产物（IR / cubin）按 specialization 分桶，同一 kernel 函数 `stride=1` 和 `stride=2` 会编译出不同版本。

### 7.5 如何禁用 constant-1 fold

旧 patch（针对 `AttrsDescriptor` / `compute_spec_key`）在 3.6 **无效**。可选方案：

```python
# 方案 1：装饰器参数（推荐）
@triton.jit(do_not_specialize=["stride"])
def kernel(a, stride, N): ...

# 方案 2：按位置
@triton.jit(do_not_specialize=[1])
def kernel(a, stride, N): ...
```

`do_not_specialize` 令 `specialize_value=False`，`handle_long_type` 中 `val == 1` 分支不触发，参数保持为 `i32` 运行时参数。

```python
# 方案 3：显式 constexpr（主动 fold，而非隐式）
def kernel(a, stride: tl.constexpr, N): ...
```

### 7.6 与 ixtriturbo patch 的对应关系

```python
# ixtriturbo 旧 patch（3.6 之前有效）
attrs_cls.is_equal_to_1 = staticmethod(lambda x: False)
triton_jit.compute_spec_key = compute_spec_key_without_equal_to_1
```

| 旧 patch 行为 | 3.6 等价目标 | 3.6 实现位置 |
|---------------|-------------|-------------|
| 禁用 `is_equal_to_1` | 不再有 `tt.equal_to` 属性 | 已默认移除 |
| 禁用 `compute_spec_key` 的 `"1"` 分支 | 阻止 `val==1` → `constexpr` | `specialize.cc:238-240` |
| 保留对齐特化 `"D"` | `tt.divisibility=16` | `BaseBackend.parse_attr` |

3.6 等价 patch 应针对 `native_specialize_impl`（或直接用 `do_not_specialize`），而非 `AttrsDescriptor`。
