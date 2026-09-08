# `tilelang.jit` 深入分析：从 Python 到 GPU Kernel 的完整流程

> 本文以 `layernorm.py` 中 `@tilelang.jit(out_idx=[-3, -2, -1])` 为入口，逐层深入分析 `tilelang.jit` 的完整机制。

---

## 1. `@tilelang.jit` 概览

### 1.1 用法示例（来自 `layernorm.py`）

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-3, -2, -1])
def _layernorm_fwd(N, D, eps=1e-5, blk_m=1, threads=256, in_dtype="bfloat16", out_dtype="bfloat16"):
    accum_dtype = "float"

    @T.prim_func
    def main(
        X: T.Tensor((N, D), in_dtype),
        gamma: T.Tensor((D,), in_dtype),
        beta: T.Tensor((D,), in_dtype),
        Y: T.Tensor((N, D), out_dtype),
        Mean: T.Tensor((N,), accum_dtype),
        Rstd: T.Tensor((N,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, blk_m), threads=threads) as bx:
            # ... kernel body ...
        return main
```

### 1.2 两种执行模式

| 模式 | 判断方式 | 行为 | 示例 |
|------|---------|------|------|
| **lazy** | 函数体内定义了 `@T.prim_func` 并显式 `return` 一个 `PrimFunc` | 调用装饰器返回 `JITKernel` 对象，可手动调用 | `kernel = _layernorm_fwd(N, D); result = kernel(x, g, b)` |
| **eager** | 函数体使用 DSL builder 模式，不显式 return PrimFunc，通过 tensor 类型标注构建 | 调用装饰器即编译并立即执行 kernel | `gemm(A, B, C)` 直接执行 |

`layernorm.py` 使用的是 **lazy 模式**：外层函数 `_layernorm_fwd` 内部定义了 `@T.prim_func` 修饰的 `main`，并 `return main`。

---

## 2. 入口：`jit()` 函数

**文件：** `tilelang/tilelang/jit/__init__.py`, line 564-628

### 2.1 签名

```python
def jit(
    func: Callable | PrimFunc | None = None,
    *,
    out_idx: list[int] | int | None = None,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    execution_backend: ExecutionBackend | None = None,
    verbose: bool | None = None,
    pass_configs: dict | None = None,
    debug_root_path: str | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITImpl:
```

### 2.2 内部逻辑

`jit()` 是一个支持两种调用形式的装饰器：

```python
# 形式一：带参数 @tilelang.jit(out_idx=[-3, -2, -1])
# 此时 func is None，返回 decorator 函数

# 形式二：无参数 @tilelang.jit
# 此时 func 是被装饰的函数，直接应用 decorator
return decorator(func) if func is not None else decorator
```

核心逻辑在 `decorator` 内部：

```python
def decorator(func):
    mode = "auto"
    # 1. 通过 prim_func() 将原始函数包装为 JITFunc
    pf = prim_func(func, eager_jit=True)
    # 2. 获取源码和签名（用于后面的 IR 生成和缓存）
    func_source = inspect.getsource(pf.orig_func)
    signature = inspect.signature(pf.orig_func)
    # 3. 返回 JITImpl 实例
    return JITImpl(func=pf, ..., mode=mode)
```

**关键：** `prim_func(func, eager_jit=True)` 并不立即生成 TIR，而是返回一个 `JITFunc` 包装对象，真正的编译延迟到 `JITImpl.__call__()` 时。

---

## 3. `prim_func()` 与 `JITFunc`

**文件：** `tilelang/tilelang/language/eager/builder.py`, line 1505-1545

### 3.1 `prim_func()` 函数

```python
def prim_func(func, *, eager_jit=False):
    sig = inspect.signature(func)
    ir_gen = mutate(func)          # 关键：AST 变换，生成 IRGenerator
    func_annot = get_type_hints(func)
    annot = {param.name: type_hint for param in sig.parameters.values()}

    if eager_jit:
        # 返回 JITFunc（惰性包装，不立即编译）
        return JITFunc(func, sig, arg_names, tensor_args, ...)
    else:
        # 立即构建 TIR PrimFunc
        builder = Builder()
        with builder.prim_func(func.__name__):
            ir_gen.gen(builder)(**annot)
        return builder.get()
```

### 3.2 `mutate()` — AST 变换引擎

**文件：** `tilelang/tilelang/language/eager/ast.py`, line 650

`mutate(func)` 对 Python 函数进行静态 AST 分析：

1. 解析函数的 Python 源码为 AST
2. 遍历 AST 节点，将每个 Python 操作映射到 `Builder` 的方法调用
3. 返回 `IRGenerator`，其 `.gen(builder)` 方法在给定 `Builder` 上执行 IR 构建

例如：
- `T.copy(...)` → `builder.eval(...)`
- `for i, j in T.Parallel(...)` → `builder.ctx_for(...)`
- `X_local[i, j] = expr` → `builder.bind(...)` / `builder.assign_slice(...)`

### 3.3 `JITFunc` — 两阶段编译器

`JITFunc` 实现了 **两阶段（two-phase）编译**：

```
Phase 1 (编译时):
  - 用户传入 N, D, blk_m, threads 等编译时常量
  - 运行 Python 函数体一次以构建 TIR 模板
  - TIR 模板中 constexpr 变量被保留为 Var（占位符）
  - 结果缓存在 p1_cache 中

Phase 2 (运行时):
  - 用户传入 X, gamma, beta 等运行时应存
  - 从 buffer 的实际 shape/strides 提取 constexpr 变量的值
  - 代入 TIR 模板生成最终 PrimFunc
```

```python
class JITFunc:
    def parse_args(self, *args, **kwargs):
        # 返回 (key, tensor_args)
        # key = (p1_key, p2_key) 用于缓存
        if not self.tensor_args:
            p1_key = self._argument_binder.bind_no_tensor_key(args, kwargs)
            return (p1_key, None), {}

        bound = self._argument_binder.bind(args, kwargs)
        tir_temp = self.p1_cache.get(bound.p1_key, None)
        if tir_temp is None:
            tir_temp = self._build_tir_template(**bound.compile_kwargs)
            self.p1_cache[bound.p1_key] = tir_temp
        p2_key = tir_temp._parse_phase2_key(**bound.tensor_args, **bound.compile_kwargs)
        return (bound.p1_key, p2_key), bound.tensor_args
```

---

## 4. `JITImpl` — JIT 编译的门面

**文件：** `tilelang/tilelang/jit/__init__.py`, line 257-541

`JITImpl` 是一个 `@dataclass`，也是用户与 `tilelang.jit` 交互的主要接口。

### 4.1 核心字段

```python
@dataclass
class JITImpl:
    out_idx: list[int] | int | None         # 输出 tensor 的索引
    execution_backend: str | None           # 执行后端
    target: TargetLike | None               # 编译目标
    target_host: TargetLike | None          # 主机编译目标
    verbose: bool | None
    pass_configs: dict | None               # TVM Pass 配置
    debug_root_path: str | None             # 调试输出路径
    compile_flags: list[str] | str | None   # 额外编译标志
    func_source: str                        # 原始函数源码
    signature: inspect.Signature            # 原始函数签名
    mode: Literal["auto", "lazy", "eager"]  # 执行模式
    func: JITFunc                           # 被包装的函数对象
```

### 4.2 `__call__()` — 入口方法

```python
def __call__(self, *args, **kwargs):
    # 1. 处理调优参数
    has_tune_params = "__tune_params" in kwargs
    kwargs.update(kwargs.pop("__tune_params", {}))

    # 2. 推断模式（lazy / eager）
    if self.mode == "auto":
        self.mode = self._infer_jit_mode(*args, **kwargs)
        self.func.set_mode(self.mode)

    # 3. 解析参数，获取缓存 key
    key, kernel_args = self.func.parse_args(*args, **kwargs)

    # 4. 检查缓存，编译若未命中
    kernel = self._kernel_cache.get(key, None)
    if kernel is None:
        kernel = self.compile(*args, **kwargs)
        self._kernel_cache[key] = kernel

    # 5. lazy 模式返回 kernel 对象；eager 模式立即执行
    if self.mode == "eager":
        return kernel(*kernel_args.values())
    else:
        return kernel
```

### 4.3 `compile()` — 触发真正编译

**文件：** `tilelang/tilelang/jit/__init__.py`, line 442-473

```python
def compile(self, *args: _P.args, **kwargs: _P.kwargs) -> _Ret:
    # 1. 获取最终 PrimFunc（lazy: 直接返回已构建好的；eager: 代入 shape 后重新生成）
    prim_func = self.get_tir(*args, **kwargs)

    # 2. 调用模块级 compile() → cached() → JITKernel 编译流程
    kernel_result = compile(
        prim_func,
        out_idx=self.out_idx,
        execution_backend=self.execution_backend,
        target=self.target,
        target_host=self.target_host,
        verbose=self.verbose,
        pass_configs=self.pass_configs,
        compile_flags=self.compile_flags,
    )

    # 3. 调试模式：保存 CUDA 源码 + TIR 脚本到文件
    if self.debug_root_path:
        func_name = getattr(self.func, "__name__", "jit_kernel")
        is_cutedsl = (self.execution_backend or self.target) == "cutedsl"
        kernel_suffix = "py" if is_cutedsl else "c"
        kernel_file = f"tilelang_jit_kernel_{func_name}.{kernel_suffix}"
        program_file = f"tilelang_jit_program_{func_name}.py"
        makedirs(self.debug_root_path, exist_ok=True)
        with open(path.join(self.debug_root_path, kernel_file), "w") as f:
            print(kernel_result.get_kernel_source(), file=f)
        with open(path.join(self.debug_root_path, program_file), "w") as f:
            print(prim_func.script(), file=f)

    return kernel_result
```

#### 4.3.1 `compile()` 的调试手段：环境变量 vs. `pass_configs`

想要观察这段 `compile()` 过程中每个 pass 前后的 IR 变换以及 AST dump，有两种途径：

| 控制方式 | 是否需要改代码 | 手段 | 何时设置 | 输出内容 |
|---------|:----------:|------|---------|---------|
| **环境变量** `TILELANG_PASS_DIFF` | **否** — 零侵入 | 运行前设置 | `TILELANG_PASS_DIFF=terminal python my_script.py` | 终端彩色 diff |
| | | | `TILELANG_PASS_DIFF=html python my_script.py` | HTML 报告 |
| | | | `TILELANG_PASS_DIFF=both python my_script.py` | 两者兼有 |
| | | 输出目录 | `TILELANG_PASS_DIFF_OUTPUT=./mydir` (默认 `tmp/pass_diff_output/`) | HTML 文件路径 |
| | | `TILELANG_VERBOSE=1` | 可替代 `verbose=True` | 编译各阶段日志 |
| **`pass_configs` 参数** | **是** — 需改 `@jit` 装饰器 | 在装饰器中添加 | `PassConfigKey.TL_ENABLE_DUMP_IR: True` | 每个 pass 前后 IR 写入文件 `./dump_ir/` |
| | | | `PassConfigKey.TL_AST_PRINT_ENABLE: True` | stdout 打印 TIR AST |

> **推荐使用 `TILELANG_PASS_DIFF`**：它是全自动的 hook，无需修改代码，在每个 pass 执行后自动捕获 IR 的 before/after 并输出 unified diff（终端彩色或 HTML 报告），让你直观看到每一行 IR 的增加和删除。详见下文 §4.5.1。

### 4.4 AST 的获取与变换 — `mutate()` 引擎

**文件：** `tilelang/tilelang/language/eager/ast.py`, line 650-699

#### 4.4.1 获取 AST：`get_ast()`

**文件：** `tilelang/tilelang/language/eager/utils.py`, line 59-66

```python
def get_ast(func: Callable):
    _, start = inspect.getsourcelines(func)         # 获取函数的起始行号
    filename = inspect.getsourcefile(func)           # 获取源文件路径
    source = inspect.getsource(func)                 # 获取函数源码（字符串）
    source = _remove_leading_ident(source)           # 去除公共缩进
    source = "\n" * (start - 1) + source             # 填充空行使行号对齐
    tree = ast.parse(source, filename=filename)       # Python AST parser
    return tree
```

通过标准库 `inspect.getsource()` + `ast.parse()` 将 Python 函数解析为 AST 树。

#### 4.4.2 AST 变换：`mutate()`

`mutate()` 在 `get_ast()` 得到 AST 后，执行以下流程：

```python
def mutate(func: Callable[_P, _T]) -> IRGenerator[_P, _T]:
    tree = utils.get_ast(func)                         # 1. 获取 AST
    filename = inspect.getsourcefile(func)
    nonlocals = utils.get_func_nonlocals(func)         # 2. 捕获闭包变量

    mut = DSLMutator(nonlocals, func.__globals__,       # 3. DSL 变换器
                     Path(filename).name)
    tree = mut.visit(tree)                              #    遍历 AST，每个 Python 节点
                                                        #    被替换为 Builder 方法调用

    make_closure = utils.get_compiled_object(           # 4. 编译为可执行对象
        tree, "make_closure", filename)
    ...
```

`DSLMutator`（`ast.py` line 119-611）是一个 `ast.NodeTransformer`，它将每个 Python 语法结构翻译为 `Builder` 的方法调用：

| Python 源码 | DSLMutator 转换结果 | 最终调用 Builder |
|---|---|---|
| `T.copy(X, X_smem)` | `__tb.eval(T.copy(...))` | `builder.eval(...)` |
| `for i in T.serial(N):` | `for _tmp in __tb.ctx_for(...):` | `builder.ctx_for(...)` |
| `sum_row[i] = ...` | `__tb.bind('sum_row', ...)` | `builder.bind(...)` |
| `return main` | `__tb.ret(main)` | `builder.ret(...)` |
| `if cond:` | `for _br in __tb.ctx_if(cond):` | `builder.ctx_if(...)` |
| `a + b` | `__tb.eval(__tb.op(...))` | 在 builder 上下文中 eval |

> **关键设计**：AST 变换后的代码**不是直接执行** Python 原始函数，而是将每个语句翻译为对 `Builder` 对象的调用。`Builder` 内部调用 TVM 的 `IRBuilder` 逐步构建 TIR IR。

#### 4.4.3 从变换到可执行代码：`get_compiled_object()`

**文件：** `tilelang/tilelang/language/eager/utils.py`, line 72-86

```python
def get_compiled_object(source, name, filename=None, globals=None):
    # 编译 AST 为 Python 字节码
    compiled = compile(source, filename, "exec")
    # 在新命名空间中执行，提取名为 `name` 的对象
    locs = {}
    exec(compiled, globals, locs)
    return locs[name]
```

AST → Python 字节码编译 → `exec()` 执行 → 取出变换后的闭包函数。

该闭包函数被包装为 `IRGenerator`：

```python
@dataclass
class IRGenerator(Generic[_P, _T]):
    gen: Callable[[BaseBuilder], Callable[_P, _T]]   # 接收 Builder 返回可调用函数
    source: str                                        # 变换后 AST 的源码文本
    extra_type_hints: dict                             # 额外类型标注
```

当 `IRGenerator.gen(builder)(**args)` 被调用时：
1. `gen(builder)` 将 `builder` 注入为 `__tb` 上下文
2. 返回一个可调用函数
3. 调用它将 `**args` 传入 DSL 变换后的代码体
4. 代码体中每个 `__tb.xxx()` 调用最终在 `Builder` 上构建 TIR

### 4.5 Pass 前后 IR 对比与调试

TileLang 提供了多种调试手段来观察编译过程中的 IR 变化。其中有两种方式最直接：

1. **环境变量 `TILELANG_PASS_DIFF`**（推荐）— 零侵入，无需修改代码，自动拦截每个 pass 并输出 before/after 的 unified diff
2. **`pass_configs` 参数** — 通过 `@tilelang.jit(pass_configs={...})` 控制，需要修改装饰器参数

> **注意**：这两个机制的工作层级不同。`TILELANG_PASS_DIFF` 钩子挂在 `tvm.ir.transform.Pass.__call__` 上（在 `PassContext` 之外），因此可以捕获**所有** pass 的 IR 变换，包括 `PassContext` 内部编排的 pass chain。而 `TL_ENABLE_DUMP_IR` 是通过 `PassContext` 的 `instruments` 注册，只在 `PassContext` 生命周期内生效。

#### 4.5.1 方法一（推荐）：`TILELANG_PASS_DIFF` 环境变量 — 全自动 pass diff

**文件：** `tilelang/tilelang/utils/pass_diff_hook.py`

这是最方便的调试方式。在运行脚本前设置环境变量 `TILELANG_PASS_DIFF`，TileLang 启动时会自动安装一个 hook 到 `tvm.ir.transform.Pass.__call__`，在每个 pass 执行前和执行后捕获 IR，计算 unified diff 并输出。

**三种模式：**

```bash
# 终端彩色 diff（默认 3 行 context）
TILELANG_PASS_DIFF=terminal python my_script.py

# 生成 HTML 报告（交互式，可折叠展开、复制源码、切换主题）
TILELANG_PASS_DIFF=html python my_script.py

# 两者同时
TILELANG_PASS_DIFF=both python my_script.py
```

**控制输出目录：**

```bash
TILELANG_PASS_DIFF=html TILELANG_PASS_DIFF_OUTPUT=./my_diff_output python my_script.py
```

默认 HTML 输出路径为 `tmp/pass_diff_output/pass_diff_<timestamp>.html`。

**效果示例（HTML 报告）：**

- 每个 pass 是一个可折叠的面板
- 发生变更的 pass 自动展开，显示带行号的 side-by-side diff
- 未变更的 pass 折叠显示完整 IR
- 顶部工具栏可切换 dark/light 主题、全部展开/折叠
- 支持 `📋 Copy` 按钮一键复制 before/after IR

**工作原理：**

```python
# pass_diff_hook.py 的核心逻辑：
def _patched_call(self, mod):
    before_script = mod.script()           # 捕获 before IR
    result = _original_call(self, mod)     # 执行原始 pass
    after_script = result.script()         # 捕获 after IR
    diff = compute_diff(before, after)     # 计算 unified diff
    # 输出：terminal 彩色 / HTML 渲染
    return result
```

> 这个 hook **默认关闭（零开销）**，只有设置了 `TILELANG_PASS_DIFF` 才激活。设置在 `tilelang/__init__.py` line 218-222：

```python
from .utils.pass_diff_hook import install_pass_diff_hook as _install_pass_diff_hook
_install_pass_diff_hook()
```

#### 4.5.2 方法二：`TL_ENABLE_DUMP_IR` — 通过 `PassContext` 的 `instruments` dump 每个 pass 前后的 IR

**文件：** `tilelang/tilelang/transform/pass_config.py`, line 280-284
**启用位置：** `tilelang/tilelang/jit/kernel.py`, line 236-240

```python
# 在 JITKernel._compile_and_create_adapter() 中：
pass_instruments = []
if pass_configs.get(PassConfigKey.TL_ENABLE_DUMP_IR):    # "tl.enable_dump_ir"
    dump_ir_path = pass_configs.get(
        PassConfigKey.TL_DUMP_IR_DIR, "./dump_ir")       # "tl.dump_ir_path"
    pass_instruments.append(tvm.ir.instrument.DumpIR(dump_dir=dump_ir_path))

with tvm.transform.PassContext(..., instruments=pass_instruments):
    artifact = tilelang.lower(...)
```

使用方法：

```python
import tilelang
from tilelang.transform import PassConfigKey

@tilelang.jit(
    out_idx=[-3, -2, -1],
    pass_configs={
        PassConfigKey.TL_ENABLE_DUMP_IR: True,                       # 启用 IR dump
        PassConfigKey.TL_DUMP_IR_DIR: "./my_dump_ir",                # 输出目录（可选）
    }
)
def _layernorm_fwd(N, D, ...):
    ...
```

效果：在每个 Pass 执行前后，`tvm.ir.instrument.DumpIR` 会将当前 IRModule 的文本形式写入 `<dump_ir_path>/<pass_name>_before.txt` 和 `<dump_ir_path>/<pass_name>_after.txt`。

#### 4.5.3 方法三：`debug_root_path` — 保存最终结果

在 `JITImpl.compile()`（`tilelang/jit/__init__.py` line 455-471）中，若设置了 `debug_root_path`，编译完成后会将最终 CUDA 源码和 TIR 脚本保存到磁盘：

```python
@tilelang.jit(out_idx=[-3, -2, -1], debug_root_path="./debug_output")
def _layernorm_fwd(N, D, ...):
    ...
```

输出文件：
- `tilelang_jit_kernel_<func_name>.cu` — 生成的 CUDA kernel 源码
- `tilelang_jit_program_<func_name>.py` — TIR 脚本形式的 PrimFunc

#### 4.5.4 方法四：`TL_AST_PRINT_ENABLE` — 打印 TIR AST

```python
pass_configs={
    PassConfigKey.TL_AST_PRINT_ENABLE: True,           # "tl.ast_print_enable"
}
```

启用后，TileLang 会在 `PreLowerSemanticCheck` 阶段调用 `tilelang.analysis.ASTPrinter()(mod)`，将 TIR 的 AST 文本表示打印到 stdout，适合在小规模 kernel 上快速查看 IR 结构。

**相关代码**（`tilelang/tilelang/engine/semantic_check.py` line 15-28）：

```python
def should_enable_ast_print(pass_ctx=None):
    return bool(pass_ctx and pass_ctx.config.get(PassConfigKey.TL_AST_PRINT_ENABLE, False))

def PreLowerSemanticCheck(mod):
    if should_enable_ast_print():
        tilelang.analysis.ASTPrinter()(mod)    # 打印完整 TIR AST
    tilelang.analysis.NestedLoopChecker()(mod)
    tilelang.analysis.FragmentLoopChecker()(mod)
```

#### 4.5.5 方法五：`TL_LAYOUT_VISUALIZATION_ENABLE` — layout 可视化

```python
pass_configs={
    PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,   # "tl.layout_visualization_enable"
    PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "svg", # "pdf", "png", "svg", "all"
}
```

在 `LayoutInference` pass 之后，将 fragment/shared memory 的 layout 以图形方式输出，帮助理解数据排布和向量化效果。

#### 4.5.6 方法六：`verbose=True` — 编译过程日志

```python
@tilelang.jit(out_idx=[-3, -2, -1], verbose=True)
def _layernorm_fwd(N, D, ...):
    ...
```

也可以通过环境变量设置：`TILELANG_VERBOSE=1`。

启用后，`tilelang.jit` 会在编译的各个阶段通过 `jit_phase` 上下文管理器输出日志，包括 lower 阶段、adapter 创建等。

#### 4.5.7 方法七：获取中间 IR 做自定义分析

在代码中手动调用 `tilelang.engine.lower.lower_to_host_device_ir()` 获取 pass 处理后的 IRModule，再自行检查：

```python
from tilelang.engine.lower import lower_to_host_device_ir

host_mod, device_mod, params, target, target_host = \
    lower_to_host_device_ir(prim_func, target="cuda")
print(host_mod.script())    # 打印 host 侧 TIR
print(device_mod.script())  # 打印 device 侧 TIR
```

#### 调试方法总结

| 方法 | 配置方式 | 输出内容 | 适用场景 |
|------|---------|---------|---------|
| `TILELANG_PASS_DIFF` | **环境变量**（推荐） | 终端彩色 diff / HTML 报告 | 零侵入的全局 pass 级别 IR 变换分析，最能直观看到每行 IR 的变化 |
| `TL_ENABLE_DUMP_IR` | `pass_configs` | 每个 pass 前后 IR 文本（写文件） | 需要留存文件记录、定位哪个 pass 出了问题 |
| `debug_root_path` | `@jit` 参数 | 最终 CUDA 源码 + TIR 脚本 | 检查最终生成结果是否正确 |
| `TL_AST_PRINT_ENABLE` | `pass_configs` | TIR AST 文本（stdout） | 快速查看 IR 结构 / 调试 AST 级别的问题 |
| `TL_LAYOUT_VISUALIZATION_ENABLE` | `pass_configs` | Layout 图 (svg/png/pdf) | 调试 memory layout / 向量化 |
| `verbose=True` / `TILELANG_VERBOSE=1` | `@jit` 参数 / 环境变量 | 日志输出 | 追踪编译流程 / 耗时 |
| `lower_to_host_device_ir()` | Python API | IRModule | 需要自定义 IR 分析 |

---

## 5. `compile()` → `cached()` → 缓存系统

### 5.1 模块级 `compile()` 函数

**文件：** `tilelang/tilelang/jit/__init__.py`, line 91-170

```python
def compile(func, out_idx, execution_backend, target, ...):
    # 从 PrimFunc 属性中提取函数级配置
    func_attrs = func.attrs
    if "tilelang_out_idx" in func_attrs: out_idx = ...
    if "tilelang_pass_configs" in func_attrs: pass_configs = ...
    if "tilelang_compile_flags" in func_attrs: compile_flags = ...

    return cached(func, out_idx, ..., target=target, ...)
```

### 5.2 `cached()` — 带磁盘/内存缓存的编译

**文件：** `tilelang/tilelang/cache/__init__.py`

```python
def cached(func, out_idx, ..., target, execution_backend, ...):
    # 1. 解析后端
    cache, norm_target, execution_backend, verbose = _resolve_cache_dispatch(
        target, execution_backend, verbose
    )
    # 2. 委托给具体的 KernelCache 实现
    return cache.cached(func, out_idx, ..., target=norm_target, ...)
```

### 5.3 `KernelCache` 缓存策略

**文件：** `tilelang/tilelang/cache/kernel_cache.py`

缓存 key 由以下要素的 SHA256 哈希决定：

| 要素 | 说明 |
|------|------|
| `func` | TIR script 的 SHA256 哈希 |
| `out_idx` | 输出索引 |
| `args_repr` | 参数的 repr |
| `target` | 编译目标 |
| `execution_backend` | 执行后端 |
| `pass_configs` | Pass 配置 |
| `version` | tilelang 版本 |
| `tilelang_lib` | C++ 库文件的 SHA256（可选） |

缓存分层：
1. **内存缓存** (`_memory_cache`): `dict[str, JITKernel]`
2. **磁盘缓存** (`CACHE_DIR/<version>/<platform>/kernels/<key>/`):
   - `device_kernel.cu`: GPU kernel 源码
   - `host_kernel.cu`: Host wrapper 源码
   - `kernel_lib.so`: 编译后的共享库
   - `params.pkl`: Kernel 参数序列化

---

## 6. `JITKernel` — 编译后的内核包装

**文件：** `tilelang/tilelang/jit/kernel.py`, line 39-846

### 6.1 构造函数

```python
class JITKernel:
    def __init__(self, func, out_idx, execution_backend, target, ...):
        self.prim_func = func
        self.target = determine_target(target, return_object=True)
        self.execution_backend_spec = resolve_execution_backend_spec(execution_backend, self.target)

        # 编译并创建 adapter
        adapter = self._compile_and_create_adapter(func, out_idx)
        self.adapter = adapter
        self.torch_function = adapter.func   # 可被 torch 调用的函数
```

### 6.2 `_compile_and_create_adapter()` — 真正的编译核心

```python
def _compile_and_create_adapter(self, tilelang_func, out_idx):
    with (
        jit_phase("lower", verbose=verbose, **phase_context),
        tvm.transform.PassContext(opt_level=3, config=pass_configs, ...),
        self.target,  # 设置 target scope
    ):
        artifact = tilelang.lower(
            tilelang_func,
            target=target,
            target_host=target_host,
            enable_host_codegen=enable_host_codegen,
            enable_device_compile=enable_device_compile,
        )

    self.artifact = artifact

    # 根据 backend 创建对应的 adapter
    if execution_backend == "tvm_ffi":
        adapter = TVMFFIKernelAdapter(...)
    elif execution_backend == "cython":
        adapter = CythonKernelAdapter(...)
    elif execution_backend == "nvrtc":
        adapter = NVRTCKernelAdapter(...)
    elif execution_backend == "torch":
        adapter = MetalKernelAdapter(...)
    elif execution_backend == "cutedsl":
        adapter = CuTeDSLKernelAdapter(...)
```

执行后端对比：

| 后端 | 适用平台 | 特点 |
|------|---------|------|
| `tvm_ffi` | CUDA | 默认后端，通过 TVM FFI 调用，支持 DLPack 与 PyTorch 互通 |
| `cython` | CUDA | 通过 Cython 生成 C++ wrapper，编译为 .so |
| `nvrtc` | CUDA | 使用 NVRTC 运行时编译 CUDA 源码 |
| `torch` | Metal (Apple) | 通过 PyTorch Metal 后端 |
| `cutedsl` | CUDA (CuTe) | 使用 CuTe DSL 后端 |

---

## 7. `tilelang.lower()` — TVM Pass Pipeline

**文件：** `tilelang/tilelang/engine/lower.py`, line 296-341

### 7.1 整体流程

```python
def lower(func_or_mod, target, ...):
    # Step 1: Lower to host/device IR
    host_mod, device_mod, params, target, target_host = lower_to_host_device_ir(...)

    # Step 2: Device codegen
    codegen_mod = device_codegen(device_mod, target)
    kernel_source = codegen_mod.inspect_source()

    # Step 3: Compile into CompiledArtifact
    return CompiledArtifact(host_mod, device_mod, params, kernel_source, ...)
```

### 7.2 `lower_to_host_device_ir()` — CUDA Pass Pipeline

**文件：** `tilelang/tilelang/engine/lower.py`, line 258-293

```python
def lower_to_host_device_ir(func_or_mod, target, ...):
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    # 运行 target-specific pass pipeline
    pipeline = resolve_pipeline(target)    # CUDA → CUDAPassPipelineBody
    mod = pipeline.lower(mod, target)

    # 分离 host 和 device functions
    host_mod = Filter(is_host_call)(mod)
    device_mod = Filter(is_device_call)(mod)

    return host_mod, device_mod, params, target, target_host
```

### 7.3 CUDA Pass Pipeline 详解

**文件：** `tilelang/tilelang/cuda/pipeline.py`, line 68-254

CUDA pipeline 分为 **Prologue** 和 **Body** 两部分，包含约 **40 个 pass**：

#### Prologue（前置 passes）

| Pass | 作用 |
|------|------|
| `BindTarget` | 绑定 target 信息 |
| `MaterializeKernelLaunch` | 将 `T.Kernel` 展开为实际的 thread/block 绑定 |
| `LetInline` | 内联 let 绑定（可选） |
| `AddWrapperForSingleBufStore` | 为单 buffer store 添加 wrapper |
| `LegalizeNegativeIndex` | 规范化负索引 |
| `InjectAssumes` | 注入 assume 信息以加速 TVM prover |
| `Simplify` | IR 简化 |
| `LayoutReducer` | 为 reduce 操作设置 layout |
| **`ProducerConsumerWarpSpecialized`** | CUDA 专用：warp specialized 变换（sm_90+ TMA） |
| **`LowerBlackwell2SM`** | CUDA 专用：Blackwell 2SM 变换 |
| `IfStmtBinding` | Normalize if-without-else |
| **`PipelinePlanning`** | 流水线规划（分析访存模式） |
| **`InjectSoftwarePipeline`** | 注入软件流水线（多缓冲、异步拷贝） |
| **`LayoutInference`** | **核心 Pass**：推断 fragment/shared memory 的 memory layout |
| **`LowerTileOp`** | **核心 Pass**：将高层 tile 操作降级为低层操作（如 `T.copy` → `cp.async`） |
| `LowerL2Persistent` | CUDA 专用：L2 persistent |
| `DecoupleTypeCast` | 解耦类型转换与向量化 |
| `LegalizeVectorizedLoop` | 验证向量化循环合法性 |
| `LegalizeSafeMemoryAccess` | 添加安全边界检查 |
| `LowerAccessPtr` | 将 pointer metadata 降为标准 `tvm_access_ptr` |
| `HoistNonRestrictParams` | Hoist non-restrict 参数标注 |

#### Body（后续 passes）

| Pass | 作用 |
|------|------|
| `LowerSharedTmem` | CUDA 专用：shared.tmem 初始化 |
| `PlanAndUpdateBufferAllocationLocation` | 规划 buffer 分配位置 |
| `LowerSharedBarrier` | CUDA 专用：mbarrier（sm_90+）支持 |
| `FuseMBarrierArriveExpectTx` | CUDA 专用：mbarrier arrive 融合 |
| `HoistGlobalBufferAllocations` | 提升全局 buffer 分配 |
| `LowerOpaqueBlock` | 将 opaque block 降级 |
| `NarrowDataType` | 将数据类型缩窄到 32 位 |
| `FlattenBuffer` | 展平多维 buffer |
| `ConfigIndexBitwidth` | 配置索引位宽 |
| `VectorizeLoop` | 循环向量化 |
| `StorageRewrite` | 存储重写 |
| `LoopUnswitching` | 循环 unswitch |
| `UnrollLoop` | 循环展开 |
| `LowerThreadAllreduce` | 线程级 allreduce |
| `LowerLDGSTG` | CUDA 专用：LDG/STG 优化 |
| `LowerHopperIntrin` | CUDA 专用：Hopper intrinsic（sm_90+） |
| `AnnotateDeviceRegions` | 标注 device region |
| `SplitHostDevice` | 分离 host/device 代码 |
| `MergeSharedMemoryAllocations` | 合并共享内存分配 |
| `InjectFenceProxy` | CUDA 专用：TMA fence proxy |
| `ThreadSync` | 插入 thread synchronization |
| `MakePackedAPI` | 生成 packed API |
| `Simplify` | 最终简化 |
| `LowerDeviceKernelLaunch` | 降级 device kernel launch |
| `PersistThreadblock` | CUDA 专用：persistent threadblock |

---

## 8. 完整数据流：从 Python 到 CUDA Kernel

以 `layernorm.py` 的调用 `kernel = _layernorm_fwd(N, D, ...)` 为例：

```
用户调用
    │
    ▼
JITImpl.__call__(N, D, eps=1e-5, blk_m=1, ...)
    │  ├─ 推断 mode = "lazy"
    │  └─ _layernorm_fwd 是 lazy style（内部有 @T.prim_func）
    │
    ▼
JITFunc.parse_args(N, D, blk_m=1, threads=256, ...)
    │  ├─ 无 tensor_args → bind_no_tensor_key()
    │  └─ key = (args_tuple, None)
    │      (所有参数都是编译时常量，没有运行时 tensor)
    │
    ▼
JITImpl.compile(N, D, ...)
    │  └─ get_tir() → JITFunc.get_tir()
    │      ├─ p1_key 缓存未命中
    │      ├─ is_lazy_style = True（检测到 inner @T.prim_func）
    │      ├─ 直接调用 _layernorm_fwd(N, D, ...)
    │      │   在 Python 中执行：
    │      │   ├─ 调用 @T.prim_func 定义的 `main`
    │      │   │  └─ Builder 构建 TIR IR
    │      │   │     ├─ with T.Kernel(ceildiv(N/blk_m), threads=256) as bx
    │      │   │     ├─ T.alloc_shared((blk_m, D), in_dtype)
    │      │   │     ├─ T.alloc_fragment(...)
    │      │   │     ├─ T.copy(...)
    │      │   │     ├─ for i,j in T.Parallel(blk_m, D)
    │      │   │     ├─ T.reduce_sum(...)
    │      │   │     └─ return main  → 返回 PrimFunc
    │      │  └─ return PrimFunc
    │      └─ 得到 tvm.tirx.PrimFunc 对象
    │
    ▼
tilelang.jit.compile(prim_func, out_idx=[-3,-2,-1], ...)
    │  └─ cache.cached(prim_func, ...)
    │
    ▼
KernelCache.cached(prim_func, ...)
    │  ├─ 生成缓存 key（TIR script + 参数 + 版本 + 平台）
    │  ├─ 检查内存缓存 → 未命中
    │  ├─ 检查磁盘缓存 → 未命中
    │  └─ 创建 JITKernel(prim_func, ...)
    │
    ▼
JITKernel.__init__(prim_func, out_idx, target="auto", ...)
    │  ├─ target → determine_target("auto") → Target("cuda -arch=sm_89")
    │  ├─ backend → resolve_execution_backend("auto") → "tvm_ffi"
    │  └─ _compile_and_create_adapter(prim_func, out_idx)
    │
    ▼
JITKernel._compile_and_create_adapter()
    │  └─ tilelang.lower(prim_func, target, ...)
    │      └─ lower_to_host_device_ir(prim_func, target)
    │          ├─ mod = IRModule({"main": prim_func})
    │          ├─ resolve_pipeline(target) → CUDA Pipeline
    │          │  ├─ CUDAPassPipelineBodyPrologue (前置 passes)
    │          │  └─ CUDAPassPipelineBody (后续 passes)
    │          ├─ pipeline.lower(mod, target)  → 经过 ~40+ passes
    │          └─ Filter(is_host_call/is_device_call) → host_mod, device_mod
    │      │
    │      └─ device_codegen(device_mod, target)
    │          └─ codegen_mod.inspect_source() → CUDA 源码
    │      └─ return CompiledArtifact(host_mod, device_mod, params, kernel_source, ...)
    │
    ▼
TVMFFIKernelAdapter(artifact.params, result_idx, target, func_or_mod, 
                     host_mod, device_mod, rt_mod, device_kernel_source, ...)
    │  └─ adapter.func → PyTorch-compatible callable
    │      内部绑定 DLPack tensor 转换和 kernel launch
    │
    ▼
JITKernel.torch_function = adapter.func
    │
    ▼
返回 JITKernel 对象给用户

用户调用 kernel(x, gamma, beta)
    ▼
JITKernel.__call__(x, gamma, beta)
    └─ self.torch_function(x, gamma, beta)
       └─ TVMFFI 通过 DLPack 接收 PyTorch tensor
          └─ 设置 kernel 参数
             └─ 启动 CUDA kernel
```

---

## 9. 核心设计亮点

### 9.1 两阶段编译（Phase 1 + Phase 2）

```
Phase 1: 构建 TIR 模板
  - 输入: 编译时常量 (N, D, blk_m, threads, ...)
  - 输出: TIR 模板（含 constexpr var 占位符）
  - 缓存: p1_cache — 以编译时参数为 key

Phase 2: 代入 shape 信息
  - 输入: 运行时 tensor (X, gamma, beta)
  - 从 buffer shape 中提取实际值
  - 代入 TIR 模板得到最终 PrimFunc
  - 缓存: kernel_cache — 以 (p1_key, p2_key) 为 key
```

### 9.2 AST 级别代码变换

`mutate()` 将 Python 函数体静态分析为 AST，将每个 Python 语句映射到 `Builder` 的 method：

| Python 结构 | Builder 方法 | TIR 结构 |
|-------------|-------------|----------|
| `T.copy(src, dst)` | `eval()` | `BufferStore` / `Copy` |
| `for i in T.serial(N)` | `ctx_for()` | `For` (serial) |
| `for i in T.Parallel(N)` | `ctx_for()` | `For` (parallel) |
| `T.alloc_shared(shape, dtype)` | `bind_immutable()` | `SBlock` allocate |
| `T.alloc_fragment(shape, dtype)` | `bind_immutable()` | register buffer |
| `T.reduce_sum(src, dst, dim=1)` | `eval()` | `Reduce` |
| `if cond:` | `ctx_if()` → `with_frame(If)` | `IfThenElse` |

### 9.3 多层缓存加速

```
JITFunc.p1_cache          # TIR template 缓存（Python dict）
JITImpl._kernel_cache     # 已编译 JITKernel 缓存（Python dict）
KernelCache._memory_cache # 进程内内存缓存
KernelCache._disk_cache   # 文件系统持久化缓存（跨进程）
```

### 9.4 执行后端抽象

通过 `BaseKernelAdapter` 抽象统一不同后端的 kernel 执行：

```
BaseKernelAdapter (接口)
  ├── TVMFFIKernelAdapter   (CUDA 默认, DLPack 互通)
  ├── CythonKernelAdapter   (CUDA, C++ wrapper)
  ├── NVRTCKernelAdapter    (CUDA, 运行时编译)
  ├── MetalKernelAdapter    (Apple Metal)
  └── CuTeDSLKernelAdapter  (CuTe DSL)
```

---

## 10. 总结

`tilelang.jit` 是一条从 **Python DSL → AST → TIR IR → TVM Pass Pipeline → CUDA Source → GPU Kernel** 的完整编译链路。

| 层次 | 组件 | 职责 |
|------|------|------|
| **Python DSL** | `tilelang.language` | 用户编写 kernel 的 Pythonic 方式 |
| **AST 变换** | `mutate()` / `IRGenerator` | 将 Python 代码静态分析为 AST，生成 IR 构建指令 |
| **IR 构建** | `Builder` | 在 TVM IRBuilder 上构建 TIR |
| **JIT 调度** | `JITImpl` / `JITFunc` | 两阶段编译、参数解析、缓存管理 |
| **Pass Pipeline** | `tilelang.lower()` + CUDA passes | ~40 个 TVM pass 优化和降级 TIR |
| **代码生成** | `device_codegen()` | TIR → CUDA C++ 源码 |
| **执行** | `BaseKernelAdapter` | 编译、链接、启动 GPU kernel |
