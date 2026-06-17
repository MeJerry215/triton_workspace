# Triton 中 AST 到 TTIR 的完整流程

这份笔记说明 `@triton.jit` 内核从 **Python 源码** 到 **Python AST**，再到 **TTIR module** 的完整生成链路。

## 1) 从 Python 函数拿到源码

入口对象是 `JITFunction`（文件：`triton/python/triton/runtime/jit.py`）。

- `JITFunction` 内部保存了函数源码字符串 `self._src`
- 源码通常来自被 `@triton.jit` 装饰的 Python 函数
- 后续编译都围绕这份源码和调用时的 specialization 信息进行

## 2) 源码解析为 Python AST

`JITFunction.parse()` 做了 AST 解析：

1. `tree = ast.parse(self._src)`
2. 断言 `tree` 是 `ast.Module`
3. 断言 `tree.body` 只有一个 `ast.FunctionDef`
4. 返回 `tree`

这里得到的是 **Python 标准 AST**，还不是 Triton IR。

## 3) 进入 `ast_to_ttir`（前处理阶段）

入口函数：`triton/python/triton/compiler/code_generator.py` 中的 `ast_to_ttir(...)`。

它先做 lowering 前的类型和特化准备：

1. 按 `src.signature` 构造参数类型列表 `arg_types`
2. 用 `src.constants` 把 constexpr 参数/子结构替换成 constexpr type
3. 组装 `prototype = ASTFunction([], arg_types, src.constants, src.attrs)`
4. 计算内核表示名（用于 specialization 区分）
5. 创建 `CodeGenerator(...)`

## 4) 创建 `CodeGenerator`（准备 IR 构建环境）

`CodeGenerator.__init__` 的关键初始化：

- 创建语义层和 builder  
  - 普通 Triton 路径：`ir.builder(context)` + `TritonSemantic`
  - Gluon 路径：`gluon_ir.GluonOpBuilder(context)` + `GluonSemantic`
- 初始化模块：`self.module = self.builder.create_module()`（若未传入现成 module）
- 保存作用域（全局捕获、局部符号表、函数原型、编译选项等）

重点：**module 在这里就已经存在了，但还是空壳**。

## 5) `generator.visit(fn.parse())` 如何把 AST 变成 TTIR

这一句是核心驱动：

```python
generator.visit(fn.parse())
```

本质是 `ast.NodeVisitor` 分发机制：

1. 根节点是 `ast.Module`，进入 `visit_Module`
2. `visit_Module` 继续遍历子节点（那个 `FunctionDef`）
3. 进入 `visit_FunctionDef`，开始真正发射 IR

`visit_FunctionDef` 内部关键动作：

1. `get_or_insert_function(...)` 创建/获取 TTIR function
2. `self.module.push_back(self.fn)` 把函数挂到 module
3. 处理参数绑定（Python 形参 -> IR value）
4. `visit_compound_statement(node.body)` 逐条遍历函数体语句
5. 每个 `visit_*`（如赋值、if、for、算术、load/store、call）调用 `semantic + builder` 发射对应 TTIR op
6. 处理 return、函数签名修正、`self.fn.finalize()`

所以 `visit` 的意义不是“返回 module”，而是**在遍历过程中持续向 `self.module` 写入 IR**（副作用）。

## 6) `ast_to_ttir` 收尾

`visit` 完成后：

1. 取出 `module = generator.module`
2. 绑定 `module.context = context`
3. 执行 `module.verify()` 做 IR 合法性检查
4. 通过后返回 module；失败则抛错

至此，`ast_to_ttir` 输出的就是后续编译 pipeline 使用的 TTIR module。
在 CUDA 路径下，这个 module 紧接着会进入 `make_ttir` stage，做第一轮 TTIR 规范化和优化。

## 7) 一句话总结

`ast_to_ttir` 不是“简单遍历 AST”，而是一个完整 lowering 入口：

- 先做 specialization/type 前处理
- 再以 Python AST visitor 为调度骨架
- 在各个 `visit_*` 里通过 `semantic + builder` 发射 TTIR
- 最后校验并返回 module

---

## 8) 可对照阅读的关键文件

- `triton/python/triton/runtime/jit.py`  
  `JITFunction.parse()`：源码 -> Python AST
- `triton/python/triton/compiler/code_generator.py`  
  `ast_to_ttir(...)`：编译入口  
  `CodeGenerator.__init__`：builder/module 初始化  
  `CodeGenerator.visit(...)`：统一 visit 包装与错误定位  
  `visit_Module / visit_FunctionDef / visit_compound_statement / 各类 visit_*`：语句表达式 lowering

## 9) `ir.cc` 中“如何真正构造 TTIR”

你提到的 `triton/python/src/ir.cc`（约 802-1847）本质是：

- 用 pybind11 把 C++ 的 `TritonOpBuilder` 暴露成 Python 的 `ir.builder`
- Python 层调用的 `builder.create_*` 最终都落到这里
- 这里再调用 MLIR/Triton dialect 的 C++ op 构造逻辑

可以把它看成 **Python 前端到 C++ IR 构造器的桥接层**。

### 9.1 builder 生命周期与插入点

这部分决定“IR 往哪里插入”：

- `create_module()`：创建 `ModuleOp`
- `set_insertion_point_to_start/end/after(...)`：设置当前插入位置
- `get_insertion_point()` / `restore_insertion_point(...)`：保存/恢复插入点
- `create_block()` / `create_block_with_parent(...)`：创建基本块

所以 `code_generator.py` 里看到的：

- `then_block = self.builder.create_block()`
- `self.builder.set_insertion_point_to_start(then_block)`

就是在显式控制后续 op 写入哪个 block。

### 9.2 函数与控制流构造

函数和控制流相关 API：

- `get_or_insert_function(...)`：创建或复用函数符号
- `ret(...)` / `call(...)`
- `create_if_op(...)`, `create_for_op(...)`, `create_while_op(...)`
- `create_yield_op(...)`, `create_branch(...)`, `create_cond_branch(...)`

这对应 `visit_FunctionDef`、`visit_If`、`visit_For` 等 AST visitor 的 lowering 动作。

### 9.3 算子构造（算术 / 内存 / 内建）

`ir.cc` 暴露了大量 `create_*`，它们直接创建 Triton/MLIR op，例如：

- 算术：`create_add/create_mul/create_fdiv/...`
- 比较：`create_icmp*`, `create_fcmp*`
- 内存：`create_load/create_store/create_masked_load/create_masked_store`
- 张量/布局：`create_reshape/create_broadcast/create_trans/create_splat`
- 原子：`create_atomic_cas/create_atomic_rmw`
- 内建：`create_get_program_id/create_get_num_programs/create_dot`

这些 API 通常由 `semantic.py` 间接调用，而不是 AST visitor 直接拼 C++ op。

### 9.4 一条最小调用链（从 Python 语义到 TTIR op）

以 `tl.program_id(0)` 为例，调用链通常是：

1. `CodeGenerator.visit_*` 识别到该语义调用
2. 调 `TritonSemantic.program_id(axis)`
3. 语义层调用 `self.builder.create_get_program_id(axis)`
4. `ir.cc` 中的 pybind 包装转到 C++ `self.create<GetProgramIdOp>(axis)`
5. op 按当前 insertion point 插入到当前 block，成为 TTIR 的一部分

所以“如何构造 TTIR”的最终答案是：

- AST 负责遍历与调度
- semantic 负责语义规则与类型层动作
- `ir.cc` 暴露的 `builder.create_*` 负责真正把 op 插入 module/block

## 10) Triton IR 这几个 `.td` 文件的关系

你列的这 6 个文件是 Triton Dialect 的 TableGen 元定义，关系可以按“底层定义 -> 约束接口 -> 具体 op”理解。

### 10.1 依赖关系（从 include 看）

最核心依赖链是：

1. `TritonDialect.td`：定义 dialect 本体（名字 `tt`、依赖方言等）
2. `TritonTypes.td`：定义 Triton 自定义类型（`ptr`、`tensordesc` 等）
3. `TritonAttrDefs.td`：定义 enum attributes（cache、evict、rounding、program dim 等）
4. `TritonInterfaces.td`：定义通用 trait（编码一致、shape 约束等 NativeOpTrait）
5. `TritonOpInterfaces.td`：定义特定 op interface（如 `DotOpInterface`、`TransposeOpInterface`）
6. `TritonOps.td`：定义具体 `tt.*` ops（load/store/dot/reduce/func/call/...）

其中 `TritonOps.td` 会 include 前面几乎所有文件，因此它是“汇总并落地具体操作定义”的中心文件。

### 10.2 每个文件各自负责什么

- `TritonDialect.td`  
  声明方言级信息：方言名、命名空间、依赖 dialect、类型注册入口等。

- `TritonTypes.td`  
  声明 type 系统：哪些类型算 `TT_PtrLike`、`TT_Tensor`、`TT_Type`，以及自定义类型构造规则。

- `TritonAttrDefs.td`  
  声明属性枚举：比如 `CacheModifier`、`EvictionPolicy`、`InputPrecision`，并决定文本 IR 里打印字符串。

- `TritonInterfaces.td`  
  声明“通用约束 trait”：主要给 op 复用，比如 operands/result 的 shape 或 encoding 一致性约束。

- `TritonOpInterfaces.td`  
  声明“语义接口”：给某类 op 提供统一访问方法/验证逻辑（例如 dot/trans/descriptor 系列）。

- `TritonOps.td`  
  真正定义每个 `tt.*` op 的 operands/results/attrs/assemblyFormat/verifier/builder/interface 绑定。

### 10.3 在编译链中的位置

这些 `.td` 不直接“执行编译”，而是作为规范源：

- 经过 TableGen 生成 C++ 声明/实现（op/type/attr/interface 的样板和注册代码）
- Python/C++ `builder.create_*` 最终创建的 `tt.*` op，语义上就受这些 `.td` 定义约束
- 也就是说：`semantic.py`/`ir.cc` 在“使用”这些定义，`.td` 在“定义规则”
