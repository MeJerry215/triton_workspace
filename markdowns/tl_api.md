# Triton 语言 API 说明（`language` 与 `experimental`）

下文分别说明稳定入口 `triton.language` 与实验入口 `triton.experimental`（当前以 **Gluon** 为主）的职责、子模块与典型用法。路径基于本仓库：`triton/python/triton/language`、`triton/python/triton/experimental`。

---

## 一、`triton.language`：稳定内核编写 API

**定位**：在 `@triton.jit` 内核里使用的**官方稳定** Python 表面语法；由 `triton.language.__init__` 统一再导出，并维护 `__all__` 供自动补全与文档索引。

**典型引用**：

```python
import triton.language as tl
```

### 1. 子包与子模块

| 符号 | 说明 |
|------|------|
| **`tl.core`**（在包根大量 re-export） | 标量/块类型、`tensor`、`constexpr`、`program_id`、`num_programs`、`load`/`store`、`dot`/`dot_scaled`、原子操作、`reduce`、`where`、`make_block_ptr`、`tensor_descriptor` 等内核核心语义。 |
| **`tl.standard`** | 常用组合算子：`sum`、`max`/`min`、`softmax`、`sort`、`topk`、`cumsum`、`zeros`/`zeros_like`、`swizzle2d` 等。 |
| **`tl.math`** | 逐元素数学：`exp`、`log`、`sqrt`、`sin`/`cos`、`fma`、`div_rn` 等（与 `core` 中部分符号共同出现在包根）。 |
| **`tl.random`** | 设备端随机数：`philox`、`rand`/`randn` 及 4x 变体等。 |
| **`tl.extra`** | **按后端动态加载**的子包（见 `language/extra/__init__.py`）：构建时随 CUDA/HIP 等注册，例如 `triton.language.extra.cuda` 下的设备函数；**不是**单一固定文件列表。 |
| **`tl.extra.libdevice`** | 对 libdevice 风格数学/工具函数的封装（单独模块）。 |
| **`tl.target_info`** | 与当前编译目标相关的查询类能力（依实现而定）。 |

### 2. 能力分组（便于查阅，非独立子模块名）

- **类型与常量**：`dtype` 族（`float16`、`int32`、`constexpr`…）、`pointer_type`、`block_type`、`void` 等。
- **网格与并行**：`program_id`、`num_programs`、`range` / `static_range`。
- **内存**：`load`、`store`、`atomic_*`、`make_block_ptr`、`gather`；以及 **Tensor Descriptor**（`make_tensor_descriptor`、`tensor_descriptor` 等）用于结构化全局内存视图。
- **计算与归约**：`dot`、`associative_scan`、`reduce`、`histogram`、`inline_asm_elementwise` 等。
- **调试**：`device_print`、`device_assert`、`debug_barrier`、`static_print`、`static_assert`。
- **杂项**：`assume`、`multiple_of`、`max_contiguous` 等向编译器传递假设/提示。

### 3. 与 Gluon 的衔接

`language` 包内的 `str_to_ty` 在解析部分类型字符串时会 **lazy import** `triton.experimental.gluon.language` 下的 layout / `tensor_descriptor_type`，用于 Gluon 相关类型描述；日常写经典 Triton 内核一般无需关心。

---

## 二、`triton.experimental`：实验 API（Gluon）

**定位**：**不稳定**、可能随版本大改的接口；当前仓库中主体为 **`triton.experimental.gluon`**：另一套从 Python 到 IR 的路径（`Language.GLUON`），直接面向 **TTGPUIR** 层语义，暴露更多 **layout、warp、TMA/MMA** 等底层控制。

**典型引用**：

```python
from triton.experimental import gluon
from triton.experimental.gluon import jit, constexpr_function
# 内核内常用：
from triton.experimental.gluon import language as ttgl
# 或按需：
from triton.experimental.gluon.language import nvidia, amd
```

### 1. `triton.experimental.gluon`（包根）

| 导出（见 `gluon/__init__.py`） | 说明 |
|-------------------------------|------|
| **`jit`** | Gluon 专用 JIT：内部使用 `GluonASTSource`，`language=Language.GLUON`，源码阶段产出为 **ttgir** 扩展语义，与经典 `triton.jit`（TTIR 路径）不同。 |
| **`constexpr_function`** | 与主包类似的编译期常量函数装饰器。 |
| **`must_use_result`** | 从 `triton.language.core` 再导出：强制使用返回值，避免忽略同步/资源类结果。 |
| **`nvidia` / `amd`** | 子命名空间，对应各厂商 **runtime / 目标** 侧入口（如 hopper、blackwell、gfx1250 等模块组织）。 |

`gluon/_runtime.py` 中说明了 Gluon 模块在 `make_ir` 时会预先设置 `ttg.target`、`ttg.num-warps`、`ttg.num-ctas` 等 **module attribute**，再调用 `ast_to_ttir` 生成 IR（与经典流程的起点不同）。

### 2. `triton.experimental.gluon.language`

**定位**：Gluon 内核内使用的“语言”层 API，与 `tl` 部分同名但 **实现与类型系统不同**（`_core`、`_semantic`、`_layouts` 等私有实现）。

| 区域 | 说明 |
|------|------|
| **`_core` 再导出** | 算术、`load`/`store`、原子、`reduce`、`warp_specialize`、`allocate_shared_memory`、`thread_barrier`、`dot_fma`、`set_auto_layout`、`to_tensor` 等；许多为更贴近硬件的显式控制。 |
| **`_layouts`** | `BlockedLayout`、`NVMMASharedLayout`、`SwizzledSharedLayout`、`CoalescedLayout` 等 **显式内存/线程布局** 类型。 |
| **`_math` / `_standard`** | 与经典 `language.math` / `standard` 类似的数学与工具函数集合（命名带下划线模块，部分在 `language` 包根再导出）。 |
| **`nvidia`** | 按架构分子包：**Hopper**（`tma`、`mbarrier`、`warpgroup_mma` 等）、**Blackwell**、**Ampere**（`async_copy`、`mma_v2` 等）。 |
| **`amd`** | **CDNA3/CDNA4、GFX1250、RDNA3/RDNA4** 等子包，含异步拷贝、mbarrier、TDM 等目标相关原语。 |
| **`extra`** | 当前转发 **`libdevice`**（`from triton.language.extra import libdevice`），与经典路径共享设备数学库封装。 |

### 3. 使用注意

- **稳定性**：`experimental` 不保证长期兼容；升级 Triton 版本后应重点回归测试。
- **与 `triton.language` 混用**：需在文档与类型层面分清两套 `jit` 与 `language`；同一内核内避免混用两套语义，除非官方示例明确展示。
- **文档入口**：具体函数说明以源码 docstring 与 `python/examples/gluon` 下示例为准。

---

## 三、`triton.language` API 速查（用处 / 主要参数）

下列与 `triton/language/core.py`、`standard.py`、`math.py`、`random.py` 及包根 `__all__` 对齐。`_semantic` 等为编译器注入参数，**不要在内核里手写**。

### 3.1 标量类型与指针元数据

| API | 用处 | 主要参数 |
|-----|------|----------|
| `void` | 表示无返回值类型。 | （类型常量） |
| `int1`…`int64`、`uint8`…`uint64` | 整数标量类型。 | （类型常量） |
| `float16`、`bfloat16`、`float32`、`float64` | 浮点标量类型。 | （类型常量） |
| `float8e4b15`、`float8e4nv`、`float8e4b8`、`float8e5`、`float8e5b16` | FP8 等窄浮点类型。 | （类型常量） |
| `dtype` | dtype 元类/基类。 | （类型系统用） |
| `pointer_type` | 构造指针类型（元素类型、地址空间、是否 const）。 | `element_ty`、`address_space`、`const` |
| `pi32_t` | 预置的 `int32` 指针类型别名。 | （类型常量） |
| `block_type` | 描述「块」张量的元素类型与块形状。 | 构造参数见源码 |
| `tensor` | 块张量值类型。 | 一般由运算产生，非直接构造 |
| `constexpr` | 包装**编译期已知**的 Python 值。 | `constexpr(value)` |
| `constexpr_type` | `constexpr` 的类型对象。 | （内部） |
| `const` | **注解用**：标记只读指针语义；`store` 不能作用于 const 指针。 | 空类，作类型标记 |
| `tuple` / `tuple_type` | 内核内元组值及其类型。 | 见 `core` |
| `tensor_descriptor` / `tensor_descriptor_type` | TMA 等硬件描述符类型与值。 | 多由 `make_tensor_descriptor` 创建 |
| `PropagateNan` | `maximum`/`minimum`/`clamp` 的 NaN 传播策略枚举。 | 见 `ir.PROPAGATE_NAN` |
| `TRITON_MAX_TENSOR_NUMEL` | 单块张量元素个数上限常量。 | （常量） |

### 3.2 网格与索引

| API | 用处 | 主要参数 |
|-----|------|----------|
| `program_id` | 当前 program 在 launch grid 某轴上的索引。 | `axis`：0/1/2 |
| `num_programs` | launch grid 在指定轴上的 program 数量。 | `axis`：0/1/2 |
| `arange` | 生成半开区间 `[start, end)` 的连续整数块；`end-start` 有上限。 | `start`, `end`（多为 2 的幂约束，见 docstring） |

### 3.3 内存：load / store / 块指针 / 描述符

| API | 用处 | 主要参数 |
|-----|------|----------|
| `load` | 从指针或块指针加载张量/标量。 | `pointer`；可选 `mask`、`other`、`boundary_check`、`padding_option`、`cache_modifier`、`eviction_policy`、`volatile` |
| `store` | 向指针或块指针存储。 | `pointer`, `value`；可选 `mask`、`boundary_check`、`cache_modifier`、`eviction_policy` |
| `make_block_ptr` | 构造父张量中一块区域的**块指针**。 | `base`, `shape`, `strides`, `offsets`, `block_shape`, `order` |
| `advance` | 块指针按各维偏移前进（**必须接住返回值**）。 | `base`, `offsets` |
| `make_tensor_descriptor` | 构造 Tensor Descriptor（支持 TMA 时由硬件支持 load/store）。 | `base`, `shape`, `strides`, `block_shape`；可选 `padding_option` |
| `load_tensor_descriptor` | 通过描述符按偏移加载一块。 | `desc`, `offsets` |
| `store_tensor_descriptor` | 通过描述符按偏移存储一块。 | `desc`, `offsets`, `value` |

### 3.4 原子操作（均返回操作**前**的旧值）

| API | 用处 | 主要参数 |
|-----|------|----------|
| `atomic_cas` | compare-and-swap。 | `pointer`, `cmp`, `val`；可选 `sem`, `scope` |
| `atomic_xchg` | 原子交换。 | `pointer`, `val`；可选 `mask`, `sem`, `scope` |
| `atomic_add` / `atomic_max` / `atomic_min` | 原子加 / 最大 / 最小。 | `pointer`, `val`；可选 `mask`, `sem`, `scope` |
| `atomic_and` / `atomic_or` / `atomic_xor` | 原子按位与/或/异或。 | 同上 |

### 3.5 形状、视图与类型转换

| API | 用处 | 主要参数 |
|-----|------|----------|
| `full` | 指定形状与 dtype 的常量填充张量。 | `shape`, `value`, `dtype` |
| `broadcast` / `broadcast_to` | 两元广播或广播到目标形状。 | `input`（及 `other` 或 `*shape`） |
| `trans` | 维置换；无 `dims` 时交换最后两维。 | `input`, `*dims`（可选） |
| `permute` | 显式维置换（无默认交换两维）。 | `input`, `*dims` |
| `cat` | 沿维拼接两块。 | `input`, `other`, `can_reorder` |
| `join` | 在**新最小维**上堆叠两tensor。 | `a`, `b` |
| `split` | 沿最后一维（大小须为 2）拆成两个tensor。 | `a` |
| `reshape` | 同元素数改形状。 | `input`, `*shape`, `can_reorder` |
| `view` | **已弃用**，请用 `reshape(..., can_reorder=True)`。 | 同 reshape 旧用法 |
| `expand_dims` | 插入长度为 1 的新维。 | `input`, `axis`（int 或序列） |
| `cast` | 数值转换或 `bitcast`。 | `input`, `dtype`；可选 `fp_downcast_rounding`, `bitcast` |
| `gather` | 沿维 gather。 | `src`, `index`, `axis` |
| `slice` | 表示切片 `start:stop:step` 的 Triton 对象。 | `slice(start, stop, step)` |

### 3.6 线性代数

| API | 用处 | 主要参数 |
|-----|------|----------|
| `dot` | 2D/3D 块矩阵乘（3D 为 batch）。 | `input`, `other`；可选 `acc`, `input_precision`, `allow_tf32`, `max_num_imprecise_acc`, `out_dtype` |
| `dot_scaled` | 微缩放点积（MX 格式等）。 | `lhs`, `lhs_scale`, `lhs_format`, `rhs`, `rhs_scale`, `rhs_format`；可选 `acc`, `fast_math`, `lhs_k_pack`, `rhs_k_pack`, `out_dtype` |

### 3.7 逐元素算术（包根，与 `tl.math` 重叠部分）

| API | 用处 | 主要参数 |
|-----|------|----------|
| `add` / `sub` / `mul` | 逐元素加减乘。 | `x`, `y`；可选 `sanitize_overflow`（`constexpr`） |
| `minimum` / `maximum` / `clamp` | 逐元素最小/最大/夹紧。 | 两元或 `clamp(x, min, max)`；可选 `propagate_nan` |
| `where` | 按条件选 `x` 或 `y`（**两侧都会求值**）。 | `condition`, `x`, `y` |

### 3.8 归约、扫描与直方图

| API | 用处 | 主要参数 |
|-----|------|----------|
| `reduce` | 用户自定义 `combine_fn` 归约（可 tuple 输入）。 | `input`, `axis`, `combine_fn`, `keep_dims` |
| `associative_scan` | 结合律扫描（前缀型）。 | `input`, `axis`, `combine_fn`, `reverse` |
| `histogram` | 宽度为 1、从 0 起的直方图。 | `input`, `num_bins`；可选 `mask` |

### 3.9 `tl.standard`（包根再导出）

| API | 用处 | 主要参数 |
|-----|------|----------|
| `cdiv` | 向上取整除法 `(x + div - 1) // div`。 | `x`, `div` |
| `sigmoid` | Sigmoid。 | `x` |
| `softmax` | Softmax。 | `x`；可选 `dim`, `keep_dims`, `ieee_rounding` |
| `ravel` | 展平为连续视图。 | `x`；可选 `can_reorder` |
| `swizzle2d` | 2D 索引 swizzle（见 docstring 图示）。 | `i`, `j`, `size_i`, `size_j`, `size_g` |
| `zeros` / `zeros_like` | 全零张量。 | `shape`, `dtype` / `input` |
| `max` / `min` | 沿轴最大/最小；可返回索引。 | `input`；可选 `axis`, `return_indices`, `return_indices_tie_break_left`, `keep_dims` |
| `argmax` / `argmin` | 最大/最小值的索引。 | `input`, `axis`；可选 `tie_break_left`/`keep_dims` |
| `sum` | 求和。 | `input`；可选 `axis`, `keep_dims`, `dtype` |
| `xor_sum` | 整数按位异或归约。 | `input`；可选 `axis`, `keep_dims` |
| `reduce_or` | 整数按位或归约。 | `input`, `axis`；可选 `keep_dims` |
| `cumsum` / `cumprod` | 前缀和/前缀积。 | `input`；可选 `axis`, `reverse`, `dtype`（`cumsum`） |
| `sort` | 排序（当前主要支持**最内维**）。 | `x`；可选 `dim`, `descending` |
| `topk` | 取前 k 大（基于 sort 实现）。 | `x`, `k`；可选 `dim` |
| `bitonic_merge` | Bitonic 归并步骤（高级排序网络）。 | `x`；可选 `dim`, `descending` |
| `flip` | 沿维翻转（该维长度须为 2 的幂）。 | `x`；可选 `dim` |
| `interleave` | 沿最后一维交错 `a`、`b`。 | `a`, `b` |

### 3.10 `tl.math`（包根再导出）

| API | 用处 | 主要参数 |
|-----|------|----------|
| `umulhi` | 无符号乘积的高半部分。 | `x`, `y`（整型，见 dtype 检查） |
| `exp`, `exp2`, `log`, `log2` | 指数/对数。 | `x` |
| `cos`, `sin`, `sqrt`, `sqrt_rn`, `rsqrt` | 三角/平方根类。 | `x` |
| `abs` | 绝对值（按 dtype 分支）。 | `x` |
| `fdiv` | 快速除法。 | `x`, `y`；可选 `ieee_rounding` |
| `div_rn` | IEEE 最近偶数舍入除法（fp32）。 | `x`, `y` |
| `erf`, `floor`, `ceil` | 误差函数/下取整/上取整。 | `x` |
| `fma` | 融合乘加。 | `x`, `y`, `z` |

另：`import triton.language.math as m` 可访问同一批函数模块命名空间。

### 3.11 `tl.random`

| API | 用处 | 主要参数 |
|-----|------|----------|
| `philox_impl` | Philox 状态机若干轮更新。 | `c0`…`c3`, `k0`, `k1`；`n_rounds`（constexpr） |
| `philox` | 带种子的 Philox 一步。 | `seed`, `c0`…`c3`；可选 `n_rounds` |
| `randint` / `randint4x` | 伪随机 int 块（4x 更高效）。 | `seed`, `offset`；可选 `n_rounds` |
| `uint_to_uniform_float` | 随机整数映射到 `[0,1)` 浮点。 | `x` |
| `rand` / `rand4x` | `U(0,1)` 的 float32 块。 | `seed`, `offset`；可选 `n_rounds` |
| `randn` / `randn4x` | 正态近似（基于均匀分布变换）。 | `seed`, `offset`；等 |
| `pair_uniform_to_normal` | 两均匀样本变一对正态样本。 | （见 `random.py`） |

### 3.12 元编程、调试与内联汇编

| API | 用处 | 主要参数 |
|-----|------|----------|
| `must_use_result` | 装饰器：返回值未使用则报错。 | 目标函数或 `(msg)` 形式 |
| `static_print` | **编译期**打印（类似 Python `print` 参数）。 | `*values`, `sep`, `end`, … |
| `static_assert` | **编译期**断言。 | `cond`, `msg` |
| `assume` | 告知编译器条件恒为真。 | `cond` |
| `device_print` | **设备端**打印（需调试环境等）。 | `prefix`, `*args`；可选 `hex` |
| `device_assert` | **设备端**断言（需 `TRITON_DEBUG` 等）。 | `cond`, `msg`；可选 `mask` |
| `debug_barrier` | 块内线程同步屏障。 | 无 |
| `multiple_of` | 编译提示：值均为某倍数。 | `input`, `values`（`constexpr` 序列） |
| `max_contiguous` | 编译提示：前若干元素连续。 | `input`, `values` |
| `max_constancy` | 编译提示：前若干元素分段常数。 | `input`, `values` |
| `inline_asm_elementwise` | 逐元素内联汇编（如 PTX）。 | `asm`, `constraints`, `args`, `dtype`, `is_pure`, `pack` |
| `map_elementwise` | 将标量 JIT 函数映射到张量各元素（可分支短路）。 | `scalar_fn`, `*args`；可选 `pack` |

### 3.13 迭代与控制流辅助

| API | 用处 | 主要参数 |
|-----|------|----------|
| `static_range` | 用于 `for`；**编译期**界与步长，利于展开。 | `end` 或 `start`, `end`；可选 `step`（均为 `constexpr`） |
| `range` | 设备循环迭代；可带软件流水线等提示。 | `end` 或 `start`, `end`；可选 `step`, `num_stages`, `loop_unroll_factor`, `disallow_acc_multi_buffer`, `flatten`, `warp_specialize`, `disable_licm` |
| `condition` | `while` 条件包装，可关 LICM。 | `arg1`（条件）, `disable_licm` |

### 3.14 子模块（非函数列表）

| API | 用处 | 说明 |
|-----|------|------|
| `extra` | 后端扩展包入口。 | 子模块随构建注册，如 `extra.cuda` |
| `math` | 数学函数模块对象。 | 与包根再导出内容一致 |
| `target_info` | 目标信息查询。 | 见模块内 API |

---

## 四、`triton.experimental.gluon` API 速查（用处 / 主要参数）

### 4.1 包根 `triton.experimental.gluon`

| API | 用处 | 主要参数 |
|-----|------|----------|
| `jit` | Gluon 内核 JIT，走 `Language.GLUON` / ttgir。 | `fn`；关键字：`version`, `repr`, `launch_metadata`, `do_not_specialize`, `do_not_specialize_on_alignment`, `debug`, `noinline` |
| `constexpr_function` | 标记可在编译期执行的辅助函数。 | 与主包 `constexpr_function` 用法一致 |
| `must_use_result` | 同 `tl`：强制使用返回值。 | 见 3.12 |
| `nvidia` | NVIDIA 目标相关子模块树。 | `hopper`, `blackwell` 等 |
| `amd` | AMD 目标相关子模块树。 | 如 `gfx1250` |

### 4.2 `triton.experimental.gluon.language`（与经典 `tl` 对照）

该包根从 `_core`、`_layouts`、`_math`、`_standard` 再导出，**语义贴近 TTGPUIR**，部分名称与 `tl` 相同但实现不同。常见项如下（参数模式多与 `tl` 类似，细节以 `gluon/language/_core.py` 为准）。

| API | 用处 | 主要参数（与 classic 差异提示） |
|-----|------|--------------------------------|
| `allocate_shared_memory` | 在 Gluon 中分配带 **layout** 的 shared 区，返回描述符。 | `element_ty`, `shape`, `layout`；可选 `value` |
| `thread_barrier` | 线程级同步屏障。 | 无用户参数（`_semantic` 注入） |
| `warp_specialize` | 将不同工作划分到 warp 子组执行（高级控制流）。 | `functions_and_args`, `worker_num_warps`, `worker_num_regs` |
| `dot_fma` | 显式 FMA 路径的块乘/点积累加形式。 | `a`, `b`, `acc` |
| `num_warps` / `num_ctas` | 读取当前内核的 warp 数 / CTA 数（编译期或语义层）。 | 无用户参数 |
| `set_auto_layout` | 为值关联自动推导的 layout 提示。 | `value`, `layout` |
| `to_linear_layout` / `convert_layout` | layout 转换。 | 见 `_core` 各签名 |
| `to_tensor` | 转为张量表示。 | `x` |
| `bank_conflicts` | 估算/查询分布类型与 shared 类型间的 bank 冲突数。 | `distr_ty`, `shared_ty` |
| `fp4_to_fp` | FP4 数据沿某维解包并 cast 到目标浮点元素类型。 | `src`, `elem_type`, `axis` |
| `distributed_type` / `shared_memory_descriptor(_type)` | 分布式或 SMEM 描述类型。 | 见 `_core` / `_layouts` |
| `_math` 再导出 | 与 `tl.math` 类似的逐元素数学。 | `exp`, `log`, `fma`, … |
| `_standard` 再导出 | `cdiv`, `sum`, `zeros` 等。 | 与 classic 类似 |
| **Layouts** | 显式指定数据在 warp/SMEM 中的排布。 | `AutoLayout`, `BlockedLayout`, `NVMMASharedLayout`, `SwizzledSharedLayout`, `CoalescedLayout`, … |

### 4.3 `gluon.language.nvidia` / `amd`（按架构）

- **`nvidia.hopper`**：`tma`（张量内存加速器描述符与 load/store）、`mbarrier`、`mma_v2`、`async_copy`、`warpgroup_mma` 等；各子模块内函数均有 docstring（参数因原语而异）。
- **`nvidia.blackwell`**：新一代指令/类型封装（如 `tma`、`float2` 等子模块）。
- **`nvidia.ampere`**：`async_copy`、`mbarrier`、`mma_v2` 等。
- **`amd.*`**：`cdna3`/`cdna4`/`gfx1250`/`rdna3`/`rdna4` 下的 `async_copy`、`mbarrier`、`tdm` 等目标原语。

**说明**：厂商子包 API 数量多且随硬件迭代快，本文只列模块职责；写内核时请直接阅读对应目录下 `*.py` 中 `@builtin` 或 JIT 函数的参数表。

### 4.4 `gluon.language.extra`

| API | 用处 | 主要参数 |
|-----|------|----------|
| `libdevice` | 与 `triton.language.extra.libdevice` 相同命名空间。 | 各设备数学函数见 libdevice 封装 |

---

## 五、对照小结

| 维度 | `triton.language` | `triton.experimental.gluon` |
|------|-------------------|-----------------------------|
| 稳定性 | 稳定、面向用户 | 实验性 |
| JIT | `triton.jit` | `triton.experimental.gluon.jit` |
| IR 起点 | TTIR → TTGIR → … | TTGIR（`Language.GLUON`） |
| 布局/硬件原语 | 相对隐藏，偏自动 | 显式 layout、更多厂商原语 |
| 主用途 | 通用 GPU 内核 | 探索下一代 API、细粒度映射硬件 |

如需扩展阅读：经典编译阶段见同目录其他笔记（如 `triton.md`）；环境变量见 `env.md`。
