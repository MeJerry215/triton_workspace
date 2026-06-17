# 开发与阅读 LLVM 源码：基础知识提纲

本文面向需要在 `llvm-project` 里 **读代码、改 pass、跟调试** 的开发者，整理 **应先掌握的概念** 与 **和源码的对应关系**。LLVM 仓库同时包含 **LLVM IR 后端**、**MLIR**、**Clang** 等；你当前涉及的 Triton / MLIR 路径会 **更重 MLIR**，但 LLVM IR 与 Pass 框架仍是公共底座，建议都了解轮廓。

---

## 1. 仓库里有什么（先建立地图）

| 目录 | 内容 |
|------|------|
| `llvm/` | LLVM 核心：IR、Pass、CodeGen、目标无关优化、各 Target |
| `mlir/` | MLIR：多层 IR、方言、Pass、Lowering 到 LLVM 等 |
| `clang/` | C/C++/ObjC 前端（若只做 MLIR 可后学） |
| `compiler-rt/`、`libcxx/` 等 | 运行时、标准库实现等 |
| `lld/` | 链接器 |

**心智模型**：`llvm/lib/IR` 定义 **LLVM IR 数据结构**；`llvm/lib/Transforms` 是 **IR 上的变换**；`mlir` 是 **另一套可扩展 IR**，常通过 **LLVM Dialect + Translation** 落到 LLVM IR。

---

## 2. C++ 与 LLVM 风格（读源码时每天都会遇到）

### 2.1 容器与 ADT（`llvm/ADT/`）

源码里大量 **不用 `std::vector` 而用 `SmallVector`**（小容量栈上分配）、**`DenseMap` / `DenseSet`**（哈希表，指针/整数 key 快）、**`StringRef`**（非拥有字符串视图）、**`ArrayRef`**（只读数组视图）。读 API 时注意：**返回值/参数常是 `ArrayRef`/`StringRef`，不拷贝字符串**。

### 2.2 `cast` / `dyn_cast` / `isa`

LLVM 用 **类层次 + RTTI**（如 `Instruction`、`CallInst`）。习惯写法：

- `isa<T>(x)`：是否为某类型  
- `cast<T>(x)`：已知是则转换  
- `dyn_cast<T>(x)`：若是则返回指针，否则 `nullptr`  

MLIR 里类似：`dyn_cast<SomeOp>(op)` 等。

### 2.3 错误处理

常见 **`Error` / `Expected<T>`**（`llvm/Support/Error.h`），或函数返回 **`bool` + `Error`**。读代码时跟 **`fail()` / `success()`**。

### 2.4 调试输出

- **`LLVM_DEBUG(...)`** 宏：仅在 **Debug 构建** 且开启对应 `-debug` 时打印。  
- **`errs()` / `dbgs()`**：`raw_ostream` 输出；`dbgs()` 常给调试。

### 2.5 注释里的术语

- **TODO / FIXME**：未竟或待修。  
- **NFC**：**No Functional Change**，纯重构、无行为变化。

---

## 3. LLVM IR 核心（读 `llvm/lib/IR`、`include/llvm/IR`）

### 3.1 层次结构

```text
Module
 └── Function
       └── BasicBlock
             └── Instruction（SSA 形式）
```

- **Module**：一个翻译单元级容器，含函数、全局变量、元数据等。  
- **Function**：有 **参数列表**、**返回类型**、**BasicBlock 列表**，即控制流图（CFG）。  
- **BasicBlock**：**单入口**、内部指令**顺序执行**，**最后一条必须是 terminator**（`br`、`ret`、`switch` 等）。  
- **Instruction**：产生或不产生 `Value`；类型为 **`Value` 子类**。

### 3.2 SSA 与 PHI

LLVM IR 是 **静态单赋值**：每个 `Value` 在 **某条指令或参数** 处唯一定义。块合并处用 **`PHINode`** 汇合来自不同前驱的值。

### 3.3 类型系统

**`Type`**：`IntegerType`、`PointerType`、`FunctionType`、`StructType` 等。理解 **指针元素类型**、**opaque pointer**（新版本 LLVM 里指针不携带 pointee 类型，需从上下文推断）。

### 3.4 IR 的两种视角

- **内存中的图**：`Instruction*`、`User`/`Use` 链表（谁用谁）。  
- **文本形式**：`.ll` 文件，与内存 IR 一一对应，适合用 `opt` 做实验。

### 3.5 `IRBuilder`

在指定插入点 **创建指令** 的辅助类（`llvm/IR/IRBuilder.h`）。写 pass 时常在某 `BasicBlock` 末尾或某指令前插入。

---

## 4. Pass 与 Pass Manager（LLVM IR）

### 4.1 Pass 做什么

**Pass** 是对 **Module / Function / Loop 等单元** 的一次遍历或变换。可 **读分析结果**（Analysis），可 **改 IR**（Transform）。

### 4.2 新 Pass Manager（当前主流）

入口概念：**`PassManager<Module>`**、**`FunctionPassManager`** 等；Pass 常实现 **`PassInfoMixin<YourPass>`** 模式。Pass 之间通过 **AnalysisManager** 查询 **`AAResults`、`DominatorTree`** 等，并声明 **失效关系**（preserved analyses）。

读代码时可从 **`llvm/lib/Transforms`** 里选一个简单 transform（如部分 `InstCombine` 或 `SimplifyCFG` 的接口）看 **注册与 run 函数**。

### 4.3 与 MLIR Pass 的对比（避免混）

| | LLVM IR Pass | MLIR Pass |
|---|--------------|-----------|
| 操作对象 | `Module` / `Function` / … | `Operation`（任意方言 op） |
| 注册 | Pass 插件、`opt` 管线 | `PassRegistration`、TableGen `def` |
| 分析 | `AnalysisManager` | `AnalysisManager`（概念类似） |

---

## 5. MLIR 核心（读 `mlir/`，与 Triton/LLVM 子项目强相关）

### 5.1 结构：Operation → Region → Block → Operation

- **Operation**：带 opcode、属性、操作数、结果；可挂 **多个 Region**。  
- **Region**：**Block 的列表**；表示一段结构化控制流区域。  
- **Block**：MLIR 的「基本块」，内含 **Operation** 序列与 **块参数**（常作 MLIR 的 φ 等价物）。  
- **Value**：`OpResult` 与 `BlockArgument` 等。

**最小示例（单 Region、单 Block）**：`func.func` 是一条 **Operation**；花括号里是它的 **第 0 个 Region**（函数体）；该 Region 里默认有 **一个 Block**（未显式写 `^bb0` 时，打印机常省略块名）；块内 `arith.addi`、`func.return` 等都是 **Operation**。`%a` 是该 **Block** 的 **块参数**（函数形参）；`%0` 是 `arith.addi` 的 **结果 Value**（`OpResult`）。

```mlir
// Operation: func.func ； Region: { ... } ； Block: 内含下面三行所在的隐式首块
func.func @f(%a: i32) -> i32 {
  %0 = arith.addi %a, %a : i32   // Operation，产生 Value %0
  func.return %0 : i32           // Operation（终结符）
}
```

**稍复杂示例（一个 Operation 挂两个 Region）**：`scf.if` 本身是一条 **Operation**；`{ ... }` 与 `else { ... }` 分别是它的 **then Region** 与 **else Region**（各含至少一个 Block）。`scf.yield` 把值交回给 `scf.if` 的结果 `%r`。

```mlir
func.func @g(%cond: i1, %x: i32) -> i32 {
  %r = scf.if %cond -> (i32) {
    // Region #0（then）：通常一个 Block，以 scf.yield 结束
    scf.yield %x : i32
  } else {
    // Region #1（else）
    %c1 = arith.constant 1 : i32
    %y = arith.addi %x, %c1 : i32
    scf.yield %y : i32
  }
  func.return %r : i32
}
```

**容易误解的一点：Region 并不是「在 Block 里面」**。正确层级永远是：

**Operation → Region → Block → Operation → …**（子 op 若再带 Region，则继续往下嵌套）。

`.mlir` 文本里 **`scf.if` 与两个 `{ ... }` 缩进较深**，容易看成「先进入块再进入 region」。实际上：`func.func` 的 **函数体 Region** 里有一个 **入口 Block**；这个 Block 里依次排着 **`scf.if`** 和 **`func.return`** 等 **Operation**。其中 **`scf.if` 这一条 op** 自己再挂 **两个子 Region**（then / else）；**每个子 Region 内部**才有自己的 **Block 列表**（上例里各一个块，里面是 `scf.yield` 等）。

简图（只看嵌套，不写全 SSA）：

```text
func.func (Operation)
└── Region：函数体
    └── Block^bb0（入口块）
        ├── scf.if (Operation)  ← 与下面两个 Region 是父子关系，不是「块套在块里」的平级乱序
        │   ├── Region：then
        │   │   └── Block
        │   │       └── scf.yield ...
        │   └── Region：else
        │       └── Block
        │           └── arith...; scf.yield ...
        └── func.return (Operation)
```

对应关系可记：**大括号包的是挂在某条 Operation 上的 Region；Region 里才是 Block；块里才是子 Operation**。不要被缩进骗成「Region 在 Block 内」——**Block 在 Region 内**。

**一个 Region 里多个 Block 长什么样（「else 路径」多块）**：`scf.if` 的 then/else 在 SCF 里通常 **各自只允许一个 Block**（结构化 if）。若要在 **语义上的 else** 里出现 **多条基本块**（先算一步再跳下一步），常见写法是在 **函数体这一个 Region** 里用 **`cf` 控制流** 展开，下面 **整个花括号仍是 `func.func` 的唯一函数体 Region**，其中包含 **`^then`、`^else`、`^else_mid`、`^merge` 等多个 Block**；走 else 时会经过 **`^else` → `^else_mid`** 两个块再汇入 `^merge`。

```mlir
// 需加载 cf 方言。函数体 = 1 个 Region，内含 4 个 Block（入口块隐式为第一条分支所在块时可写为从 cond_br 开始）
func.func @cfg_if_else(%cond: i1, %x: i32) -> i32 {
  cf.cond_br %cond, ^then, ^else
^then:
  cf.br ^merge(%x : i32)
^else:
  %t = arith.constant 0 : i32
  cf.br ^else_mid
^else_mid:
  %c1 = arith.constant 1 : i32
  %y = arith.addi %x, %c1 : i32
  cf.br ^merge(%y : i32)
^merge(%r : i32):
  func.return %r : i32
}
```

上例中：**同一 Region**（函数体）的 Block 列表顺序由终结符串联；**else 侧**占 **`^else` 与 `^else_mid` 两个 Block**，比单块 else 多了一次跳转，用来示意「Region 内多块 CFG」。

### 5.2 方言（Dialect）

**Dialect** 是 MLIR 里 **一组协同设计的 IR 扩展** 的注册单元，不是「只有一堆 op 名字」那么简单。

- **命名空间**：打印成 `dialect.op`（如 `arith.addi`、`func.call`）。**同一 opcode 名在不同 dialect 里可以并存**，靠前缀区分。
- **通常包含**：
  - **Operations**：用 ODS（TableGen）或 C++ 注册的具体算子与控制流构造。
  - **Types / Attributes**（可选）：该层抽象需要的类型与属性（如 `tensor` 在 `builtin`，部分方言自定义 attribute）。
  - **解析与打印**：如何把文本解析成 op、如何打印回 `.mlir`。
  - **Dialect 级扩展**：例如向 **RewritePatternSet** 注册 canonicalization、或注册 **DialectConversion** 里用的目标模式。
- **加载**：使用某 dialect 的 op 前，**Context** 里要 **load** 对应 `Dialect` 对象（`mlir-opt`、嵌入编译器里通常会加载一整包方言）。

**Lowering（降级）**：多指把 **高抽象方言** 里的 op **重写**成 **更低抽象** 的 op（可能在别的 dialect 里），反复进行直到接近 **LLVM Dialect**，再用 **`LLVMIRTranslation`** 之类机制 **translate** 成 **LLVM IR**（这是「出 MLIR 进 LLVM」的一步，和「在同一 MLIR 里改 op」不同）。

---

### 5.3 接口（Interface）——和「方言」是什么关系

容易混淆点：**Interface 不是「某个 dialect 自带的子 API」**，而是挂在 **具体的 Operation 或 Type** 上的 **能力描述**；**同一个 Interface 可以由多个不同 dialect 里的 op 实现**。

| 概念 | 作用 |
|------|------|
| **Trait** | 编译期固定性质（如 `Terminator`、`SameOperandsAndResultType`），常决定 **结构合法性** 与 **模式匹配**。 |
| **OpInterface** | 给 **Operation** 增加一组 **运行时多态** 方法（C++ 里通过 `InterfaceMap` 分发）。例如「像函数一样可被调用」不只有一种 op，`func.call` 与别的方言的 call 都可挂 **`CallOpInterface`**。 |
| **TypeInterface / AttrInterface** | 同理，挂在 **Type** / **Attribute** 上。 |

**为何需要 Interface**：通用 Pass（如 **Inliner、CallGraph**）不可能为每个方言的 call op 写死类型；它们只认 **`CallOpInterface` / `CallableOpInterface`**：只要能解析 **被调者、实参、结果**，就能走同一套算法。

**常见例子**（均在 `include/mlir/Interfaces/` 或各方言扩展里）：

- **`CallOpInterface`**：表示一次调用；可查询 callee、实参等。
- **`CallableOpInterface`**：表示「带一块可执行 Region 的可调用体」；与 **CallGraph 的节点**对应。
- **`SymbolOpInterface`**：符号名、可见性、是否可删除等；**符号表**与 **DCE** 会用。
- **`BranchOpInterface` / `RegionBranchOpInterface`**：控制流分析与 **SSACFG** 相关。
- **`InlinerInterface`**（方言实现）：**方言负责**「如何把 return 变成跳转、多块内联合法性」；**通用 inliner** 只调接口。

**Dialect 与 Interface 的分工一句话**：**Dialect 决定「这是哪一族 op / 类型」**；**Interface 决定「通用变换把不把你当成某种角色」**。读 Triton / LLVM 子树时，看到 `dyn_cast<CallableOpInterface>(op)` 就是在走 **接口**；看到 `triton` 前缀则是在看 **方言**。

### 5.4 ODS / TableGen（`*.td`）——在 MLIR 里具体怎么用

**ODS（Operation Definition Specification）** 是 MLIR 用 **TableGen（`.td`）** 描述 **Dialect / Operation / Attribute / Type** 的规范。你在 `include/mlir/Dialect/...` 里看到的 **`def XxxOp : ...`** 不是 C++，而是 **TableGen 记录**：构建时由 **`mlir-tblgen`** 按所选 **后端** 生成 **`*.inc`** 或内嵌片段，再被 **`#include`** 进真正的 **`.h` / `.cpp`**。

更一般的 TableGen 背景（工具链、后端名、与 LLVM 后端 `.td` 的区别）见 **§7**；本节只回答 **ODS 工作流** 与 **和 C++ 的对应关系**。

#### 5.4.1 `.td` 里的 `def` 是什么、谁在生成

- **`def FooOp : ...`**：声明一条 **TableGen 记录**，描述一个 op 的 **参数、结果、trait、region** 等。  
- **生成**：由 **CMake** 里 **`mlir_tablegen(...)`**（或等价封装）调用 **`mlir-tblgen`**，典型后端包括：  
  - **`--gen-op-decls`** / **`--gen-op-defs`**：生成 op 的 **声明/定义** 片段；  
  - **`--gen-dialect-decls`** 等：方言注册、doc 等（视 `CMakeLists.txt` 而定）。  
- **日常开发**：一般 **不必手敲** `mlir-tblgen`；改 **`.td` → 重新编译**，构建系统会重跑生成。

#### 5.4.2 怎么判断一段 C++ 是「TableGen 生成的」

同时满足越多条越可信：

1. **文件/包含名**：常见模式是手写 **`FooOps.h`** 只有几行 **`#include "FooOps.h.inc"`**，**真正大头在 `.inc`**；生成物常在 **build 树**（如 `build/tools/mlir/include/...`）或 **源码树里由生成脚本检入的 `.inc`**（视项目策略而定）。  
2. **文件头注释**：生成文件常带 **`Autogenerated`** / **`DO NOT EDIT`** / **`mlir-tblgen`** 等字样（具体措辞随版本略变）。  
3. **代码形态**：大量 **样板**——`::build`、`getXxxAttr`、`verify`、`parse`、`print` 的重复模式；或整段 **`GET_OP_LIST`**、**`GEN_PASS_*`** 宏。  
4. **不在「手写目录」里**：若同一目录既有 **`MyDialect.cpp`**（手写）又有 **`MyDialect.cpp.inc`**（生成），后者通常不直接编辑。

**Pass** 同理：`Passes.td` 里 **`def Inliner`** 生成 **`impl::InlinerBase`**、选项成员等；**`InlinerPass.cpp`** 里 **`#define GEN_PASS_DEF_INLINER`** 即展开生成代码。

#### 5.4.3 从 C++（或 `.inc`）反查对应 `.td`

按优先级试：

1. **在生成文件顶部或注释里** 有时会写 **源 `.td` 路径**（若有）。  
2. **在 `llvm-project/mlir/include`（及 `lib`）里全文搜索**：  
   - 已知 **Op 类名**（如 `AddIOp`）→ 搜 **`def AddIOp`** 或 **`class AddIOp`（ODS 继承名）**；  
   - 已知 **op 打印名**（如 `arith.addi`）→ 搜 **`"arith.addi"`** 或 **`addi` 在对应 dialect 的 `.td`**。  
3. **看方言的 `CMakeLists.txt`**：搜索 **`mlir_tablegen`**，参数里会列出 **输入 `.td` 与生成 `.inc` 文件名**，可建立 **「哪个 td 产哪个 inc」** 的映射。  
4. **从 `#include`**：打开 **包装用 `.h`**，看它 **`#include` 哪个 `.inc`**，再在 **build 目录** 找到该 `.inc` 的生成规则（或同仓库里已提交的 `.inc`），回溯 **CMake 中的输入 td**。

#### 5.4.4 修改流程（避免改错文件）

- **改 op 语义 / 签名 / trait**：改 **`.td`**，必要时再改 **少量手写** 的 **`*Dialect.cpp`（注册）** 或 **额外 C++ 辅助**。  
- **不要**直接改 **`*.h.inc` / `*.cpp.inc`** 里生成体（下次生成会覆盖）。  
- 若 IDE **跳不到 `def`**：在 **`mlir/include`** 里 **文本搜索 `def YourOp`**，或先 **完整 build** 让索引包含 **build 下生成文件**。

### 5.5 Pass 管线

**`mlir-opt`**：命令行拼 pass pipeline。Pass 定义在 **`include/mlir/Transforms/Passes.td`** 等，与 LLVM 侧 **`mlir/lib/Transforms`** 实现对应。

---

## 6. CodeGen 与 Target（可选深入）

若读 **指令选择、寄存器分配、ELF 输出**：

- **`llvm/lib/CodeGen`**：目标无关机器码阶段。  
- **`llvm/lib/Target/<Arch>`**：具体架构。  

顺序大致：**SelectionDAG / GlobalISel** → **寄存器分配** → **汇编/对象文件**。初学可后移，除非做后端。

---

## 7. TableGen（LLVM 与 MLIR 共用）

**TableGen** 是 LLVM 工程里的 **代码生成器**：你用一种声明式的 **`.td`（TableGen）语言** 写「记录 / 类层次 / 表数据」，再运行 **`tblgen`** 的某个 **后端（backend）**，输出 **C/C++ 头文件片段、`.inc` 文件、或文档**。目标是 **单一事实来源**：避免手写成百上千份重复的指令枚举、属性列表、pattern 展开。

### 7.1 为什么读 LLVM/MLIR 一定会碰到它

- **LLVM 后端**：目标机器的 **指令格式、寄存器、调用约定、指令选择 DAG pattern** 大量写在 **`lib/Target/<Arch>/*.td`**，生成 C++ 里用的枚举、Matcher、解码表等。  
- **MLIR**：**ODS（Operation Definition Specification）** 用 `.td` 定义 **Operation、Dialect、Traits、Interfaces**；**`Passes.td`** 里定义 **Pass 名称与命令行选项**，生成 `*Passes.h.inc` 等。  
- **Clang** 等子项目也有 TableGen 驱动的表（此处从略）。

不跑构建也能读 `.td`，但 **和实际编译进产物的 C++ 对齐** 时，必须看 **生成结果**（通常在 build 目录的 `*.inc`）。

### 7.2 `.td` 语言在说什么（读文件最小技能）

不必学全语言，能认这些即可：

- **`class` / `def`**：`def` 一条具体记录；`class` 是可继承模板。  
- **`multiclass` / `defm`**：批量展开多条 `def`（减少重复）。  
- **字段与类型**：`string name`、`list<...>`、`dag`（DAG 形状，用于 pattern）。  
- **`include`、注释**：和 C 类似。

读 MLIR 的 op 定义时，你会看到 **`def MyOp : ...`**、**`let arguments = (ins ...)`** 等，都是 ODS 在 TableGen 上的语法糖。

### 7.3 两个常用工具：`llvm-tblgen` 与 `mlir-tblgen`

二者都是 **叫 `tblgen` 的可执行程序**，输入同一种 **`.td` 语法**，差别在于 **内置的「后端（generator）」不同**：同一套 TableGen 语言，**换工具 = 换一批可用的 `--gen-xxx` 开关**，产出面向 **LLVM IR / 目标后端** 或 **MLIR** 的 C++。

**区别一句话**：**`llvm-tblgen`** 服务 **传统 LLVM**（指令、intrinsic、CallingConv、ISel pattern 等）；**`mlir-tblgen`** 服务 **MLIR**（ODS op、Pass 声明、方言注册片段等）。日常开发里 **几乎从不手动选**：**CMake** 在编译 LLVM 目标、编译 MLIR 方言时 **分别调用** 对应可执行文件。

| 工具 | 典型场景（谁在跑） |
|------|-------------------|
| **`llvm-tblgen`** | 构建 **某 Target**（如 RISC-V、ARM）时，处理 **`lib/Target/<Arch>/*.td`**，生成 **指令编码、汇编助记符、DAG 匹配** 等；或处理 **`include/llvm/IR/Intrinsics.td`** 生成 **intrinsic 枚举与实现表**。 |
| **`mlir-tblgen`** | 构建 **MLIR** 时，处理 **`mlir/include/mlir/Dialect/.../*.td`**（ODS）、**`Passes.td`** 等，生成 **`FooOp` 的 C++ 样板**、**`GEN_PASS_*`** 等。 |

**举例（帮助建立直觉，不必背命令）**：

- **LLVM 侧**：`llvm-tblgen` + 后端 **`--print-records`** 或目标相关的 **`--gen-register-info`** 等（具体以该 Target 的 `CMakeLists.txt` 为准），输入类似 **`RISCVInstrInfo.td`**，得到 **`RISCVGen*.inc`**，被 **`RISCV*.cpp` `#include`**。你在改 **汇编格式 / 指令定义** 时触发的往往是这一类。  
- **MLIR 侧**：`mlir-tblgen` + **`--gen-op-decls`** / **`--gen-op-defs`**，输入 **`ArithOps.td`** 里 **`def AddIOp`**，得到 **`ArithOps.h.inc`** 里 **`class AddIOp`** 的声明与实现片段。你在改 **`arith.addi`** 参数、trait 时触发的往往是这一类。

**和「场景」的关系**：读 **LLVM 后端 / intrinsic** 源码时跟 **`llvm-tblgen`**；读 **MLIR 方言 op、mlir-opt 的 Pass** 时跟 **`mlir-tblgen`**。两者 **不混用同一条生成命令**——对同一份 `.td`，若错误地用另一个工具，会 **缺少对应后端** 或 **生成内容不对**。

CMake 里通常写成 **自定义命令**：输入 `.td` + 若干 include 路径 → 输出 **`Something.inc`** → 再由 **`#include "Something.inc"`** 编进 `.cpp`。若你改 `.td` 却 **没触发重新生成**，会出现 **旧宏名、缺符号、与手写 C++ 不一致**。

### 7.4 在 MLIR 里你最常遇到的生成物

- **Op 定义**：`include/mlir/Dialect/...` 下 `*.td` → 生成 **`*Op.h` 里的声明/定义片段** 或独立 `.inc`（视 CMake 写法）。最终得到 **`FooOp::build`**、**`getOperand()`** 等 API。  
- **Pass 定义**：`Passes.td` 里 `def MyPass` → **`GEN_PASS_DECL_*` / `GEN_PASS_DEF_*`** 宏，和 **命令行选项成员变量**。

读 **实现** 时：若函数体只有 **`#define GEN_PASS_DEF_INLINER`** 之类，真正逻辑在 **`.cpp`**；**选项字段名** 往往在 **生成的 `*Base` 类** 里。

### 7.5 在 LLVM 后端里 TableGen 的典型角色（轮廓）

- **指令与编码**：寄存器类、立即数范围、汇编助记符。  
- **SelectionDAG / GlobalISel**：**`dag` pattern** 描述「IR 上这样一段子图 → 选成哪条机器指令」。  
- **CallingConv、RegisterInfo**：表驱动生成大量 **switch/数组**，减少手写错误。

你若不做目标后端，只需知道：**目标相关 C++ 里大量 `// Generated by TableGen` 别手改**。

### 7.6 实操：如何跟源码

1. 在 **`include` / `lib`** 里搜 **`.td`**，打开你关心的 `def`。  
2. 在 **build 目录** 搜同名 **`*.inc`** 或看 **CMakeLists** 里 **`mlir_tablegen` / `tablegen`** 规则，确认 **生成文件名**。  
3. 在 **`.cpp` / `.h`** 里 **`#include`** 或 **`grep GEN_` / `grep 你的 Pass 名`**，看 **宏展开后的接口**。  
4. 改行为时：**优先改 `.td` 并重新构建**；只有生成器不支持时才改 **生成器 C++**（`mlir-tblgen` 源码，高阶）。

### 7.7 常见问题

- **跳转不到 Op 定义**：类名在 **生成代码** 里，IDE 需索引 build 目录或 **先完整编译** 一次。  
- **改了 `.td` 没生效**：清 **ninja 目标** 或删对应 `.inc` 再编；确认 **CMake 依赖** 是否把该 `.td` 列为输入。  
- **与 Inliner 等 Pass 笔记衔接**：MLIR 的 **`InlinerPass`** 选项与 **`createInlinerPass`** 声明，很多来自 **`Passes.td`**，不是手写在头文件里。

---

## 8. 构建、测试、调试（动手必备）

### 8.1 CMake

LLVM 用 **out-of-tree 构建**。常用 **`-DLLVM_ENABLE_PROJECTS="mlir;clang"`**、**`-DCMAKE_BUILD_TYPE=Debug`**（调试）、**`-DLLVM_ENABLE_ASSERTIONS=ON`**。  
生成 **`compile_commands.json`**（`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`）给 **clangd / IDE** 跳转。

### 8.2 测试

- **`lit`**：驱动测试。  
- **`FileCheck`**：检查命令输出或 IR 文本是否匹配 **注解**。  
- 测试多在 **`llvm/test`、`mlir/test`**，读几条 `.ll` / `.mlir` 测试比只看文档快。

### 8.3 常用命令

- **`opt -passes=...`**：跑 LLVM IR pass。  
- **`mlir-opt ...`**：跑 MLIR pass。  
- **`llc`**：IR → 汇编；**`lli`**（JIT 执行，视构建而定）。

---

## 9. 建议学习顺序（可并行）

1. **LLVM IR 手册**（官方 LangRef）：Module / Function / SSA / 主要指令。  
2. 在纸上画 **一个小函数的 BasicBlock 与 CFG**，对照 `.ll`。  
3. 读一个 **简单 FunctionPass** 的源码（如何遍历 BB、如何用 `IRBuilder`）。  
4. **MLIR**：Operation / Region / Block；跑 **`mlir-opt --help`**，读一条 pipeline。  
5. 再读与你任务相关的子系统（如 **MLIR Inliner + CallGraph**）。

---

## 10. 官方与社区资源

- **LLVM Language Reference**：IR 语义权威来源。  
- **MLIR 文档**（`mlir/docs/` 与官网）：Dialect、Traits、Pass。  
- **LLVM Developer Policy**、**Coding Standards**：提交代码前必读。  

---

## 11. 与本仓库笔记的衔接

- 若主要跟 **Triton / MLIR**：优先巩固 **Region-Block-Operation、Symbol、CallGraph、Inliner**。  
- 若跟 **LLVM 优化与后端**：优先 **LLVM IR + PassManager + 目标相关 CodeGen**。  

阅读源码时：**先确定文件在 `llvm/` 还是 `mlir/`**，再选用 **BasicBlock** 还是 **Block** 的心智模型，避免混淆。

---

*文档随学习进度可自行在后续章节补充：例如具体 Pass 案例、GDB/LLDB 断点位置、或某 Target 的指令 lowering 笔记。*
