# `dot_scaled` 兼容桥实现详解：从 Python DSL 到 Triton intrinsic

> 面向读者：了解 Python 和基本矩阵乘法，但刚接触 AI 编译器、SSA、lowering 或 Triton  
> 对应 ntops 版本：`efa2303de8b85019d9d011a5cadce26321ca0a85`  
> 对应 ninetoothed 环境：0.26.0，源码提交 `b77f930dc6c8b016e09adf33570d55a7bc8376c1`  
> 核心实现：[scaled_grouped_mm.py](../src/ntops/torch/scaled_grouped_mm.py)  
> 结构测试：[test_scaled_grouped_mm.py](../tests/test_scaled_grouped_mm.py)

## 1. 先给出结论

`scaled_grouped_mm` 的 kernel application 调用了 `ntl.dot_scaled`。当前 ninetoothed 版本能看见
这个 Python 调用，却不知道它是一种需要 **二维 block tensor** 的特殊矩阵运算，最终会生成：

```python
dot_scaled(lhs, None, "bf16", rhs, rhs_scale, "e2m1", ...)
```

这里有两个问题：

1. 函数名是裸的 `dot_scaled`，而 Triton intrinsic 的正确名字是 `tl.dot_scaled`；
2. 更深层的问题是，编译器没有把它标记成 dot 类 block 运算，可能无法选择正确的 block
   program 路径和操作数坐标。

兼容桥不是简单做一次字符串替换，而是在编译流水线的两个位置配合工作：

```text
Python application
      │
      │ ① 前端补丁：识别 dot_scaled
      ▼
带 ntops_dot_scaled 标记的 SSA linalg.dot
      │
      │ ninetoothed 的布局分析、调度和其他 pass
      ▼
      │ ② emitter 补丁：识别标记，构造 block load
      ▼
tl.dot_scaled(lhs_block, None, "bf16",
              rhs_block, rhs_scale_block, "e2m1",
              acc=accumulator, fast_math=True, rhs_k_pack=True)
      │
      ▼
Triton 编译器 → 平台中间表示/设备代码
```

两个补丁只在 `ninetoothed.make(...)` 构造 kernel 的 `with` 代码块内安装。退出代码块时，
无论编译成功还是抛异常，原函数都会恢复。`RLock` 用来避免多个桥接调用互相覆盖恢复状态。

这是一种 **算子侧的临时编译兼容措施**，不是正式修改 ninetoothed 后端。理解它的最好方式是：
它在不 fork 编译器的前提下，临时教会编译器识别并发射一个新 intrinsic。

## 2. 读懂代码前需要的编译器背景

### 2.1 AI 算子为什么也需要编译器

普通 PyTorch 算子可以调用已经编译好的库函数。ninetoothed 的工作方式不同：开发者先用
Python DSL 描述张量如何分块以及每个 block 如何计算，框架再把这段描述转换为具体后端代码。

一个高度简化的 AI 编译流水线是：

```text
高级算子语义
  例如 grouped MXFP4 GEMM
        │
        ▼
DSL / 源语言
  Python 函数、tile、expand、dot_scaled
        │
        ▼
前端分析
  解析 AST、识别调用、推导类型
        │
        ▼
中间表示（IR）
  tensor、load、for、linalg.dot 等 operation
        │
        ▼
编译 pass
  布局传播、循环/分块处理、特化、调度
        │
        ▼
后端 emitter
  生成 Triton Python / CUDA / TileLang 等目标代码
        │
        ▼
后端编译器
  Triton/LLVM/厂商工具链
        │
        ▼
设备代码
```

兼容桥覆盖的是“前端分析”和“后端 emitter”两个环节。它没有修改 Triton 自身，也没有实现
新的硬件指令。

### 2.2 AST：编译器看到的不是正在运行的 Python 值

AST 是 Abstract Syntax Tree，抽象语法树。比如：

```python
x = ntl.dot_scaled(a, None, "bf16", b, scale, "e2m1", acc, True, True, True)
```

解析后大致包含 `Assign`、`Call`、`Attribute`、`Name`、`Constant` 等节点。编译器可以从
`Call` 节点取出函数名 `dot_scaled` 和参数表达式，而不需要把它当普通 Python 函数立即执行。

ninetoothed 的 `_ApplicationSSABuilder._lower_call` 就处在这一层。它依次询问 memory、
reduction、linear algebra 和 elementwise handler：“你是否认识这个函数？”认识的 handler
把 AST call 转成 IR operation；所有 handler 都不认识时才报 lowering error。

建议先阅读 Python 官方的
[`ast` 文档](https://docs.python.org/3/library/ast.html)，重点理解 `Call`、`Attribute`、
`Name` 和 `Constant`，不必一开始学习完整 Python grammar。

### 2.3 IR、operation、operand、result 和 attribute

IR（Intermediate Representation，中间表示）是源代码与目标代码之间的结构化语言。直接在
字符串上做替换很难分析类型、shape、依赖和控制流；IR 把这些信息放进明确的数据结构。

本版本 ninetoothed 的 SSA IR 核心对象可以概括为：

```python
Type(kind, shape, dtype, attrs)
Value(name, type)
Operation(opcode, operands, results, attrs, regions)
Block(args, operations)
Program(inputs, outputs, blocks, metadata)
```

以一个普通 dot 为例，可以抽象成：

```text
%result = linalg.dot(%lhs, %rhs) : tensor<...>
```

- `linalg.dot` 是 opcode，表示操作的种类；
- `%lhs`、`%rhs` 是 operands，即输入值；
- `%result` 是 result；
- type 保存 tensor/scalar、shape 和 dtype；
- attrs 保存不适合表达成普通运行时 operand 的编译期信息；
- region 可容纳循环或条件分支内部的 block。

这里的 SSA 是 Static Single Assignment。直观上，每个 `%临时值` 只定义一次。后续优化可以
通过“谁定义了这个值、谁使用它”建立清晰的数据流，而不必猜测可变变量在某一行的状态。
循环中的 accumulator 通常由循环 region/block argument 表达，而不是反复覆盖同一个 SSA 值。

本项目的 IR 在设计思想上接近 MLIR，但不要把“ninetoothed SSA”误认为完整 MLIR。学习概念时
可以参考 [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) 和
[MLIR Rationale](https://mlir.llvm.org/docs/Rationale/Rationale/)，实际字段和 pass 仍以
ninetoothed 源码为准。

### 2.4 lowering、pass 和 emitter 分别是什么

这三个词经常混用，但职责不同：

| 术语 | 本文中的含义 | 本桥中的例子 |
| --- | --- | --- |
| lowering | 把一种较高层表示转换成更低层、约束更明确的表示 | AST `dot_scaled` → SSA operation |
| compiler pass | 遍历并分析或改写 IR | 布局/特化/调度，同时保留自定义 attr |
| emitter/codegen | 把 IR 翻译为目标语言源码或目标 IR | SSA operation → `tl.dot_scaled(...)` |

“lowering”不一定一次完成。真实编译器常有多级 IR：框架图 IR、linalg IR、GPU IR、LLVM IR、
机器指令。每一级逐步丢掉高层抽象，并加入更具体的设备信息。

### 2.5 block program 与逐元素 program

GPU kernel 通常不是让一个 program instance 只算一个标量。矩阵乘更适合让一个 instance
负责一个二维 tile：一次生成二维 `tl.arange` 坐标，加载 A/B block，再调用矩阵 intrinsic。

```text
一个 program instance

       K block
      ┌────────┐
M     │ A tile │
block └────────┘
          ×
      ┌──────────── N block
      │ B tile
      └────────────
          ↓
      M×N accumulator tile
```

ninetoothed emitter 用 `block_program` / `native_block_program` 标记当前是否在这种 block
语义下生成代码。`tl.dot_scaled` 的 lhs、rhs 和 scale 都需要符合约定的二维 block shape，
因此兼容桥拒绝在标量路径中发射它。

Triton 的
[矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
适合建立“一个 program 计算一个 tile、K 维在内部循环”的直觉。

## 3. `dot_scaled` 与 MXFP4 背景

### 3.1 为什么不能把它当普通函数调用

[`tl.dot_scaled`](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html)
不是普通 Python 数值函数。它是 Triton 编译器认识的语言 intrinsic，输入是编译期可分析的
block tensor，后续会进入 Triton IR 的 `tt.dot_scaled` operation，再由平台后端选择硬件指令
或软件模拟路径。

它表达的概念可以简化为：

\[
D = \operatorname{matmul}(\operatorname{scale}(A, S_A),
                           \operatorname{scale}(B, S_B)) + C.
\]

对应的 Triton dialect operation 可参考
[`tt.dot_scaled`](https://triton-lang.org/main/dialects/TritonOps.html)。

### 3.2 本算子的 W4A16 特化

当前 `scaled_grouped_mm` 路径固定为：

- lhs：BF16 activation，不需要 lhs scale；
- rhs：E2M1 MXFP4 weight，每 byte 打包两个 4-bit 元素；
- rhs scale：E8M0，每 32 个 K 元素共享一个 scale；
- accumulator：FP32；
- output：BF16。

因此实际 intrinsic 是：

```python
tl.dot_scaled(
    lhs,
    None,
    "bf16",
    rhs,
    rhs_scale,
    "e2m1",
    acc=accumulator,
    fast_math=True,
    rhs_k_pack=True,
)
```

`lhs_k_pack` 没有显式写出，使用 Triton 默认值 `True`；对 BF16 lhs 不发生 FP4 byte 解包。
`out_dtype` 也使用默认 FP32，与 FP32 accumulator 一致。最终写入 BF16 output 时再转换。

MXFP4 的格式背景可以阅读 OCP 的
[Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。
想直接看可运行算法，可继续学习 Triton
[Block Scaled Matrix Multiplication 教程](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)。

## 4. 没有兼容桥时发生了什么

### 4.1 前端的 handler 链

ninetoothed 前端先从 `ntl.dot_scaled(...)` 取出叶子名称 `dot_scaled`，把每个位置参数 lower
成 SSA value，然后依次尝试：

```python
handlers = (
    self._lower_memory_call,
    self._lower_reduction_call,
    self._lower_linalg_call,
    self._lower_elementwise_call,
)
```

官方 `_lower_linalg_call` 只特别认识 `dot`、`matmul` 和 transpose。它对 `dot_scaled`
返回 `None`。随后 elementwise handler 发现调用来自语言 namespace，于是生成一个通用的：

```text
call.dot_scaled(...)
```

这一步没有保留“它是二维矩阵 block 运算”的强语义。

### 4.2 通用 Triton call fallback

Triton target 对 `exp`、`sqrt`、`atomic_add`、`block_dot` 等名字有显式映射；未知名字采用：

```python
function = functions.get(name, name)
return f"{function}({', '.join(args)})"
```

所以 `dot_scaled` 仍然是 `dot_scaled`，不会自动变成 `tl.dot_scaled`。实际生成物中的关键行是：

```python
v13 = dot_scaled(v4, v5, v6, v7, v8, v9, accumulator, v10, v11, v12)
```

仅给 `functions` 字典增加 `"dot_scaled": "tl.dot_scaled"` 仍不够稳妥。原因是编译器还需要：

- 把此 operation 计入 `has_dot`，从而选择二维 block program；
- 知道 lhs、rhs、scale 分别应使用哪组 block 坐标；
- 区分需要从 tensor load 的值和已经在寄存器/局部表达式中的 accumulator；
- 保证普通 dot 不被改变；
- 固化当前 W4A16 所需的格式和 pack 参数。

这就是兼容桥要同时改前端和 emitter 的原因。

## 5. 兼容桥总体结构

实现入口是 `_enable_dot_scaled_lowering()`：

```python
_LOWERING_LOCK = threading.RLock()


@contextlib.contextmanager
def _enable_dot_scaled_lowering():
    # 1. 定义 replacement
    # 2. 加锁
    # 3. 保存 original
    # 4. 安装 replacement
    try:
        yield
    finally:
        # 5. 恢复 original
```

调用点只有 kernel 构造阶段：

```python
with _enable_dot_scaled_lowering():
    kernel = _cached_make(ntops.kernels.scaled_grouped_mm.premake, jagged)
```

离开 `with` 后，kernel launch 不再需要 monkeypatch。`_cached_make` 使用 `functools.cache`；同一
`premake + jagged + config` 组合只在首次构造时调用 `ninetoothed.make`。缓存命中时仍会短暂
进入兼容桥上下文，但不会重复完成实际构造。

## 6. 第一部分：前端 lowering 补丁

### 6.1 只拦截目标调用

replacement 的第一条规则是：

```python
if name != "dot_scaled":
    return original_lower(self, name, node, operands, operations)
```

这体现了兼容补丁的重要原则：**最小拦截面**。普通 `dot`、`matmul`、transpose 和未来官方
handler 仍走原实现。补丁不是重新实现整个 linear algebra lowering。

由于函数被赋给 `_ApplicationSSABuilder` 类，Python descriptor 机制会在实例调用时自动传入
`self`，所以 replacement 的签名必须与原方法兼容。

### 6.2 检查十个位置参数

kernel 中的调用提供十个位置参数：

| 序号 | 参数 | 当前固定值/来源 |
| ---: | --- | --- |
| 0 | `lhs` | BF16 activation block |
| 1 | `lhs_scale` | `None` |
| 2 | `lhs_format` | `"bf16"` |
| 3 | `rhs` | packed MXFP4 weight block |
| 4 | `rhs_scale` | E8M0 scale block |
| 5 | `rhs_format` | `"e2m1"` |
| 6 | `acc` | FP32 accumulator block |
| 7 | `fast_math` | `True` |
| 8 | `lhs_k_pack` | `True` |
| 9 | `rhs_k_pack` | `True` |

补丁要求 `len(operands) == 10`，否则抛出 `LoweringError`。这是在 IR 形成前阻止参数错位。

需要注意：当前补丁只检查参数数量，没有逐项验证几个编译期常量的值。如果 application 被改成
FP8 lhs 或 `fast_math=False`，emitter 仍会发射当前硬编码的 W4A16 语义。这是临时桥的已知
限制，也是正式上游实现必须改进的地方。

### 6.3 构造带标签的 SSA operation

核心 lowering 是：

```python
return self._emit(
    operations,
    "linalg.dot",
    operands=(
        operands[0].name,  # lhs
        operands[3].name,  # rhs
        operands[4].name,  # rhs_scale
        operands[6].name,  # accumulator
    ),
    attrs={"ntops_dot_scaled": True},
    result_type=operands[6].type,
)
```

可以把它读成下面这条伪 IR：

```text
%result = linalg.dot(%lhs, %rhs, %rhs_scale, %acc)
          {ntops_dot_scaled = true}
          : type(%acc)
```

每个设计点都有具体目的：

- **复用 `linalg.dot` opcode**：让 ninetoothed 已有的 `has_dot`、block program 和相关 pass
  把它当矩阵运算，而不是普通函数；
- **保留四个运行时 operand**：它们需要真实 load 或引用局部 accumulator；
- **不保留六个固定配置 operand**：当前实现是 W4A16 专用桥，emitter 直接写出其常量；
- **增加 `ntops_dot_scaled` attr**：把特殊 dot 与所有普通 dot 区分开；
- **结果类型复用 accumulator type**：dot 的返回是与 FP32 block accumulator 同形同类的值。

严格地说，给 `linalg.dot` 放四个 operand 并不是通用 linalg.dot 契约，而是 ntops 与兼容
emitter 之间的私有约定。只有带 `ntops_dot_scaled=True` 的 operation 才能这样解释。

### 6.4 为什么标签放在 attr 中

attribute 适合保存编译期语义：它不需要在设备运行时占一个寄存器，也不需要从内存加载。
使用唯一前缀 `ntops_` 可以降低与未来官方属性重名的风险。

该标签还形成了一条跨 pass 的“语义线索”：只要中间 pass 在复制/改写 operation 时保留 attrs，
后端就能识别它。若某个新 pass 丢弃 attrs，兼容桥会失效，因此升级编译器时必须保留生成物
测试，而不能只做 Python import 测试。

## 7. 第二部分：SSA emitter 补丁

### 7.1 普通 dot 继续委托原 emitter

emitter replacement 的第一条规则与前端类似：

```python
if not operation.attrs.get("ntops_dot_scaled"):
    return original_emit(operation, context, coords=coords)
```

因此补丁不会改变普通 BF16/FP16 `tl.dot`、标量展开 dot 或其他后端路径。

这里 patch 的是模块函数 `_emit_linalg_dot`，不是类方法，所以 replacement 的第一个参数是
`operation` 而不是 `self`。

### 7.2 限制在 block program

```python
if not (context.block_program or context.native_block_program):
    raise RuntimeError("dot_scaled requires a block-program lowering")
```

这是一个非常关键的防御检查。`tl.dot_scaled` 期望矩阵 block；如果当前 emitter 正在逐元素
生成标量代码，强行输出 intrinsic 会得到错误 shape、错误 load，或者把编译问题拖到更难理解
的 Triton 错误中。

前端把 operation 标成 `linalg.dot`，正是为了让统一 emitter 的 `has_dot` 分析能够选择 block
program。因此“前端标记”和“后端检查”组成闭环。

### 7.3 解包四个 SSA operand

```python
lhs, rhs, rhs_scale, accumulator = operation.operands
```

这一顺序必须和前端构造 operation 的顺序完全一致。IR 本身没有为四个位置命名；变量名只是
emitter 代码为了可读性建立的解释。修改一侧而忘记另一侧会造成最危险的语义错位。

### 7.4 为三个 tensor 计算 block 坐标

```python
lhs_axes = _value_axes(lhs, context)
rhs_axes = _value_axes(rhs, context)
scale_axes = _value_axes(rhs_scale, context)
```

`_value_axes` 从 value type/layout 中取得当前 block 的逻辑轴。例如它们可以抽象为：

```text
lhs axes       = (BLOCK_M, BLOCK_K)
rhs axes       = (BLOCK_K / 2, BLOCK_N)   # 两个 FP4/byte
rhs_scale axes = (BLOCK_K / 32, BLOCK_N)  # 一个 scale/32 个 K 元素
```

随后 `context.target.block_coords(axes)` 为每个轴构造可广播的 `tl.arange`：

```text
二维 block 坐标 ≈ (tl.arange(0, rows)[:, None],
                  tl.arange(0, cols)[None, :])
```

不同 tensor 必须分别计算 axes。scale 在 K 方向比 rhs 短 16 倍，不能直接复用 rhs 坐标。

### 7.5 把 tensor value 变成真正的 block load

```python
lhs_value = _emit_element(lhs, target.block_coords(lhs_axes), context)
rhs_value = _emit_element(rhs, target.block_coords(rhs_axes), context)
scale_value = _emit_element(rhs_scale, target.block_coords(scale_axes), context)
```

这里的 `_emit_element` 名字容易误导：在 vector/block program 中，传入的是二维坐标张量，
返回的会是带 pointer arithmetic、mask 和 `other` 的整个 `tl.load` 表达式，不只是一个 Python
标量。

mask 和越界填充值来自 ninetoothed arrangement 中的 tensor 描述：

- lhs 越界填 0；
- rhs 越界填 0；
- E8M0 scale 越界填 127，即 scale 1。

这也是为什么不能在 emitter 中只写变量名：IR 的 value 需要结合 layout、program id、tile
坐标和边界 mask，才能变成正确的后端 load。

### 7.6 accumulator 为什么使用 `_emit_value`

```python
accumulator_value = _emit_value(accumulator, context)
```

accumulator 已经是 K 循环中的局部 block value，通常对应 `tl.full(..., 0, tl.float32)` 或上一轮
`tl.dot_scaled` 的结果。它不是全局内存 tensor，所以不应再次构造 pointer 和 `tl.load`。

可以这样记忆：

- A/B/scale 是“从哪里读数据” → `_emit_element(...coords...)`；
- accumulator 是“前面计算出的值” → `_emit_value(...)`。

### 7.7 生成最终 Triton 表达式

emitter 返回一个 Python 源码字符串：

```python
return (
    f'tl.dot_scaled({lhs_value}, None, "bf16", {rhs_value}, '
    f'{scale_value}, "e2m1", acc={accumulator_value}, '
    "fast_math=True, rhs_k_pack=True)"
)
```

生成物的结构简化后是：

```python
acc_next = tl.dot_scaled(
    tl.load(lhs_ptrs, mask=lhs_mask, other=0),
    None,
    "bf16",
    tl.load(rhs_ptrs, mask=rhs_mask, other=0),
    tl.load(scale_ptrs, mask=scale_mask, other=127),
    "e2m1",
    acc=acc,
    fast_math=True,
    rhs_k_pack=True,
)
```

此时 Triton 前端能识别 `tl.dot_scaled`，并继续 lower 到 `tt.dot_scaled`。兼容桥的职责到这里
结束；之后能否在特定设备编译和高效执行，是 Triton 平台后端的职责。

## 8. 第三部分：安全地安装与恢复补丁

### 8.1 为什么需要 context manager

monkeypatch 改的是进程全局 Python 对象：

```python
python_frontend._ApplicationSSABuilder._lower_linalg_call = lower_linalg_call
ssa_emitter._emit_linalg_dot = emit_linalg_dot
```

如果安装后不恢复，后续所有 ninetoothed 编译都会看到 ntops 的私有行为。上下文管理器把副作用
限定到一个词法作用域：

```python
with _enable_dot_scaled_lowering():
    # 只有这里期望使用临时 lowering
    kernel = ninetoothed.make(...)
# 离开后已经恢复
```

Python 的
[`contextlib.contextmanager`](https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager)
会在 `yield` 处运行 `with` body。如果 body 抛异常，异常会回到 generator 的 `yield` 位置，
但 `finally` 仍会执行，所以适合表达“临时替换—使用—恢复”。

### 8.2 为什么 `original_*` 在锁内捕获

关键顺序是：

```python
with _LOWERING_LOCK:
    original_lower = CurrentClass._lower_linalg_call
    original_emit = module._emit_linalg_dot
    install_replacements()
    try:
        yield
    finally:
        restore(original_lower, original_emit)
```

若先在锁外捕获 original，两个线程可能发生：

```text
线程 A：读取 original
线程 A：安装 patch A
线程 B：把 patch A 当成自己的 original
线程 A：恢复真正 original
线程 B：安装 patch B
线程 B：最终却恢复 patch A   ← 全局状态泄漏
```

在锁内读取、安装和恢复，使所有使用这把锁的桥接调用串行发生。replacement 闭包虽然定义在
`original_lower` 赋值之前，但 Python 闭包在调用时解析变量；实际调用发生在赋值之后。

### 8.3 为什么使用 `RLock`

`threading.RLock` 是可重入锁：同一个线程可以多次 acquire，只要对应次数 release。假设未来
一个受桥保护的构造流程内部又进入相同上下文，普通 `Lock` 会让线程等待自己并死锁，`RLock`
则允许嵌套。

Python 官方的
[`RLock` 文档](https://docs.python.org/3/library/threading.html#rlock-objects)
适合补充 acquire/release 计数和可重入所有权的概念。

### 8.4 这是否完全线程安全

不是。准确说，它做到了：

- 使用该 context manager 的多个线程不会互相破坏捕获/恢复顺序；
- 同线程嵌套不会死锁；
- 异常退出会恢复原函数；
- 非 `dot_scaled` 和非标记 dot 委托原实现。

但它不能阻止另一个完全不使用 `_LOWERING_LOCK` 的线程在 patch 生效期间直接调用
`ninetoothed.make`。该线程仍会看到被替换的全局函数。由于 replacement 对其他操作进行委托，
多数编译行为不会改变，但这不等价于隔离。

因此文档称其为“受锁保护、作用域受限的兼容桥”，而不宣称它是严格线程隔离的插件系统。
服务进程若会并发编译多类 kernel，应优先把能力上游化，或把编译放入隔离进程。

### 8.5 `finally` 能处理和不能处理什么

`finally` 可以覆盖正常返回和普通 Python exception。它不能保证在进程被 `SIGKILL`、解释器
崩溃或机器掉电时执行；但此时整个进程状态都会消失，不存在把 patch 留给下一个进程的问题。

如果底层编译器 native extension 发生 segmentation fault，Python `finally` 可能没有机会运行。
这不会污染已经终止的进程，却说明 native crash 不能通过 context manager 恢复或诊断。

## 9. 为什么选择临时兼容桥

### 9.1 优点

- 修改范围只在 ntops，不要求用户 fork 或替换整个 ninetoothed 包；
- 复用 ninetoothed 已有 layout、block program 和 tensor load 生成；
- 能快速验证 `dot_scaled` 的算子表达和最终 Triton source；
- 对普通 dot 路径保持委托；
- 易于在 ninetoothed 正式支持后整体删除。

### 9.2 代价

- 依赖 `_ApplicationSSABuilder`、`_emit_linalg_dot`、`_value_axes`、`_emit_element` 等私有符号；
- 编译器升级可能修改函数签名、IR 契约或 block-program 判定；
- 使用带自定义 attr 的四 operand `linalg.dot`，不属于正式通用 IR 设计；
- W4A16 参数被 hard-code，扩展到 W4A4、FP8 或不同 pack 方式时容易产生分支膨胀；
- 进程全局 monkeypatch 无法实现真正的线程局部隔离；
- 只解决 ninetoothed → Triton 的表达，不保证 Triton 平台后端支持。

### 9.3 为什么没有自动软件回退

兼容桥的目标是生成正确 intrinsic，不是改变算子回退政策。如果 backend 不支持，把 MXFP4
完整反量化为 BF16 再 GEMM 会引入巨大中间张量和不同性能语义。当前实现选择暴露编译/运行
失败，并在测试中对不支持平台跳过端到端用例。

若将来增加回退，应显式区分：

```text
native/emulated dot_scaled path
             vs
dequantize-to-BF16 + grouped GEMM path
```

两条路径需要分别做正确性、内存峰值和延迟测试，不能只保证 API 返回结果。

## 10. 如何验证兼容桥

### 10.1 当前自动化测试

测试不要求真正 launch GPU kernel，而是构造 kernel 并读取 compilation artifact：

```python
with _enable_dot_scaled_lowering():
    kernel = ninetoothed.make(
        *ntops.kernels.scaled_grouped_mm.premake(False),
        max_num_configs=1,
    )

source = "\n".join(
    str(value) for value in kernel._compilation.artifact.sources.values()
)
assert source.count("tl.dot_scaled(") == 1
```

运行方法：

```bash
pytest -q tests/test_scaled_grouped_mm.py::test_scaled_grouped_mm_lowers_to_direct_dot_scaled
```

这个测试证明：

- 前端 attr 成功传到 emitter；
- emitter 走了专用分支；
- 生成物使用正确的 Triton namespace；
- application 的一个 dot 没有被重复展开成多个 intrinsic。

它不证明：

- Triton 后端能在当前设备完成编译；
- 生成 kernel 的数值正确；
- 使用硬件原生 MX 指令；
- 性能优于反量化 baseline。

这些必须通过平台端到端测试和后端代码检查补充。

### 10.2 手工比较桥接前后源码

下面的脚本只打印包含 `dot_scaled(` 的关键行：

```bash
python - <<'PY'
import ninetoothed
import ntops
from ntops.torch.scaled_grouped_mm import _enable_dot_scaled_lowering


def dot_scaled_lines(kernel):
    source = str(
        kernel._compilation.artifact.sources["application.triton.py"]
    )
    return [line.strip() for line in source.splitlines()
            if "dot_scaled(" in line]


plain = ninetoothed.make(
    *ntops.kernels.scaled_grouped_mm.premake(False),
    max_num_configs=1,
)
with _enable_dot_scaled_lowering():
    bridged = ninetoothed.make(
        *ntops.kernels.scaled_grouped_mm.premake(False),
        max_num_configs=1,
    )

print("without bridge:", *dot_scaled_lines(plain), sep="\n")
print("with bridge:", *dot_scaled_lines(bridged), sep="\n")
PY
```

生成的 pointer arithmetic 和 mask 很长。调试时先看以下结构，不必逐字符阅读索引公式：

```text
桥接前：v13 = dot_scaled(v4, ..., accumulator, ...)
桥接后：v13 = tl.dot_scaled(tl.load(A...), None, "bf16",
                            tl.load(B...), tl.load(scale...), "e2m1",
                            acc=accumulator, ...)
```

### 10.3 建议补充的测试

如果继续维护该桥，建议增加：

1. **恢复测试**：在上下文前后分别断言两个目标函数对象与原对象相同；
2. **异常恢复测试**：在 `with` 内主动抛异常，确认 `finally` 恢复；
3. **嵌套测试**：同线程嵌套两层 context，确认最终恢复；
4. **并发测试**：多个桥接线程反复构造，确认没有捕获到临时 replacement；
5. **普通 dot 回归**：启用桥时生成普通 `ntl.dot`，源码不应改变；
6. **常量验证测试**：若增加 format/flag 检查，覆盖每个错误值；
7. **版本兼容测试**：在支持范围内的多个 ninetoothed 版本运行结构测试；
8. **端到端测试**：在支持 `dot_scaled` 的设备上比较独立 MXFP4 解码参考。

## 11. 常见故障与定位顺序

### 11.1 生成物仍然是裸 `dot_scaled`

按顺序检查：

1. `ninetoothed.make` 是否确实位于 `_enable_dot_scaled_lowering()` 的 `with` 内；
2. 是否导入了预期的 ntops/ninetoothed 环境；
3. application 的调用叶子名是否仍为 `dot_scaled`；
4. 是否存在 compiler cache，返回了桥接前构造的 artifact；
5. ninetoothed 是否修改了 `_lower_linalg_call` 的签名或 handler 顺序；
6. 自定义 attr 是否在某个 pass 中丢失。

### 11.2 报 `dot_scaled requires a block-program lowering`

说明 emitter 看见了 attr，但调度没有选择 block program。重点检查：

- operation 是否仍使用 `linalg.dot` opcode；
- output arrangement 是否保留二维 tile；
- value axes 是否为二维；
- ninetoothed 的 `has_dot` / `vector_block_program` 判定是否发生版本变化；
- 某个 pass 是否把 dot 分解成逐元素循环。

不要简单删除该检查。删除后可能得到更隐蔽的 shape 或错误代码生成问题。

### 11.3 Triton 编译报 shape/scale layout 错误

检查三组 tile：

```text
A:       (BLOCK_M, BLOCK_K)
B packed:(BLOCK_K / 2, BLOCK_N)
B scale: (BLOCK_K / 32, BLOCK_N)
```

同时检查 nibble 顺序、`rhs_k_pack=True`、K 是否为 32 的倍数，以及 scale 是否保持 `(K/32,N)`
而没有做不符合 Triton API 的转置。

### 11.4 Python 结构测试通过，但设备编译失败或崩溃

这说明桥已经完成自己的职责，问题进入 Triton 或平台 backend。继续收集：

- PyTorch、Triton、驱动和设备版本；
- 生成的 `application.triton.py`；
- Triton IR/LLVM IR/平台 IR（如果工具链支持 dump）；
- 最小 shape 和完整 native stack trace；
- 官方 Triton block-scaled matmul 教程能否在同一环境运行。

不要把“生成了 `tl.dot_scaled`”等价成“平台支持 `dot_scaled`”。

### 11.5 普通 ninetoothed 编译行为被影响

检查是否存在：

- `finally` 没有执行的 native crash；
- 其他代码直接修改同一私有函数；
- 未使用同一锁的并发编译；
- 嵌套 context 恢复顺序错误；
- 新版本编译器把目标函数移动到其他模块，而旧引用仍被 patch。

最直接的诊断是记录上下文前、上下文内、上下文后两个函数对象的 `id()`。

## 12. 如何把兼容桥演进为正式编译能力

长期方案不应继续扩大 monkeypatch，而应在 ninetoothed 中增加正式 operation。推荐路径如下：

### 12.1 定义独立 IR operation

优先定义类似：

```text
%result = linalg.dot_scaled(%lhs, %lhs_scale?, %rhs, %rhs_scale?, %acc)
          {lhs_format="bf16", rhs_format="e2m1",
           fast_math=true, lhs_k_pack=true, rhs_k_pack=true}
```

不要长期复用带四 operand 的 `linalg.dot`。独立 opcode 能让 verifier、pass 和所有 backend
明确知道语义。

### 12.2 前端完整验证静态参数

正式 lowering 应验证：

- format 是否属于支持集合；
- scale 为 `None` 的条件；
- pack 方向；
- lhs/rhs/acc rank 和 dtype；
- K 维关系、FP4 packed 维关系和 scale group size；
- result type 与 accumulator/output policy。

格式字符串和布尔 flag 应进入 attrs，而不是在 emitter hard-code。

### 12.3 让 pass 显式认识新 operation

需要逐个审计依赖 `linalg.dot` / `linalg.matmul` 的代码：

- dot 检测和 block-program 选择；
- layout propagation；
- shape/type inference；
- decomposition；
- specialization；
- scheduling；
- operation cloning 时的 attrs/regions 保留。

不能只加 frontend 和 Triton emitter，因为中间 pass 可能把新 op 当未知操作拒绝或错误改写。

### 12.4 后端 capability 与 fallback

正式能力应让 backend 明确回答：

```text
supports_dot_scaled(format pair, scale kind, pack layout, target architecture)?
```

根据 capability 选择：

- 原生 `dot_scaled`；
- Triton 官方软件模拟；
- 显式反量化回退；
- 编译期明确报错。

平台特化应停留在 capability、layout conversion 和 target lowering 层，不应污染 PyTorch 公共
接口或通用 grouped-MoE 语义。

### 12.5 建立多层测试金字塔

```text
                 设备性能/ISA 检查
               端到端数值正确性
            Triton source / IR golden
         SSA verifier 与 pass 单元测试
      Python API、shape、dtype、错误测试
```

底层测试数量多、执行快；越靠上越依赖硬件、数量更少但证据更强。只测其中一层都不足以证明
完整编译能力。

## 13. 推荐学习路线与资料

### 第一阶段：看懂 Python DSL 如何被“读成代码”

1. [Python `ast` 文档](https://docs.python.org/3/library/ast.html)：自己用 `ast.dump` 查看一个
   `dot_scaled` 调用；
2. [LLVM “My First Language Frontend” 教程](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)：
   理解 lexer/parser、AST、IR generation 的分层；
3. 在 ninetoothed 源码中跟读 `_lower_call` → `_lower_linalg_call` → `_emit`。

练习：给一个虚构的 `ntl.square(x)` 调用画出 AST，并设计一条 `math.square` 伪 IR，但先不
修改项目代码。

### 第二阶段：建立 SSA 和多级 IR 直觉

1. [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)：重点看 operation、value、
   block、region、attribute；
2. [MLIR Rationale](https://mlir.llvm.org/docs/Rationale/Rationale/)：理解为什么编译器需要多级
   抽象，而不是从 Python 直接拼机器码；
3. 阅读 ninetoothed `ir/ssa.py` 中 `Type`、`Value`、`Operation`、`Block`、`Program`。

练习：手写三行伪 SSA 表达 `acc = zeros; acc2 = dot(a,b,acc); store(acc2,out)`，标出每个
value 的唯一定义点。

### 第三阶段：学习 Triton 的 block 编程模型

1. [Triton 矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：
   理解 program id、tile、K 循环、mask 和 accumulator；
2. [`triton.language` API](https://triton-lang.org/main/python-api/triton.language.html)：熟悉
   `arange`、`load`、`store`、`dot`；
3. 对比本项目生成物中的 `tl.load` pointer arithmetic 与教程中的手写 pointer。

练习：只看 shape，不看长索引公式，验证 A/B/scale 三个 block 是否满足 `dot_scaled` 的 API。

### 第四阶段：学习 microscaling

1. [OCP MX 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：
   理解共享 scale、E2M1 和 block size；
2. [`tl.dot_scaled` API](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html)：
   理解 format、scale shape 和 pack 参数；
3. [Triton block-scaled matmul 教程](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)：
   理解逻辑 scale layout 与硬件友好 layout 的区别；
4. [`tt.dot_scaled` dialect 文档](https://triton-lang.org/main/dialects/TritonOps.html)：观察
   Python intrinsic 如何进入更低层 IR。

练习：解释为什么 `(K/2,N)` 的 packed E2M1 权重对应 `(K/32,N)` 的 scale，而不是
`(K/64,N)`。

### 第五阶段：理解临时 patch 的工程风险

1. [Python `contextmanager`](https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager)：
   练习在异常时恢复一个全局变量；
2. [Python `RLock`](https://docs.python.org/3/library/threading.html#rlock-objects)：构造同线程嵌套
   acquire 的例子；
3. 为本桥补写异常恢复和嵌套测试；
4. 最后尝试设计不使用 monkeypatch 的正式 `linalg.dot_scaled` operation。

## 14. 术语速查

| 术语 | 简明解释 |
| --- | --- |
| DSL | 为特定问题设计的语言；这里用 Python 描述 tensor layout 和 kernel 计算 |
| AST | 保留代码语法结构的树，例如函数调用及其参数 |
| IR | 编译器内部、比源码更规则的程序表示 |
| SSA | 每个临时值只有一个定义点的 IR 组织方式 |
| operation/opcode | IR 中的一条操作及其类别名 |
| operand/result | operation 的输入值/输出值 |
| attribute | 编译期元数据，不一定成为运行时参数 |
| lowering | 从高层表示变换到更低层表示 |
| pass | 对 IR 执行分析或改写的一次遍历 |
| emitter | 把 IR 生成某种目标源码或目标 IR 的组件 |
| intrinsic | 编译器内建理解、可映射到特殊 IR/指令的操作 |
| block/tile | 一个 program instance 一次处理的张量子块 |
| monkeypatch | 运行时替换现有 Python 对象或函数的实现 |
| context manager | 用 `with` 管理进入/退出及异常清理的协议 |
| `RLock` | 同一线程可重复获取的互斥锁 |
| MXFP4 | 使用 E2M1 4-bit 元素和共享 scale 的 microscaling 格式 |
| E8M0 | 8-bit exponent-only scale 编码 |

## 15. 一句话复盘

这个兼容桥的本质不是“给函数名前面加 `tl.`”，而是先在前端把一个未知 Python 调用恢复成
编译器能够调度的 dot 语义，再在后端利用 layout 信息生成三个正确的 block load 和一个
`tl.dot_scaled`；context manager、`finally` 与 `RLock` 则负责把临时的全局修改控制在尽可能
小的生命周期内。
