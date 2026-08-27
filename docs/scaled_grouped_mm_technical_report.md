# ntops `scaled_grouped_mm`（W4A16 MXFP4）技术报告

> 报告状态：代码与本机测试已复核；海光端到端数值测试与性能、天数智芯全部实验数据待补测  
> ntops 代码版本：`efa2303de8b85019d9d011a5cadce26321ca0a85`  
> 对比基线：`9ae4166`（尚无该算子）  
> 报告日期：2026-08-26

## 摘要

本工作为 ntops 新增了与
[`torch.nn.functional.scaled_grouped_mm`](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_grouped_mm.html)
签名对齐的 W4A16 路径：激活为 BF16，权重为每字节打包两个 E2M1 数值的 MXFP4，
每 32 个 K 方向元素共享一个 E8M0 scale，累加使用 FP32，输出为 BF16。实现同时覆盖
等长 grouped GEMM 和由累计偏移 `offs` 描述的不等长 MoE expert，包括零 token expert。

核心计算使用 Triton `tl.dot_scaled`，将 FP4 解码、block scale 和矩阵乘累加放在同一条
microscaling dot 路径中，不产生完整的反量化权重中间张量。由于本次使用的 ninetoothed
0.26.0 不能把应用层 `dot_scaled` 正确发射为带 `tl.` 命名空间的 Triton intrinsic，ntops
包装层提供了一个受锁保护、仅在 kernel 构造期间生效的兼容桥。

本次修改以 **A 部分（ntops 算子实现）为主**。没有修改 ninetoothed 仓库源码，因此不把
兼容桥申报为完整的 **B 部分（通用编译后端优化）**。它只能证明正确 lowering 的结构收益：
官方路径生成未解析的 `dot_scaled(...)`，兼容桥生成一个 `tl.dot_scaled(...)`；在两类国产
平台完成运行测试前，不据此推导性能收益。

当前本机目标测试结果为 `10 passed, 4 skipped`。4 个被跳过的用例正是 uniform/jagged
两类端到端数值测试，因为当前 HIP 后端不能执行该 MXFP4 `dot_scaled` 路径。因此本文将
海光和天数智芯的缺失数据显式标成“待测”，不使用模拟结果或其他平台数据替代。

## 1. 修改内容与动机

### 1.1 解决的问题

在提交 `9ae4166` 上，ntops 没有 `scaled_grouped_mm`。这使采用 MXFP4 权重的 MoE 模型
无法通过 ntops 表达以下计算：

\[
C_g = A_g \times \operatorname{dequant}(B_g, S_g),
\]

其中 \(g\) 表示 expert/group，\(A_g\) 为 BF16 activation，\(B_g\) 为 packed E2M1
权重，\(S_g\) 为 E8M0 block scale。MoE 路由还会产生每个 expert token 数不相等、甚至
为零的情况，不能只用固定 M 的 batched GEMM 表示。

提交 `efa2303` 增加了下列能力：

- `ntops.torch.scaled_grouped_mm` 公共入口和 `ScalingType`、`SwizzleType` 导出；
- W4A16、`BlockWise1x32` 的 MXFP4 kernel；
- `(G, M, K)` 的 uniform 输入和 `(total_M, K) + offs` 的 jagged 输入；
- `uint8` 原始存储与 PyTorch 原生 `float4_e2m1fn_x2` / `float8_e8m0fnu` dtype；
- 输入、shape、offset 和暂不支持参数的严格校验；
- 独立 MXFP4 解码参考、lowering 结构检查及负向测试。

### 1.2 A/B 分类

| 内容 | 分类 | 代码位置 | 说明 |
| --- | --- | --- | --- |
| MXFP4 grouped GEMM 的 arrangement/application | A | `src/ntops/kernels/scaled_grouped_mm.py` | 新算子本体 |
| Torch 对齐接口、校验、uniform/jagged 分派 | A | `src/ntops/torch/scaled_grouped_mm.py` | 新算子包装层 |
| `dot_scaled` lowering/emitter 兼容桥 | A 中的编译兼容措施 | 同上 | 临时修改运行时对象，未改 ninetoothed 源码 |
| ninetoothed 通用 pass 或平台后端 | B | 无 | 本提交没有此类代码改动 |

这里特意不把 monkeypatch 直接归为 B：它使用了 ninetoothed 私有 Python 接口，但能力只随
本算子生效，没有形成经过上游接口、回归测试和多后端验证的通用 compiler pass。若后续把
`dot_scaled` 操作定义、SSA 属性、校验和 emitter 正式提交到 ninetoothed，才应作为独立 B
部分评估，并与官方后端比较编译成功率和运行性能。

## 2. 方案与实现

### 2.1 接口和支持范围

公共签名保持与 PyTorch API 一致：

```python
scaled_grouped_mm(
    mat_a,
    mat_b,
    scale_a,
    scale_recipe_a,
    scale_b,
    scale_recipe_b,
    swizzle_a=None,
    swizzle_b=None,
    bias=None,
    offs=None,
    output_dtype=torch.bfloat16,
    contraction_dim=(),
    use_fast_accum=False,
)
```

本版本实际支持的是该接口的 W4A16 子集：

| 参数/能力 | 支持条件 |
| --- | --- |
| `mat_a` | contiguous BF16 |
| `mat_b` | contiguous `uint8` 或 `float4_e2m1fn_x2` |
| `scale_a`, `scale_recipe_a` | 必须为 `None` |
| `scale_b` | contiguous `uint8` 或 `float8_e8m0fnu` |
| `scale_recipe_b` | 必须为 `ScalingType.BlockWise1x32` |
| 输出 | 仅 BF16 |
| `offs` | `None`，或设备上的 contiguous `int32` 累计结束位置 |
| bias、swizzle、多级 scale、非空 `contraction_dim` | 暂不支持 |
| `use_fast_accum` | 必须为 `False` |

暂不支持的组合不会静默换算法，而是抛出 `NotImplementedError` 或 `ValueError`。这样既避免
产生数值语义不一致的结果，也为将来扩展保留了原始 PyTorch 参数位置。

### 2.2 数据布局

| 张量 | uniform 逻辑 shape | jagged 逻辑 shape | 存储类型 | 含义 |
| --- | --- | --- | --- | --- |
| `mat_a` | `(G, M, K)` | `(total_M, K)` | BF16 | 激活 |
| `mat_b` | `(G, K/2, N)` | 相同 | packed E2M1/`uint8` | 每字节两个权重 |
| `scale_b` | `(G, K/32, N)` | 相同 | E8M0/`uint8` | 每 32 个 K 元素一个 scale |
| `output` | `(G, M, N)` | `(total_M, N)` | BF16 | 结果 |

E2M1 的第一个元素放在 byte 的低四位，第二个元素放在高四位。E8M0 byte `e` 表示
\(2^{e-127}\)；因此越界 scale 的填充值使用 127，对应乘法单位 1。`mat_a` 和 `mat_b`
越界位置填 0。该布局与 Triton `dot_scaled` 对 packed FP4 和 E8M0 scale 的约定一致，详见
[`triton.language.dot_scaled` 文档](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html)。

当输入使用 PyTorch 原生 FP4/FP8 dtype 时，包装层通过 `view(torch.uint8)` 提供底层存储，
不执行数值转换或复制；这使 kernel 描述保持为 ninetoothed 当前可处理的 byte tensor，同时
保留面向新 PyTorch 版本的调用兼容性。

### 2.3 分块与并行策略

ninetoothed arrangement 使用三个可搜索 block size：

- `BLOCK_SIZE_M`、`BLOCK_SIZE_N` 下界为 16；
- `BLOCK_SIZE_K` 下界为 64；
- 输出 tile 为 `(1, BLOCK_SIZE_M, BLOCK_SIZE_N)`；
- activation tile 为 `(1, BLOCK_SIZE_M, BLOCK_SIZE_K)`；
- packed weight tile 为 `(1, BLOCK_SIZE_K/2, BLOCK_SIZE_N)`；
- scale tile 为 `(1, BLOCK_SIZE_K/32, BLOCK_SIZE_N)`。

group 维 tile 固定为 1，因此不同 expert 和不同 `(M, N)` 输出 tile 可独立调度；每个 tile
内部沿 K block 串行累加。A tile 沿 N 扩展，B/scale tile 沿 M 扩展，仅建立索引映射，不创建
物理复制。累加器为 FP32，循环结束后一次写回 BF16 output。

这种策略优先保证接口和布局的通用表达，由 ninetoothed 搜索具体 block size。当前没有加入
某一国产平台专用的 warp/wave 数、shared memory swizzle 或指令调度参数，因此不能把这些
能力计为平台特化优化。

### 2.4 融合路径

application 的 K 循环直接调用：

```python
accumulator = ntl.dot_scaled(
    a_tile,
    None,
    "bf16",
    packed_b_tile,
    b_scale_tile,
    "e2m1",
    accumulator,
    True,
    True,
    True,
)
```

lowering 后目标是一个 `tl.dot_scaled(..., rhs_k_pack=True)`。与“先将全部 B 反量化到 BF16，
再调用 GEMM”的实现相比，该路径不会落地 `(G, K, N)` 的 BF16 权重中间张量，并把解码、
block scaling 和 dot accumulation 保留在同一 kernel 内。是否最终落到硬件原生 microscaling
指令或 Triton 软件模拟，由具体 Triton 平台后端决定；ntops 不在接口层假设二者等价的性能。

### 2.5 jagged expert 和零 token expert

当 `offs is None` 时执行 uniform 路径。当 `offs` 存在时：

1. 校验 `offs.shape == (G,)`、dtype 为 `int32`、非负且非递减；
2. 校验 `offs[-1] == total_M`；
3. 在开头补 0，把累计结束位置转换为 nested jagged tensor 所需 offsets；
4. 对 activation 和 output 建立 `torch.nested.nested_tensor_from_jagged` 视图；
5. 使用 `jagged_dim=1` 的同一 kernel 描述执行。

相邻 offset 相等表示该 expert 有 0 行，例如 `(4, 4, 11)` 中第二个 expert 为零 token。实现
不需要为零 token expert 增加虚假行，也不改变后续 expert 的权重索引。

### 2.6 ninetoothed lowering 兼容桥

本机使用 ninetoothed 0.26.0（源码提交
`b77f930dc6c8b016e09adf33570d55a7bc8376c1`）。直接调用 `ninetoothed.make` 能完成源码生成，
但 application 中保留的是未限定名称的：

```python
dot_scaled(lhs, None, "bf16", rhs, rhs_scale, "e2m1", ...)
```

这不是可供 Triton JIT 解析的 `tl.dot_scaled`。兼容桥在 kernel 构造期间完成两步处理：

1. Python SSA builder 把十参数 `dot_scaled` 记录为带 `ntops_dot_scaled` 属性的 `linalg.dot`；
2. SSA emitter 识别该属性并发射完整的 `tl.dot_scaled`，其他 `linalg.dot` 仍委托原实现。

为限制私有接口修改的影响，桥接代码具有以下约束：

- 用全局 `threading.RLock` 串行化安装和恢复；
- 在持锁后捕获原函数，避免并发调用捕获到另一个临时 patch；
- 只拦截操作名 `dot_scaled` 和自有 SSA 属性；
- 使用 context manager 的 `finally` 无条件恢复两个原函数；
- kernel 通过 `_cached_make` 缓存，临时 patch 不延伸到每次 kernel launch。

现有测试确认兼容桥的生成物恰好包含一个 `tl.dot_scaled(`。这是相对当前 ninetoothed 的明确
编译结构收益，但尚不是海光或天数智芯上的运行性能收益。

### 2.7 回退策略与设计取舍

当前没有在算子内部加入“完整反量化 + BF16 GEMM”的自动回退，原因是该回退会额外分配大
张量、改变性能语义，并可能掩盖平台后端缺少 `dot_scaled` 支持。当前策略是：

- 接口或布局不支持时尽早、明确报错；
- 平台后端不能编译/执行 `tl.dot_scaled` 时向调用者暴露失败；
- 独立反量化实现仅用于测试参考，不进入生产路径。

若产品要求跨平台“必定可运行”，建议后续增加显式 backend capability check 和由调用者可见
的 fallback 选项，而不是静默回退。回退性能必须与 intrinsic 路径分开统计。

## 3. 实验与效果

### 3.1 代码和软件基线

| 项目 | 版本 |
| --- | --- |
| ntops 优化前 | `9ae4166`，无 `scaled_grouped_mm` |
| ntops 当前实现 | `efa2303de8b85019d9d011a5cadce26321ca0a85` |
| Python | 3.11.9 |
| PyTorch | `2.9.0+das.opt1.dtk2604` |
| Triton | `3.5.1+das.opt1.dtk2604.torch290` |
| ninetoothed | 0.26.0，源码 `b77f930d...` |
| pytest | 9.1.1 |
| OS | Linux 5.15.0-25-generic, x86_64, glibc 2.35 |
| 当前 accelerator | `torch.cuda.get_device_name(0) == "BW"` |
| 当前运行时 | HIP 6.3.26093，`torch.version.cuda is None` |

“优化前”没有同名算子，因此不存在可信的旧 ntops kernel 延迟可供直接计算加速比。报告把
该变化定义为能力新增。后续性能对照应选择相同语义的 PyTorch/vLLM 或平台原生 W4A16
实现，并注明是否包括反量化时间，不能把“不存在”换算成无穷加速。

### 3.2 正确性方法

测试独立生成 4-bit code，按低/高 nibble 打包，并在 PyTorch 中独立解码 E2M1 和 E8M0：

1. 从 E2M1 提取 sign、2-bit exponent 和 1-bit mantissa；
2. 将 E8M0 转为 `2 ** (scale - 127)`；
3. 沿 K 每 32 个元素展开 scale；
4. 得到 BF16 weight 后使用 FP32 `torch.bmm`/`torch.mm` 累加，最后转 BF16；
5. 以 `rtol=0.03, atol=0.03` 对比 ntops 输出。

覆盖范围包括：

- uniform：`G=2, M=17, K=96, N=19`；
- jagged：`G=3, rows=(4, 0, 7), K=96, N=23`；
- 两组路径均参数化测试 raw `uint8` 和 PyTorch 原生 FP4/FP8 dtype；
- 不规则 tile 边界和零 token expert；
- unsupported 参数、dtype、scale shape、offset dtype/单调性/终值；
- 生成源码中 `tl.dot_scaled` 的唯一性。

### 3.3 当前实测结果

复现命令：

```bash
git checkout efa2303de8b85019d9d011a5cadce26321ca0a85
pytest -q tests/test_scaled_grouped_mm.py
```

本机输出：

```text
ssss..........                                                           [100%]
10 passed, 4 skipped in 14.75s
```

| 用例组 | 数量 | 结果 | 结论 |
| --- | ---: | --- | --- |
| 直接 lowering 为一个 `tl.dot_scaled` | 1 | 通过 | 兼容桥结构正确 |
| 不支持参数校验 | 8 | 通过 | 无静默降级 |
| shape 和 offsets 校验 | 1 | 通过 | uniform/jagged 前置条件生效 |
| uniform 数值，raw/native dtype | 2 | 跳过 | 当前 HIP 后端不支持该执行路径 |
| jagged 数值，raw/native dtype | 2 | 跳过 | 当前 HIP 后端不支持该执行路径 |

必须注意：`10 passed` 不能解释成海光端到端正确性通过。当前测试文件在
`torch.version.hip is not None` 时主动跳过 4 个数值用例；本机仅证明 Python 接口校验和
源码 lowering，不证明 kernel 在该设备执行正确。

### 3.4 海光与天数智芯平台结果

下表如实记录截至本报告日期的证据状态：

| 平台 | 软件/硬件信息 | 数值正确性 | 性能 | 状态 |
| --- | --- | --- | --- | --- |
| 海光 | 当前环境：HIP 6.3.26093，设备名 `BW` | 端到端数值用例被跳过；未验证 | 未测 | `tl.dot_scaled` 当前不能在该后端执行 |
| 天数智芯 | 尚未取得执行环境或原始日志 | 未验证 | 未测 | 待补测 |

因此，本版本不能给出“海光和天数智芯均正确”或“两平台均获得加速”的结论。取得平台环境
后应保留完整 stdout、软件包版本、驱动信息和原始计时数据，再替换本节“待测”字段。

### 3.5 双平台待测方案

两平台使用同一随机种子、输入布局和误差标准。建议的最小测试矩阵如下：

| 模式 | G | M/total_M | K | N | 目的 |
| --- | ---: | ---: | ---: | ---: | --- |
| uniform | 2 | M=17 | 96 | 19 | 与现有不规则边界单测一致 |
| jagged | 3 | rows=(4,0,7) | 96 | 23 | 零 token expert |
| uniform | 8 | M=1, 8, 32 | 4096 | 4096 | MoE 小 M 性能 |
| jagged | 8 | total_M=16,64,256 | 4096 | 4096 | 路由不均衡性能 |

性能测量统一采用：

- kernel 构造/编译和稳态执行分开报告；
- 预热至少 20 次，正式测量至少 100 次；
- 每次计时边界执行设备同步；
- 报告中位数和 P90 延迟，不只报告最小值；
- 逻辑 FLOPs 按 `2 * sum(M_g) * K * N` 计算；
- 同时报 tokens/s，并记录输入 dtype、block size 和最终生成后端；
- baseline 与 ntops 必须使用完全相同的 packed weight、scale 和 offsets。

建议至少设置三类 baseline，并分开解释：

1. **正确性基线**：独立解码 MXFP4 后进行 BF16 grouped matmul；
2. **端到端软件基线**：计入解码和 matmul，用于评估融合消除中间张量的收益；
3. **平台原生基线**：若 vLLM/PyTorch 或厂商库提供等价 W4A16 grouped GEMM，则作为主要
   性能对照，并记录其版本与开关。

待填结果表：

| 平台 | shape/mode | baseline | baseline ms | ntops ms | 加速比 | P90 ms | 正确性 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 海光 | 待测 | 待测 | — | — | — | — | — |
| 天数智芯 | 待测 | 待测 | — | — | — | — | — |

### 3.6 对照与消融

目前已有的编译对照如下：

| 配置 | 生成结果 | 独立效果 |
| --- | --- | --- |
| ninetoothed 0.26.0，不启用兼容桥 | `dot_scaled(...)`，无 `tl.` | 无法形成正确 Triton intrinsic |
| 启用 ntops 兼容桥 | 恰好一个 `tl.dot_scaled(...)` | 修复 lowering 结果 |

硬件性能消融尚待两平台执行，需补充以下对照：

| 消融项 | 对照目的 | 海光 | 天数智芯 |
| --- | --- | --- | --- |
| 显式解包/反量化 + BF16 GEMM vs `dot_scaled` | 量化融合收益 | 待测 | 待测 |
| uniform vs jagged（相同总 token） | jagged 调度代价 | 待测 | 待测 |
| raw `uint8` vs 原生 FP4/FP8 view | dtype 适配开销 | 待测 | 待测 |
| 不同 M/N/K block size | 搜索参数贡献 | 待测 | 待测 |

涉及官方 ninetoothed 后端的收益，目前只能报告“生成正确 intrinsic”这一编译能力差异。没有
两平台数据前，不报告执行加速比，也不声称兼容桥优于未来正式后端实现。

## 4. 工程质量与适用边界

### 4.1 测试覆盖和可追溯性

| 结论 | 代码/测试证据 |
| --- | --- |
| 分块、padding 和 FP32 累加 | `src/ntops/kernels/scaled_grouped_mm.py` |
| 公共接口和支持边界 | `src/ntops/torch/scaled_grouped_mm.py` |
| MXFP4 独立参考、uniform/jagged、零 token | `tests/test_scaled_grouped_mm.py` |
| lowering 只生成一个 intrinsic | `test_scaled_grouped_mm_lowers_to_direct_dot_scaled` |
| 修改完整差异 | `git diff 9ae4166..efa2303 -- src/ntops tests/test_scaled_grouped_mm.py` |

除目标测试外，可执行以下静态检查：

```bash
python -m compileall -q src/ntops tests/test_scaled_grouped_mm.py
git diff --check 9ae4166..efa2303
```

### 4.2 接口兼容性

- 新增导出，不修改既有算子签名；
- 若当前 PyTorch 已提供 `F.ScalingType`/`F.SwizzleType`，直接复用官方 enum；否则使用值兼容的
  本地 enum，使较早 PyTorch 版本仍能导入 ntops；
- 原生 packed dtype 不可用时仍可使用 `uint8` 存储；
- kernel cache 分开保存 uniform 和 jagged 版本；
- 所有输入必须在同一 device，且当前要求 contiguous，避免隐式 copy 改变调用成本。

### 4.3 通用能力与平台特化的边界

以下内容属于通用算子层：PyTorch 签名、MXFP4 byte 布局、E8M0 scale 布局、uniform/jagged
语义、shape 校验和 ninetoothed arrangement。以下内容由平台决定：`tl.dot_scaled` 能否编译、
是否使用原生 microscaling 指令、软件模拟策略和最优 block 配置。

当前提交没有根据设备名选择不同 kernel，也没有海光/天数智芯专用代码路径。这降低了接口
分叉，但也意味着尚未解决两平台后端缺失。后续平台特化应放在明确的 capability dispatch
之后，并保持通用参考路径和平台路径具有相同测试向量。

### 4.4 已知限制

- 仅 W4A16，不支持 activation quantization；
- 仅沿 K 的 1x32 E8M0 block scale；
- 不支持 bias、swizzle、多级 scale、非 BF16 输出和 fast accumulation；
- K 必须为正且是 32 的倍数，权重和 scale 必须使用当前固定布局；
- 依赖 `torch.nested.nested_tensor_from_jagged`；
- lowering 兼容桥依赖 ninetoothed 私有符号，升级 ninetoothed 时需要回归检查；
- 当前海光 HIP 环境不能执行该 `dot_scaled` 路径；
- 天数智芯尚未验证；
- 没有生产路径的软件反量化回退；
- 尚无双平台性能数据、自动化 benchmark 或长期性能回归阈值。

### 4.5 第三方来源

接口语义参考 PyTorch `scaled_grouped_mm`；MXFP4 权重、E8M0 scale 和 MoE 用法参考 vLLM 的
[`mxfp4.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/mxfp4.py)
及其
[`mxfp4_blockwise_moe_kernel.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/fp4/mxfp4_blockwise_moe_kernel.cu)。
Triton intrinsic 的参数和 packed layout 参考官方 `dot_scaled` 文档。

本提交没有复制 vLLM 的 CUTLASS/CUDA kernel；实际代码是 ninetoothed arrangement/application、
ntops Torch wrapper 和测试参考实现。项目许可证仍以仓库现有声明为准。外部链接指向持续更新
的上游 `main`，正式归档实验时应同时记录访问日期或锁定上游 commit。

## 5. 完整复现流程

### 5.1 构建与测试

```bash
git clone https://github.com/InfiniTensor/ntops.git
cd ntops
git checkout efa2303de8b85019d9d011a5cadce26321ca0a85
python -m pip install -e '.[testing]'
pytest -q tests/test_scaled_grouped_mm.py
python -m compileall -q src/ntops tests/test_scaled_grouped_mm.py
git diff --check 9ae4166..efa2303
```

### 5.2 环境留档

在每个平台执行并保存输出：

```bash
python - <<'PY'
import importlib.metadata as metadata
import platform
import torch

print("python", platform.python_version())
print("platform", platform.platform())
for package in ("torch", "triton", "ninetoothed", "ntops", "pytest"):
    print(package, metadata.version(package))
print("torch.version.cuda", torch.version.cuda)
print("torch.version.hip", torch.version.hip)
print("accelerator_available", torch.cuda.is_available())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print("device", index, torch.cuda.get_device_name(index))
PY
```

### 5.3 结果归档要求

双平台补测时，至少归档以下材料并在本报告中引用其相对路径：

- 当前 ntops 与 ninetoothed commit；
- `pip freeze` 或等价环境锁定信息；
- 驱动、运行时和设备型号；
- 正确性测试完整 stdout；
- benchmark 命令、随机种子和原始逐次延迟；
- 最终生成的 Triton/设备代码或编译日志；
- baseline 的版本、配置与是否包含反量化时间。

只有这些证据齐全后，才能把第 3.4—3.6 节的“待测”替换为平台正确性和性能结论。
