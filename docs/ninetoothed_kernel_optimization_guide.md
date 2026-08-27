# NineToothed Kernel 性能优化指南

## 1. 适用范围

本文面向使用 NineToothed DSL 编写 PyTorch GPU 算子的开发者和智能体，目标是
给出一套可以重复执行、可以验证的优化流程。重点不是罗列 Triton 参数，而是
回答四个问题：

1. 如何确认优化的是正确语义和真实输入；
2. 如何通过分阶段测量定位瓶颈；
3. 如何把 Triton 层面的优化思路表达成 NineToothed arrangement/application；
4. 如何公平地执行 autotune、计算加速比并记录最佳配置。

案例来自本仓库的三个融合算子：

- `rms_norm_gated`：短归约、门控和逐元素计算融合；
- `mla_rope_concat_and_cache`：多头 query RoPE/拼接与压缩 cache 写入；
- `mla_rope_kv_cache_write`：cache-only RoPE 与压缩 latent 写入。

NineToothed 使用 arrange-and-apply 范式：arrangement 决定 program 的 launch
拓扑和每个 program 看到的数据层级，application 描述单个 program 的计算。
性能优化首先是 program 映射问题，其次才是 `num_warps`、`num_stages` 等参数。

## 2. 建立可优化的基线

### 2.1 先冻结数学与接口契约

优化前必须写清：

- 输入/输出 shape、dtype、device 和允许的 strides；
- 哪些参数允许广播；
- 是否原地修改输入；
- padding、空输入、非法索引等边界语义；
- 中间计算精度和输出舍入位置；
- 实际服务中的典型输入规模。

PyTorch reference 应保持直观，不应为了速度模仿融合 kernel。它同时承担正确性
基准和未融合性能基准。例如 cache writer 的 reference 应明确分为：

```text
raw k_pe -> PyTorch RoPE -> rotated k_pe
kv_c + rotated k_pe -> paged cache scatter
```

不要把只实现 decode 的算子用 prefill 名称测试，也不要把压缩 latent cache 与
展开后的 full K/V 混为一谈。接口语义不一致时，加速比没有意义。

### 2.2 正确性先覆盖高风险维度

至少覆盖：

- FP16、BF16、FP32；
- 最小 shape、典型 shape、非 2 的幂 shape；
- contiguous 与非 contiguous tensor；
- 广播输入；
- padding/无效 slot；
- 有输出与仅原地副作用的路径；
- 所有 compile-time 语义分支。

归约和 RoPE 建议在 FP32 中间精度下与 reference 对齐，再转换为输出 dtype。
不能只比较平均误差；同时记录最大绝对误差，并使用与 dtype 相符的 `rtol/atol`。

### 2.3 使用真实推理规模

随机的小矩阵只适合快速正确性测试。性能矩阵应来自真实模型路径。例如本仓库
使用：

```text
Gated RMSNorm: hidden=128, local value heads=8, tokens=1/10/2048
MLA: heads=128, kv_lora_rank=512, rope_dim=64, cache block=16
```

至少包含单 token decode、并发 decode 和长上下文/prefill。不同规模经常选择
不同 autotune 配置，不能只测一个 shape 后宣布全局最优。

## 3. 正确理解 Arrangement

### 3.1 外层 shape 是 launch grid

`Tensor.tile()` 会把 tensor 变为层级 tensor：外层元素映射到 program，内层
tensor 是 application 接收到的 tile。所有 arrangement 返回值的外层 shape
必须兼容，否则 program 与输入块无法正确对齐。

设计 arrangement 前先写出：

```text
一个 program 负责什么？
program grid 的每个维度是什么？
program 内部需要哪些连续向量或归约轴？
哪些数据被多个 program 共享？
```

常见映射为：

| 算子类型 | 推荐 program 粒度 |
| --- | --- |
| 行归约/RMSNorm | 一个 program 处理一行或一个 normalization group |
| 普通逐元素 | 一个 program 处理一个连续 tile |
| 多头 query 变换 | 一个 program 处理 `(token, head, feature_tile)` |
| paged cache 写入 | 一个 program 处理 `(token, cache_entry_tile)` |

program 太粗会增加寄存器压力并降低 occupancy；太细会增加 launch/program 数量、
重复加载和跨 program 写入协调成本。

### 3.2 让归约轴成为 program 内层

RMSNorm 的核心是对 hidden dimension 归约。通用 arrangement 往往带来多余的
permute、flatten、索引和 mask。当前优化实现使用专用 last-dimension layout：

```python
def _arrange_last_dim(tensor, block_size):
    non_target_dims = tuple(range(tensor.ndim - 1))
    inner_shape = (1,) * len(non_target_dims) + (block_size,)
    outer_shape = (1,) * len(non_target_dims) + (-1,)
    arranged = tensor.tile(inner_shape)
    arranged = arranged.tile(outer_shape)
    arranged.dtype = arranged.dtype.squeeze(non_target_dims)
    arranged.dtype.dtype = arranged.dtype.dtype.squeeze(non_target_dims)
    return arranged
```

这里的目标不是机械复制代码，而是保证：

- 行维度只决定 program 数量；
- hidden tile 是 application 内的向量；
- 归约不跨 program；
- application 中不需要恢复复杂的原始维度索引。

对于 `group_size`，可以先 flatten，再按 group tile。完整 hidden 归约和 group
归约应分别检查 program 数量与 normalization element count。

### 3.3 共享数据不要按输出维度重复工作

MLA query 有 128 个 heads，但 `kv_c` 和 `k_pe` 是 token 级共享数据。如果每个
head 都写一次 cache，会把一次 scatter 放大为 128 次，并引入写冲突。

优化原则：

```text
query output: 每个 head 都处理
shared cache: 只由 head=0 或单独的 token program 处理
```

谓词不应只包围 store；无关 program 的 source load、RoPE 和地址计算也应尽量
被 mask，避免“最终没写，但已经做完昂贵计算”。

### 3.4 不规则 scatter 使用 source-root 与动态 strides

规则 tile 访问优先交给 NineToothed 映射。`slot_mapping` 驱动的 paged cache
scatter 无法用普通连续 tile 完整表达时，可以在 application 中使用：

```python
slot = ntl.load(slot_mapping.data_ptr() + token_idx)
block_idx = slot // cache_block_size
block_offset = slot % cache_block_size
cache_base = (
    kv_cache.data_ptr()
    + block_idx * kv_cache.stride(0)
    + block_offset * kv_cache.stride(1)
)
```

使用该方式后，开发者承担以下责任：

- 所有维度使用真实 stride，不能假设 contiguous；
- `slot < 0` 时必须 mask 后续 source load 和 cache store；
- feature 尾部必须 mask；
- 测试 sliced/strided cache；
- 避免对同一个 cache 地址产生非确定性多写。

对同一个标量如 `slot` 只 load 一次并复用。重复的动态全局内存 load 可能不会
总被编译器消除。

### 3.5 无输出算子的 driver

NineToothed 根据 arranged tensor 的外层 shape 生成 launch grid。只有原地副作用、
没有返回 tensor 的算子仍需要一个 grid driver。不要为此分配虚假 output；可以
从已有输入创建零分配 view：

```python
num_entry_tiles = ceil_div(entry_dim, tile_size)
driver = kv_c[:num_tokens, :1].expand(num_tokens, num_entry_tiles)
```

arrangement 中保留一个 source-root scalar，application 显式访问 `driver[0]`，
可以使生成 kernel 保持稳定的数据根和 autotune key。本仓库曾遇到未触碰
driver source 时稳态调用重复调优的问题。这属于当前 NineToothed/Triton 版本
相关的实现细节，升级编译器后必须重新验证。

## 4. 从 Triton 反推 NineToothed 优化

### 4.1 先分类瓶颈

| 瓶颈类型 | 典型现象 | 优先优化项 |
| --- | --- | --- |
| launch-bound | 小输入延迟几乎不随元素数变化 | 融合、减少 kernel 数、减少 wrapper 工作 |
| reduction-bound | warp 增加后更慢、同步成本高 | program 粒度、归约轴、`num_warps` |
| register-bound | 大 tile 明显变慢，occupancy 下降 | 缩小 tile、拆分独立输出、减少活跃中间值 |
| memory-bandwidth-bound | 长输入随字节数近似线性增长 | 合并读写、去中间 tensor、连续访问 |
| redundant-work-bound | head/广播维度增加后异常放大 | 共享计算、谓词前移、避免重复 scatter |
| indexing-bound | 小计算但生成 kernel 很复杂 | 专用 arrangement、减少通用 reshape/mask |

不要一开始就扩大 autotune 候选。配置搜索无法修复错误的 program 映射、重复
访存或多余中间 tensor。

### 4.2 Tile 大小

Triton `arange` 长度通常要求 2 的幂，因此常用：

```python
tile_size = 1 << (requested_size - 1).bit_length()
```

有效范围再用 mask 截断。tile 选择需要权衡：

- 大 tile：program 数少、事务可能更连续，但寄存器和无效 lane 增加；
- 小 tile：寄存器压力低，但 program/launch 开销和重复标量加载增加。

MLA cache entry 是 `512 + 64 = 576`。一次把 576 个值保持在同一 program 中会
提高寄存器压力；128-element tile 将其拆为 5 个 entry tiles，在当前设备上更
合适。该结论不能直接推广到其他 latent width 或 GPU。

### 4.3 `num_warps`

`num_warps` 不是越大越快：

- hidden=128 的短 RMS 归约通常不需要 8 warps；
- 太多 warps 会增加同步、调度和寄存器占用；
- 大量独立 rows 已经提供 grid-level parallelism，不需要再过度增加单 program
  并行度；
- 更宽的 tile、更复杂的归约可能受益于更多 warps。

本仓库使用 `(1, 2, 4, 8)` 作为有限候选，而不是固定假设。实测 winner 会随
tokens/rows 变化。

### 4.4 `num_stages`

`num_stages` 控制软件流水相关编译配置。它更可能帮助存在可重叠 load/compute
的 kernel；对很小的逐元素或单次归约 kernel，更多 stages 可能没有收益，甚至
增加资源占用。通常从 `(1, 2)` 开始，不应在没有数据时扩到很大。

### 4.5 编译期专用化

不要把不会在一次调用中变化的语义保留为运行时分支。RMSNorm 将以下组合在
`premake` 阶段选择为不同 application：

```text
has_gate / no_gate
SiLU / Sigmoid
norm_before_gate / norm_after_gate
group / non-group
```

这样每个生成 kernel 只包含当前语义需要的参数、load 和算术。相比在一个
application 中传 constexpr 后再嵌套多个分支，独立 application 也更容易检查
生成代码和定位性能回归。

专用化也有成本：组合过多会增加编译数量和缓存占用。只专用化能删除显著计算、
参数或访存的维度。

### 4.6 数值精度与转换位置

RMS、方差、RoPE 三角乘加等通常使用 FP32 中间值：

```text
load FP16/BF16 -> cast FP32 -> compute/reduce -> cast output dtype -> store
```

不要在循环内部反复进行相同 cast，也不要为了少一个 cast 改变 reference 的
舍入位置。融合后运算次序可能改变，需要用 dtype 合理容差验证。

## 5. 分阶段定位瓶颈

### 5.1 拆分原则

将 PyTorch reference 按数据流拆为可独立计时的 stages，但保持输入已经在 GPU
上，且不要把数据生成计入时间。例如 MLA：

```text
Stage A: q_pe/k_pe RoPE
Stage B: query concat
Stage C: slot -> block/offset + cache scatter
Stage D: 完整未融合路径
Stage E: NineToothed 融合路径
```

阶段时间之和不一定严格等于完整路径，因为 allocator、kernel launch、缓存状态
和 eager 调度不同。stage 数据用于判断量级和瓶颈，不用于替代端到端加速比。

### 5.2 三个案例的结论

#### Gated RMSNorm

瓶颈不是数学公式，而是通用 reduction arrangement、复杂索引和 hidden=128
时过多 warps。优化为 last-dimension 专用 layout、语义分支专用 application、
FP32 两遍归约后，大规模 prefill 的延迟显著下降。

#### MLA Decode 融合

`q_pe` 按 128 heads 展开，query RoPE 和 `[ql_nope | q_pe]` 输出写入远大于共享
cache scatter。优化重点是控制 query tile 的寄存器压力，并保证 cache 只写一次。
长上下文最终主要受 query output 带宽限制。

#### MLA Cache-only 融合

PyTorch 路径的主要浪费是独立 RoPE kernel、中间 rotated `k_pe` tensor，以及
后续 scatter launch。融合后这些往返被移除。大 token 数时，剩余瓶颈主要是
`kv_c` 的线性写带宽；此时继续增加 warps 通常不会突破内存上限。

## 6. Autotune 的正确使用

### 6.1 只调有意义的轴

当前正式 wrapper 使用：

```python
kernel = _cached_make(
    premake,
    ...,
    num_warps=(1, 2, 4, 8),
    num_stages=(1, 2),
    max_num_configs=8,
)
```

这只搜索 warp/stage，不会自动搜索 wrapper 中固定的 `block_size=128`。如果
tile 可能是主瓶颈，需要把 tile 表达为 NineToothed meta-parameter，或使用
`ninetoothed.build` 明确列出 premake/config 组合。

候选设计建议：

1. 先固定配置做小范围 sweep，确认趋势；
2. 删除明显非法或持续劣势的配置；
3. 再开启线上需要的候选集合；
4. 用 `max_num_configs` 控制首次调优成本；
5. 按真实 shape 分别记录 winner。

### 6.2 Shape-specific winner

最佳配置通常是输入 key 的函数，而不是算子的单一常量。当前 MR-V100 实测中，
同一算子在 decode、并发 decode、长上下文会选择不同 `(warps, stages)`。

报告中应写“本次设备和输入 key 上 autotuner 选择的配置”，不要写成跨 GPU、
跨 dtype 的普遍最优。若准备将 winner 固定部署，应重复运行并比较置信区间，
避免把微秒级测量噪声固化为配置决策。

### 6.3 调优成本与稳态成本

至少区分：

- cold start：生成、编译和试跑候选；
- steady state：命中配置缓存后的正常调用。

本仓库性能表只报告 steady state，加速比不包含首次搜索成本。服务中 shape key
频繁变化时，应额外报告 cold-start 次数和总调优时间。

### 6.4 原地算子的安全性

autotuner 会多次执行不同候选。对原地算子必须保证：

- 候选试跑是幂等覆盖；或
- 每次试跑前恢复被修改的输入；或
- 使用编译器支持的 reset/pre-hook。

本仓库 cache writer 对相同 slot 写入相同值，候选试跑是幂等的。累加、随机数、
队列推进、原子计数等副作用不能直接套用该假设。

### 6.5 配置与编译缓存

使用 `_cached_make` 避免每次 Python 调用重新生成 kernel。cache key 中只应包含
真正影响生成代码的静态值。不要把每次创建的新 Python 对象、无意义动态参数或
临时 output 身份引入 key。

需要观察自动调优选择时可设置：

```bash
TRITON_PRINT_AUTOTUNING=1 python benchmarks/bench_rms_norm_gated.py
```

本仓库 benchmark 还从同一 kernel handle 的 Triton autotune cache 中读取 winner。
这是为了报告配置使用的私有调试接口，不应成为生产 API 依赖。

## 7. 统一性能测试方法

仓库使用 `benchmarks/_benchmark.py` 中的共享方法：

```python
def benchmark_mean(function):
    with torch.inference_mode():
        return do_bench(
            function,
            warmup=25,
            rep=100,
            return_mode="mean",
        ) * 1000.0
```

本地 Triton 3.1.0 的 `do_bench` 会：

1. 估算单次执行时间；
2. 按 25 ms/100 ms 时间预算自适应计算 warmup 和 sample 数；
3. 每个正式样本前清空 L2 cache；
4. 使用设备 event 计时；
5. 返回所有正式样本的算术平均值。

统一加速比公式：

```text
speedup = PyTorch reference mean_us / NineToothed mean_us
```

benchmark 必须满足：

- 输入生成、随机数和一次性 table 构造在计时外；
- 两条路径使用相同 dtype、shape、strides 和数学语义；
- 编译和 autotune 在正式计时前触发；
- GPU benchmark 串行运行，避免资源竞争；
- 输出分配是否计时必须对两条路径公平；
- 原地输出在重复调用下保持有效；
- 分 stage 数值只用于分析，最终加速比用完整未融合路径计算。

平均值、p50、p90 都是合法统计量，但不能在同一张加速比表中混用。本仓库统一
使用 `mean`，因此文档、CSV 列名和公式都必须明确写 `mean`。

## 8. 检查生成的 Triton

NineToothed 抽象不会消除检查生成代码的必要性。重点观察：

- program grid 是否符合预期；
- `program_id` 到 token/head/tile 的映射；
- `arange` 宽度与 mask；
- 同一标量是否重复 load；
- 无关 program 是否仍加载共享输入；
- stride/address 表达式是否过度复杂；
- constexpr 分支是否真的被删除；
- autotune config 和 key 是否合理。

推荐工具：

- `ninetoothed.debugging.simulate_arrangement` 检查 arrangement；
- `ninetoothed.visualization.visualize_arrangement` 查看层级映射；
- `TRITON_PRINT_AUTOTUNING=1` 打印 winner；
- profiler/硬件计数器确认带宽、occupancy、寄存器和 launch 开销。

当前 JIT handle 的 `_source` 可定位生成的 Python/Triton 文件，但它是私有字段，
只适合临时调试。升级 NineToothed 后应优先使用当时版本公开的 debugging API。

## 9. 常见无效优化

| 做法 | 为什么无效或有风险 |
| --- | --- |
| 直接增加 `num_warps` | 小归约会增加同步和资源占用 |
| 只扩大 autotune 候选 | 无法修复错误 layout、重复计算或中间 tensor |
| 整个 576-wide entry 用一个 tile | 可能产生高寄存器压力和低 occupancy |
| 只 mask store，不 mask source load | 无关 program 仍完成了大部分工作 |
| 默认 tensor contiguous | sliced view/cache 会读写错误地址 |
| 把编译时间计入某一条路径 | 比较的是缓存状态，不是 kernel 性能 |
| 用 stage 时间之和代替完整路径 | 忽略 allocator、launch 和缓存状态差异 |
| 用一个 shape 的 winner 固定所有 shape | decode/prefill 的最佳并行度不同 |
| 对非幂等原地算子直接 autotune | 候选试跑会改变后续输入和结果 |
| 为无输出算子分配假 output | 增加分配/写回并可能污染 cache key |
| 比较 PyTorch FP32 与 kernel BF16 | 数学和流量不一致，加速比失真 |

## 10. 推荐优化流程

### 阶段 A：正确性

1. 写清接口、数学、dtype 和副作用；
2. 写 PyTorch reference；
3. 实现最简单 arrangement/application；
4. 覆盖边界、strides 和所有语义分支；
5. 在优化前保存正确性测试。

### 阶段 B：建立性能基线

1. 选择真实 decode/prefill shapes；
2. 用统一 `do_bench(mean)` 测完整 PyTorch 与 NineToothed；
3. 单独记录首次编译/autotune，但不混入稳态表；
4. 保存设备、PyTorch、Triton、NineToothed 版本。

### 阶段 C：定位瓶颈

1. 按数据流拆 PyTorch stages；
2. 判断 launch、归约、寄存器、带宽或重复工作；
3. 检查生成 Triton 的 program/grid/load/store；
4. 找到占比最大且可以改变的部分。

### 阶段 D：结构优化

1. 调整 program 粒度与 arrangement；
2. 消除中间 tensor 和重复 launch；
3. 共享数据只计算/写入一次；
4. 将语义分支移到 premake；
5. 降低 tile 活跃值和寄存器压力；
6. 保证动态 strides 和 mask 正确。

### 阶段 E：参数调优

1. 固定小范围 sweep 验证趋势；
2. 选择有限候选集；
3. 开启 shape-keyed autotune；
4. 记录每个输入的 winner 和 mean latency；
5. 用完整 PyTorch mean 计算 speedup。

### 阶段 F：交付检查

- 正式 API 是否使用优化实现；
- 是否残留实验入口或重复 kernel；
- reference 与实际 API 是否仍对齐；
- autotune 是否可能重复触发；
- 原地候选试跑是否安全；
- benchmark 是否排除编译且使用统一统计量；
- 文档是否列出 shape、dtype、设备、winner、延迟、加速比和限制；
- 是否只运行了相关算子的定向测试，并明确测试范围。

## 11. 参考资料

- [NineToothed 基础与 arrange-and-apply](https://ninetoothed.org/basics.html)
- [NineToothed Tensor meta-operations](https://ninetoothed.org/python_api/tensor.html)
- [NineToothed `make`](https://ninetoothed.org/python_api/generated/ninetoothed.make.html)
- [NineToothed `build`](https://ninetoothed.org/python_api/generated/ninetoothed.build.html)
- [NineToothed debugging](https://ninetoothed.org/python_api/debugging.html)
- [NineToothed visualization](https://ninetoothed.org/python_api/visualization.html)
- [Triton benchmark implementation](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py)
- [Gated RMSNorm 实现报告](rms_norm_gated_implementation.md)
- [MLA decode 融合报告](mla_rope_concat_and_cache_implementation.md)
- [MLA cache-only 融合报告](mla_rope_kv_cache_write_implementation.md)
