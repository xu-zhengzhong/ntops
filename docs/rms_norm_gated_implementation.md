# Gated RMSNorm 融合算子实现报告

## 1. 概述

本文记录 `ntops` 中 Gated RMSNorm 融合算子的设计、实现和验证结果。该算子面向 KDA、门控线性 Attention 以及新型序列模型推理场景，将门控激活、RMSNorm、权重缩放和输出写回融合到一个 NineToothed kernel 中，以减少中间张量和 kernel 调度开销。

实现参考 vLLM 的 `RMSNormGated` 原生 PyTorch 路径，具体逻辑见 [vLLM layernorm.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py#L1120-L1284)。NineToothed 的算子编程方式参考[官方文档](https://ninetoothed.org/)，并复用了仓库已有的 reduction arrangement 模式。

## 2. 对齐的接口

对外提供 PyTorch 风格函数：

```python
ntops.torch.rms_norm_gated(
    input,
    z=None,
    weight=None,
    eps=1e-5,
    group_size=None,
    norm_before_gate=False,
    activation="swish",
)
```

参数含义如下：

| 参数 | 说明 |
| --- | --- |
| `input` | 输入张量，最后一维是 hidden dimension |
| `z` | 可选门控张量；为 `None` 时退化为普通 RMSNorm |
| `weight` | 可选缩放权重，通常形状为 `(hidden_size,)` |
| `eps` | RMSNorm 数值稳定项，默认 `1e-5` |
| `group_size` | 每个 RMSNorm 分组包含的元素数；为 `None` 时归一化整个最后一维 |
| `norm_before_gate` | `True` 表示先归一化再门控，`False` 表示先门控再归一化 |
| `activation` | 支持 `"silu"`、`"swish"` 和 `"sigmoid"` |

输出形状与 `input` 相同，输出 dtype 与 `input.dtype` 相同。输入、门控和权重在 kernel 内部以 FP32 参与计算。

## 3. 数学语义

令门控激活函数为：

```text
act(z) = SiLU(z),     activation 为 "silu" 或 "swish"
act(z) = sigmoid(z),  activation 为 "sigmoid"
```

当 `z` 存在且 `norm_before_gate=False` 时，先计算 `x' = x * act(z)`；当 `z` 不存在，或 `norm_before_gate=True` 时，`x' = x`。随后对每个归一化分组计算：

```text
rms = sqrt(mean(x'^2) + eps)
y = x' / rms * weight
```

当 `z` 存在且 `norm_before_gate=True` 时，最后执行 `y = y * act(z)`。`group_size` 的语义是“每组包含多少个 hidden 元素”。例如 hidden size 为 4096、`group_size=64` 时，会沿最后一维划分为 64 元素一组，并分别计算每组 RMS；hidden size 必须能被 `group_size` 整除。

## 4. NineToothed Kernel 设计

核心实现位于 [`src/ntops/kernels/rms_norm_gated.py`](../src/ntops/kernels/rms_norm_gated.py)。

### 4.1 编译期特化

`premake` 根据 `activation`、`group_size`、`has_gate` 和 `norm_before_gate` 生成专用 kernel：

- activation 在编译期选择 SiLU/Swish 或 Sigmoid 实现；
- `group_size` 选择普通 RMSNorm 或分组 RMSNorm 路径；
- `has_gate` 和 `norm_before_gate` 是 constexpr，编译器可以消除无效分支。

### 4.2 数据布局与归约

`group_size=None` 时，复用 `ntops.kernels.reduction.arrangement` 沿最后一维组织 reduction block。指定 `group_size` 时，将输入、门控、权重和输出展平后按 group size tile，使每个 reduction 对应一个独立分组。普通路径和分组路径都使用 FP32 累加平方和。

kernel 内部先计算 RMS，再完成归一化、权重缩放和可选末端门控。输出写入调用方提供的 tensor，由输出 tensor dtype 完成回转换。

### 4.3 广播和无门控路径

Torch wrapper 将 `weight` 和 `z` 扩展到输入形状，支持 `(hidden_size,)` 权重及可广播门控张量。`weight=None` 时使用全 1 权重；`z=None` 时通过编译期 `has_gate=False` 走无门控路径。

## 5. Torch 封装

封装位于 [`src/ntops/torch/rms_norm_gated.py`](../src/ntops/torch/rms_norm_gated.py)，职责包括：

1. 校验输入维度、activation、`group_size` 和 hidden size 整除关系；
2. 创建默认权重和输出 tensor；
3. 通过 `_cached_make` 缓存按配置特化的 NineToothed kernel；
4. 传入 epsilon 与归一化元素数量并返回输出。

算子已加入 [`ntops.kernels`](../src/ntops/kernels/__init__.py) 和 [`ntops.torch`](../src/ntops/torch/__init__.py) 的公开导出列表。

## 6. PyTorch 参考实现

测试中的参考函数按 vLLM native PyTorch 语义实现，计算顺序如下：

```python
x = input.float()
z = None if z is None else z.float()
weight = weight.float()

if z is not None and not norm_before_gate:
    x = x * act(z)

if group_size is None:
    rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    output = x / rms * weight
else:
    x_group = x.reshape(*x.shape[:-1], -1, group_size)
    rms = torch.sqrt(x_group.square().mean(dim=-1, keepdim=True) + eps)
    output = (x_group / rms).reshape_as(input) * weight

if z is not None and norm_before_gate:
    output = output * act(z)

return output.to(input.dtype)
```

## 7. 功能验证

测试文件为 [`tests/test_rms_norm_gated.py`](../tests/test_rms_norm_gated.py)，覆盖：

- `float32`、`float16`、`bfloat16`；
- 普通 RMSNorm 和 `group_size=4`；
- `norm_before_gate=True/False`；
- `swish` 和 `sigmoid` 门控；
- 有门控、无门控以及默认全 1 权重；
- 非法 activation、非法 group size 和不可整除 hidden size 的参数校验。

执行命令：

```bash
pytest -q tests/test_rms_norm_gated.py --disable-warnings --maxfail=1
ruff check src/ntops/kernels/rms_norm_gated.py \
    src/ntops/torch/rms_norm_gated.py \
    tests/test_rms_norm_gated.py
python -m compileall -q src/ntops
git diff --check
```

结果：

```text
26 passed in 12.30s
ruff: All checks passed
compileall / git diff --check: passed
```

本报告只记录 Gated RMSNorm 新算子的实现和针对性验证，不包含其他算子的全量回归结论。

## 8. 性能基准

### 8.1 真实 workload 来源

性能测试采用 vLLM Qwen3-Next GDN 的实际数据布局，而不是仅使用测试用的小张量：

- Qwen3-Next 配置的 `linear_value_head_dim=128`、`linear_num_value_heads=32`，见 [Qwen3NextConfig](https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/qwen3_next.py#L949-L970)；
- GDN 中的 `RMSNormGated` 使用 `head_v_dim` 作为 hidden size，配置为 `group_size=None`、`norm_before_gate=True`，默认输出门控为 SiLU，见 [qwen_gdn_linear_attn.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L2991-L3016)；
- vLLM 在输出投影前将 `core_attn_out` 和 `z` reshape 为 `(-1, head_v_dim)`，因此每个 value head 对应一行 RMSNorm 输入，见 [GDN output projection](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L3596-L3625)；
- vLLM 的 Qwen3-Next 部署示例使用 tensor parallel size 4，性能示例使用 2048-token random prompt 和最大并发 10，见 [Qwen3-Next recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3-Next.md#benchmarking)。

因此本次按 TP=4 建模，每个 rank 有 `32 / 4 = 8` 个本地 value heads，输入行数为 `num_tokens * 8`：

| 场景 | vLLM token 数 | kernel 输入形状 |
| --- | ---: | ---: |
| 单 token decode | 1 | `(8, 128)` |
| 并发 decode | 10 | `(80, 128)` |
| 2048-token prefill | 2048 | `(16384, 128)` |

### 8.2 测试方法

可复现脚本为 [`benchmarks/bench_rms_norm_gated.py`](../benchmarks/bench_rms_norm_gated.py)。三条路径均使用 `torch.inference_mode()`，并使用 CUDA event 计时：每个场景先 warmup 50 次，再进行 30 轮测量，每轮连续执行 10 次，报告每次调用的 p50/p90 延迟。NineToothed 首次编译发生在 warmup 阶段，不计入结果。

测试配置：

- PyTorch `2.7.1`；
- Triton `3.1.0`（由当前 NineToothed 运行时生成）；
- 设备 `Iluvatar MR-V100`，compute capability `(7, 1)`；
- `input/z` 为 BF16，`weight` 为 FP32；
- `eps=1e-5`、`group_size=None`、`norm_before_gate=True`、`activation="silu"`；
- PyTorch reference 使用 vLLM 同顺序的 eager FP32 计算，不使用 `torch.compile`。

### 8.3 原始实现结果

```text
scenario             ntops p50   ntops p90   PyTorch p50   PyTorch p90   speedup   max abs error
                     (us)        (us)        (us)          (us)
single-token decode  5.395       5.415       73.640        74.417        13.65x     0.000000
10-token decode      7.813       7.850       75.590        76.238         9.67x     0.000000
2048-token prefill   469.879     469.969     344.553       345.398         0.73x     0.003906
```

结果表明，当前融合 kernel 对 decode 的小规模、高调度开销场景收益明显：单 token 和 10-token decode 分别达到 13.65x 和 9.67x 的 p50 加速。对于 2048-token prefill，PyTorch 的大规模向量化归约更有优势，NineToothed kernel 当前为 PyTorch p50 的 0.73 倍；该形状是后续 block 配置和归约并行度优化的重点。所有场景的最大绝对误差均处于 BF16 输出精度范围内。

复现命令：

```bash
python benchmarks/bench_rms_norm_gated.py
```

## 9. 优化实验版本

为定位 prefill 瓶颈，新增了不覆盖原实现的实验算子：

- DSL kernel：[`src/ntops/kernels/rms_norm_gated_optimized.py`](../src/ntops/kernels/rms_norm_gated_optimized.py)
- Torch wrapper：[`src/ntops/torch/rms_norm_gated_optimized.py`](../src/ntops/torch/rms_norm_gated_optimized.py)
- 调用入口：`ntops.torch.rms_norm_gated_optimized(...)`

### 9.1 瓶颈分析

现有实现沿用通用 `reduction_arrangement`。在当前 NineToothed/Triton 生成结果中，所有 autotune 配置均使用 `num_warps=8`，并产生较复杂的通用索引和 mask 表达式。对于 Qwen3-Next GDN 的 `hidden_size=128`，一行只包含 128 个元素，8 warps 会造成过度并行、同步和寄存器/索引开销；而 prefill 的行数很大，单行固定开销会被放大。实测固定原实现 kernel 的 `num_warps=1` 明显优于 2/4/8 warps，证明归约并行度是主要瓶颈。

在 `(16384, 128)` 上的 kernel-only 探索（预分配输出，不含 Torch wrapper 分配）如下，数值为近似 p50：

| `num_warps` | kernel 延迟 |
| ---: | ---: |
| 1 | 85 us |
| 2 | 156 us |
| 4 | 308 us |
| 8 | 610 us |

### 9.2 试验性改动

优化版本针对最后一维为 hidden dimension 的 GDN 热路径做了专用 layout：

1. 用直接的 last-dimension tile 替代通用 permutation/flatten arrangement，减少生成的索引和边界 mask。
2. 将 `has_gate`、activation、`norm_before_gate` 在 `premake` 阶段选择为不同 application 函数，避免运行时分支。
3. RMS 的平方和使用两遍 FP32 reduction：第一遍计算输入（或门控后输入）的平方和，第二遍复用归一化因子完成权重缩放和输出门控。
4. 默认针对 128 hidden elements 使用 `block_size=128, num_warps=1, num_stages=1`。这些参数是实验版本默认值，其他 hidden size 仍需单独调优。

这里需要区分两套实现的调优行为：原始 `rms_norm_gated` 的通用 arrangement 会生成多个 block-size 配置，并由 NineToothed 生成 `triton.autotune`；本实验版本把 block size、warps 和 stages 固定为单一配置，因此当前生成代码不启动 `triton.autotune`，用于隔离并验证 layout/并行度优化本身。benchmark 的 warmup 会排除原始实现的首次编译和 autotune 成本，但优化版本只有编译成本。

优化版本仍可显式开启 launch 参数 autotune，例如 `num_warps=(1, 2, 4)`、`num_stages=(1, 2)`；NineToothed 会为这些组合生成 `triton.Config` 并在首次遇到输入 key 时选择最快配置。当前 benchmark 使用固定单 warp，是为了复现本次 layout 优化的独立收益。

优化版本仍保留 `group_size`、无门控、SiLU/Swish、Sigmoid 以及两种门控顺序，以便和原接口做对照；原 `rms_norm_gated` 实现和 API 行为没有被修改。

### 9.3 优化版本性能

以下结果来自同一设备、同一输入和同一计时方法，同时给出原始 NineToothed、固定配置优化版本、自动调优优化版本和 PyTorch eager reference。自动调优候选为 `num_warps=(1, 2, 4, 8)`、`num_stages=(1, 2)`，`block_size` 固定为 128：

```text
scenario             original p50/p90  fixed opt p50/p90  autotuned p50/p90  PyTorch p50/p90  auto/fixed  auto/PyTorch  max err (auto)
                     (us)              (us)               (us)                (us)
single-token decode  5.365/5.392       6.616/6.698        6.613/6.663         73.179/73.900    1.00x       11.07x         0.000000
10-token decode      7.727/7.790       8.136/8.283        8.115/8.273         75.547/75.800    1.00x        9.31x         0.000000
2048-token prefill   469.842/469.900   79.575/79.888       79.590/79.735        344.300/345.215    1.00x        4.33x         0.003906
```

在本次设备上，autotuner 对三个 shape 分别选择了 `(num_warps, num_stages)=(1,2)、(1,1)、(1,1)`；与固定单 warp 配置的差异约为测量噪声范围，未发现更高的实质加速比。autotuned 版本在 2048-token prefill 上相对原实现取得 5.90x、相对 PyTorch 取得 4.33x 的 p50 加速；decode 的小形状仍由原实现的通用 layout 略占优势。最大误差为 BF16 输出量化范围内的 `0.003906`。完整输出由以下命令生成：

另外对 prefill 形状进行的固定 `num_warps=1、num_stages=1` tile sweep 结果约为：`block_size=32/64/128/256/512` 对应 `129.8/95.6/82.1/85.9/103.8 us`，因此当前 `block_size=128` 也处于候选中的最佳区域。优化版本目前只对 warp/stage 序列提供 NineToothed autotune；如需让 block size 也参与自动调优，需要进一步把专用 layout 改为可参数化的 meta tile。

```bash
python benchmarks/bench_rms_norm_gated.py
```

### 9.4 优化版本验证

独立测试位于 [`tests/test_rms_norm_gated_optimized.py`](../tests/test_rms_norm_gated_optimized.py)，覆盖两种门控顺序、两种 activation、普通/分组 RMSNorm、无门控和参数校验。测试只新增入口，不改变已有算子测试。

执行结果：`10 passed`；与原始算子测试合并执行时为 `36 passed`。

## 10. 使用示例

```python
import torch
import ntops

x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
z = torch.randn_like(x)
weight = torch.ones(4096, device=x.device, dtype=x.dtype)

y = ntops.torch.rms_norm_gated(
    x,
    z,
    weight,
    eps=1e-5,
    group_size=64,
    norm_before_gate=False,
    activation="swish",
)

# 针对 128 hidden-size GDN prefill 的实验优化入口
y_optimized = ntops.torch.rms_norm_gated_optimized(
    x,
    z,
    weight,
    eps=1e-5,
    group_size=64,
    norm_before_gate=False,
    activation="swish",
)
```
