# Gated RMSNorm 融合算子实现报告

## 1. 目标与接口

正式入口为：

```python
ntops.torch.rms_norm_gated(
    input,
    z=None,
    weight=None,
    eps=1e-5,
    group_size=None,
    norm_before_gate=False,
    activation="swish",
    block_size=128,
    num_warps=(1, 2, 4, 8),
    num_stages=(1, 2),
    max_num_configs=8,
)
```

接口覆盖 vLLM `RMSNormGated` 的门控前/后归一化、SiLU/Swish、Sigmoid、
分组 RMSNorm 和无门控路径。计算使用 FP32 累加，输出转换回输入 dtype。

## 2. 数学语义

`norm_before_gate=True`：

```text
y = RMSNorm(x, weight, eps) * activation(z)
```

`norm_before_gate=False`：

```text
u = x * activation(z)
y = RMSNorm(u, weight, eps)
```

其中：

```text
RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
```

## 3. NineToothed 实现与优化

实现位于：

- `src/ntops/kernels/rms_norm_gated.py`
- `src/ntops/torch/rms_norm_gated.py`

正式 kernel 使用针对最后一维归约的专用 layout：行维度保留在 hierarchy
外部，每个 program 只处理一个 normalization group。相比通用 reduction
arrangement，它减少了索引、mask、同步和过多 warp 带来的开销。

`has_gate`、activation、`norm_before_gate` 在 `premake` 阶段选择独立的
application，使 Triton kernel 中不保留运行时分支。归一化使用两遍 FP32
计算：第一遍完成平方和归约，第二遍完成缩放、权重和可选门控。

正式入口默认启动 NineToothed/Triton autotune：

```text
block_size = 128
num_warps  = (1, 2, 4, 8)
num_stages = (1, 2)
候选数      = 8
```

最佳配置按输入 shape、dtype、stride 和静态语义缓存。首次遇到新 key 时会
产生编译和搜索开销，后续调用直接使用缓存配置。

## 4. 正确性验证

测试覆盖 FP32/FP16/BF16、两种门控顺序、SiLU/Swish、Sigmoid、普通/分组
RMSNorm、无门控和参数校验：

```bash
pytest -q tests/test_rms_norm_gated.py
```

与另外两个新增 MLA 算子的定向测试合并执行结果为 `42 passed`。

## 5. 自动调优性能

测试环境：Iluvatar MR-V100、PyTorch 2.7.1、Triton 3.1.0，dtype 为 BF16，
hidden size 为 128。每个 token 对应 8 个本地 value heads，输入规模来自
Qwen3-Next/GDN 推理中的 decode、并发 decode 和 2048-token prefill。

PyTorch 基准由 eager RMSNorm、SiLU 和逐元素乘法组成。性能统一使用当前
Triton 3.1.0 的 `triton.testing.do_bench`：`warmup=25 ms`、`rep=100 ms`、
`return_mode="mean"`。工具先估算单次运行时间，再自动计算预热和正式采样
次数，并在每个正式样本前清空 L2 cache。加速比公式为：

```text
speedup = PyTorch mean latency / NineToothed mean latency
```

编译和自动调优在 `do_bench` 前完成，因此结果不包含首次搜索成本。

| 场景 | tokens | rows | 最佳 `(warps, stages)` | NineToothed mean (us) | PyTorch mean (us) | 加速比 | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | 1 | 8 | `(4, 1)` | 7.920 | 42.574 | 5.376x | 0.000000 |
| concurrent decode | 10 | 80 | `(1, 2)` | 8.797 | 53.346 | 6.064x | 0.000000 |
| prefill 2048 | 2048 | 16384 | `(1, 1)` | 80.433 | 350.597 | 4.359x | 0.007812 |

运行命令：

```bash
python benchmarks/bench_rms_norm_gated.py
```

结果表明最佳配置与行数有关：短 decode 的单行工作量和大规模 prefill 的
并行占用不同，不能用一个固定 `(warps, stages)` 代表所有实际输入。
