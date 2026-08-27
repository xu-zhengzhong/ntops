# MLA Decode RoPE、Query 拼接与压缩 KV Cache 写入融合报告

## 1. 实现范围

正式入口 `ntops.torch.mla_rope_concat_and_cache` 对齐 vLLM/Kimi decode
epilogue 的未量化语义：

```text
mqa_q = [ql_nope | RoPE(q_pe)]
kv_cache[slot] = [kv_c | RoPE(k_pe)]
```

输入约定：

```text
ql_nope       [B, H, L]
q_pe          [B, H, R]
kv_c          [B, L]
k_pe          [B, R] 或 [B, 1, R]
kv_cache      [num_blocks, cache_block_size, L + R]
slot_mapping  [B] int64
positions     [B] int32/int64
cos_sin_cache [max_position, R]
```

返回 contiguous `mqa_q [B,H,L+R]`，cache 原地更新；`slot=-1` 跳过写入。
`kv_c` 是低秩压缩 latent，不会展开为每个 attention head 的完整 K/V。

参考接口：

- [vLLM Kimi MLA wrapper](https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py)
- [vLLM Kimi MLA CUDA kernel](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu)
- [vLLM fusion design](https://github.com/vllm-project/vllm/blob/main/docs/design/fusions.md)

## 2. Kernel 设计

实现位于：

- `src/ntops/kernels/mla_rope_concat_and_cache.py`
- `src/ntops/torch/mla_rope_concat_and_cache.py`

arrangement 将输出划分为 `[token, head, entry_tile]`。每个 program 完成
query latent 复制、`q_pe` RoPE 和输出拼接。token 级 cache 写入仅由
`head=0` 的 program 执行一次，避免对 128 个 query heads 重复 scatter。

RoPE 采用交错 pair：

```text
y0 = x0 * cos - x1 * sin
y1 = x0 * sin + x1 * cos
```

中间计算为 FP32。tile、latent width 和 RoPE width 向上取 2 的幂，并通过
mask 截断。默认 128-element tile 用于控制寄存器压力。

## 3. 自动调优

正式 API 默认搜索：

```text
num_warps  = (1, 2, 4, 8)
num_stages = (1, 2)
max_num_configs = 8
```

NineToothed 生成 Triton autotuner，最佳配置按 shape、dtype、stride 和静态
维度缓存。不存在第二个实验入口；正式 API 本身就是优化和自动调优实现。

## 4. 正确性

测试覆盖 FP16/BF16/FP32、rank-2/rank-3 `k_pe`、padding slot、非连续输入/
cache、RoPE mask 和 vLLM 风格 alias：

```bash
pytest -q tests/test_mla_rope_concat_and_cache.py
```

结果：`8 passed`。

## 5. 分阶段与自动调优性能

实际 MLA 尺寸：

```text
H=128, L=512, R=64, cache_block_size=16, dtype=BF16
```

测试环境为 Iluvatar MR-V100、PyTorch 2.7.1、Triton 3.1.0。PyTorch 基准
是 eager RoPE、`torch.cat` 和 paged-cache scatter。性能统一使用
`triton.testing.do_bench(warmup=25, rep=100, return_mode="mean")`。它按
毫秒时间预算自适应计算预热/采样次数，并在每个样本前清空 L2 cache。
自动调优在正式计时前完成，不计入首次编译/搜索成本。加速比为 PyTorch
未融合平均延迟除以 NineToothed 平均延迟。

| 场景 | B | PyTorch RoPE mean (us) | PyTorch cache mean (us) | PyTorch 未融合 mean (us) | 最佳 `(warps, stages)` | NineToothed mean (us) | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | 1 | 78.677 | 26.469 | 118.689 | `(1, 2)` | 11.634 | 10.202x |
| concurrent decode | 10 | 134.333 | 32.289 | 186.268 | `(1, 1)` | 23.364 | 7.973x |
| long context | 2048 | 10446.408 | 70.688 | 12692.055 | `(1, 1)` | 4099.188 | 3.096x |

```bash
python benchmarks/bench_mla_rope_concat_and_cache.py
```

分阶段结果表明主要瓶颈是按 128 heads 展开的 query RoPE 和输出写入，cache
scatter 占比较小。长上下文最终受 query output 总写带宽限制。

## 6. 支持边界

当前支持 FP16、BF16、FP32，不包含 FP8 或 `fp8_ds_mla` cache 布局。该算子
是 decode query epilogue；full-key prefill 的 `k_nope | k_pe` 输出属于不同
接口。只需要 RoPE 与压缩 cache 写入时，应调用 cache-only 算子
`mla_rope_kv_cache_write`。
