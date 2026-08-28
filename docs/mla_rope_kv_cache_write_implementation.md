# MLA RoPE 与压缩 KV Cache 写入融合报告

## 1. 实现范围

正式入口执行 cache-only 融合：

```text
kv_c, k_pe(raw), position
    -> [kv_c | RoPE(k_pe)]
    -> paged compressed KV cache
```

```python
ntops.torch.mla_rope_kv_cache_write(
    kv_c,             # [T, L]
    k_pe,             # [T, R] or [T, 1, R]
    kv_cache,         # [num_blocks, cache_block_size, L + R]
    slot_mapping,     # [T] int64; -1 skips the write
    positions,        # [T] int32/int64
    cos_sin_cache,    # [max_position, R], packed [cos | sin]
    block_size=128,
    num_warps=(1, 2, 4, 8),
    num_stages=(1, 2),
    max_num_configs=8,
)
```

返回 `None`，cache 原地更新。`slot_mapping` 可以短于 source 第一维，以
兼容 CUDA Graph padding。

## 2. 压缩 Cache 语义

`kv_c` 是 MLA 的低秩共享 K/V latent，不是展开后的多头 K/V。以 DeepSeek/
Kimi 常见配置为例：

```text
kv_lora_rank = 512
rope_dim     = 64
cache entry  = 512 + 64 = 576 elements/token
```

因此该算子写入的是 `[compressed kv_c | rotated k_pe]`，而不是 128 heads 的
完整 K/V。这就是题目中“压缩 KV Cache 写入”的具体含义。

vLLM 基础 `concat_and_cache_mla` 接收已经处理好的 `k_pe`；本实现增加
`positions` 和 `cos_sin_cache`，把 RoPE 融合到 scatter 写入之前：

- [vLLM `concat_and_cache_mla`](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/cache_kernels.cu)
- [vLLM fusion design](https://github.com/vllm-project/vllm/blob/main/docs/design/fusions.md)

## 3. Kernel 与优化

实现位于：

- `src/ntops/kernels/mla_rope_kv_cache_write.py`
- `src/ntops/torch/mla_rope_kv_cache_write.py`

launch 域是 `[token, tile_feature]`，tile 宽度至少为完整 cache entry 的下一个
2 的幂。application 通过 `driver.offsets()` 取得 token/feature 坐标，再通过
NineToothed `source[...]` 完成 latent、RoPE table 和 paged cache 的间接访问。
这避免了 kernel 内显式混用 `program_id/arange`、SSA `index` 与 `i64`，可以由
稳定版 legacy frontend 和 DCU SSA frontend 使用同一份实现。latent tile 直接
复制 `kv_c`；RoPE tile 加载共享 `k_pe`、cos 和 sin，FP32 旋转后写入 cache。
cache 地址为：

```text
block_idx    = slot // cache_block_size
block_offset = slot % cache_block_size
```

kernel 支持 `slot=-1`。`kv_c[:T, :1]` 是零分配 driver，也是 SSA 的主输出
根；feature 0 将原始 `kv_c[:, 0]` 等值写回，其他 lane 仅执行 cache 副作用
写入。这个 no-op 输出使 SSA 按 token 而不是按整个 cache capacity 生成 launch
grid，同时避免原 stride=0 `expand` driver 的错误索引。wrapper 对非连续
source/cache 进行连续化；cache 临时张量在 kernel 后回写到原 view。这隔离了
部分 AMD Triton 无法编译的非单位 stride 间接 store specialization，连续的
生产输入不会产生额外复制。

AMD/DCU 使用输出锚定路径：wrapper 将 `kv_c` 和 `k_pe` 视为一个 query head，
复用已经通过 DCU 验证的 `mla_rope_concat_and_cache` 融合 kernel，并丢弃额外的
query 输出。该 kernel 内的 `[kv_c | RoPE(k_pe)]` paged-cache 写入与 cache-only
接口完全相同，但真实输出张量使 SSA/AMD 后端不需要 lower side-effect-only
launch。CUDA 路径仍直接调用 cache-only kernel，不承担这部分额外输出开销。

正式 API 默认搜索全部 8 个组合：

```text
num_warps  = (1, 2, 4, 8)
num_stages = (1, 2)
```

最佳配置按输入 key 缓存，不再保留单独实验入口。

## 4. 正确性

测试覆盖 FP16/BF16/FP32、rank-2/rank-3 `k_pe`、padding slot、CUDA Graph
padding source、非连续输入/cache 和 vLLM 风格 alias：

```bash
pytest -q tests/test_mla_rope_kv_cache_write.py
```

结果：`9 passed`，其中一个测例在 CUDA 上强制执行 DCU 输出锚定路径。

## 5. 分阶段与自动调优性能

实际输入尺寸：

```text
kv_lora_rank=512, rope_dim=64, cache_block_size=16, dtype=BF16
```

`heads=128` 是对应 MLA 模型结构信息；cache writer 本身写共享 latent，不按
head 展开。测试环境为 Iluvatar MR-V100、PyTorch 2.7.1、Triton 3.1.0。
PyTorch 基准是 eager RoPE 加两次 cache slice scatter。性能统一使用
`triton.testing.do_bench(warmup=25, rep=100, return_mode="mean")`。它按
毫秒时间预算自适应计算预热/采样次数，并在每个样本前清空 L2 cache。
自动调优在正式计时前完成，不包含首次编译和候选搜索成本。加速比为
PyTorch 未融合平均延迟除以 NineToothed 平均延迟。

| 场景 | T | PyTorch RoPE mean (us) | PyTorch cache mean (us) | PyTorch 未融合 mean (us) | 最佳 `(warps, stages)` | NineToothed mean (us) | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | 1 | 34.832 | 26.582 | 63.192 | `(4, 1)` | 8.815 | 7.168x |
| concurrent decode | 10 | 48.119 | 32.186 | 80.984 | `(4, 1)` | 9.850 | 8.221x |
| long context | 2048 | 100.371 | 70.648 | 170.537 | `(1, 1)` | 19.117 | 8.921x |

```bash
python benchmarks/bench_mla_rope_kv_cache_write.py
```

完整 entry tile 让每个 token 只需要一个逻辑 tile，长上下文下显著减少了
program 数量；decode 的 1024-lane tile 利用率较低，因此短输入延迟有所增加。
不同输入选择了不同配置，因此部署时保留按 key 自动调优比固定一组 launch
参数更合适。

## 6. 支持边界

支持 FP16、BF16 和 FP32。当前 NineToothed dtype 层未覆盖本算子所需的
FP8/`fp8_ds_mla` cache 量化布局，因此这两条路径暂不包含在实现中。prefill
和 decode 可以共享本 cache writer；两阶段的 attention/full-key 差异属于
上层算子，不改变这里的 `[kv_c | RoPE(k_pe)]` 写入语义。
