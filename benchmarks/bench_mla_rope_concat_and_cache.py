"""Benchmark MLA RoPE + compressed KV-cache insertion.

The dimensions mirror vLLM's DeepSeek/Kimi MLA decode path: 128 query heads,
512 latent channels, 64 rotary channels, and 16-token paged blocks.
"""

import torch
from _benchmark import benchmark_mean, selected_config

import ntops
from ntops.torch.utils import _cached_make

DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_HEADS = 128
KV_LORA_RANK = 512
ROPE_DIM = 64
CACHE_BLOCK_SIZE = 16
TILE_SIZE = 128
AUTOTUNE_WARPS = (1, 2, 4, 8)
AUTOTUNE_STAGES = (1, 2)
MAX_NUM_CONFIGS = 8


def _make_inputs(tokens):
    ql_nope = torch.randn(
        tokens, NUM_HEADS, KV_LORA_RANK, device=DEVICE, dtype=DTYPE
    )
    q_pe = torch.randn(tokens, NUM_HEADS, ROPE_DIM, device=DEVICE, dtype=DTYPE)
    kv_c = torch.randn(tokens, KV_LORA_RANK, device=DEVICE, dtype=DTYPE)
    k_pe = torch.randn(tokens, ROPE_DIM, device=DEVICE, dtype=DTYPE)
    num_blocks = (tokens + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE + 4
    kv_cache = torch.empty(
        num_blocks,
        CACHE_BLOCK_SIZE,
        KV_LORA_RANK + ROPE_DIM,
        device=DEVICE,
        dtype=DTYPE,
    )
    slot_mapping = torch.arange(tokens, device=DEVICE, dtype=torch.int64)
    positions = torch.arange(tokens, device=DEVICE, dtype=torch.int64)
    # Packed vLLM table: [cos(0:R/2), sin(0:R/2)].
    half = ROPE_DIM // 2
    theta = 10000 ** (-2 * torch.arange(half, device=DEVICE) / ROPE_DIM)
    phase = positions.to(torch.float32)[:, None] * theta[None, :]
    cos_sin_cache = torch.cat((phase.cos(), phase.sin()), dim=-1)
    return (
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
    )


def _torch_rope(q_pe, k_pe, positions, cos_sin_cache):
    half = ROPE_DIM // 2
    table = cos_sin_cache.index_select(0, positions)
    cos = table[:, :half].to(torch.float32)
    sin = table[:, half:].to(torch.float32)

    def rotate(x, head_broadcast):
        x0 = x[..., 0::2].to(torch.float32)
        x1 = x[..., 1::2].to(torch.float32)
        c = cos.unsqueeze(-2) if head_broadcast else cos
        s = sin.unsqueeze(-2) if head_broadcast else sin
        out = torch.empty_like(x, dtype=torch.float32)
        out[..., 0::2] = x0 * c - x1 * s
        out[..., 1::2] = x0 * s + x1 * c
        return out.to(x.dtype)

    return rotate(q_pe, True), rotate(k_pe, False)


def _torch_unfused(inputs):
    ql_nope, q_pe, kv_c, k_pe, cache, slots, positions, table = inputs
    q_rot, k_rot = _torch_rope(q_pe, k_pe, positions, table)
    output = torch.cat((ql_nope, q_rot), dim=-1)
    block = slots // CACHE_BLOCK_SIZE
    offset = slots % CACHE_BLOCK_SIZE
    cache[block, offset, :KV_LORA_RANK] = kv_c
    cache[block, offset, KV_LORA_RANK:] = k_rot
    return output


def _torch_rope_only(inputs):
    return _torch_rope(inputs[1], inputs[3], inputs[6], inputs[7])


def _torch_cache_only(inputs, k_rot):
    _, _, kv_c, _, cache, slots, _, _ = inputs
    block = slots // CACHE_BLOCK_SIZE
    offset = slots % CACHE_BLOCK_SIZE
    cache[block, offset, :KV_LORA_RANK] = kv_c
    cache[block, offset, KV_LORA_RANK:] = k_rot


def _kernel_handle(inputs):
    return _cached_make(
        ntops.kernels.mla_rope_concat_and_cache.premake,
        NUM_HEADS,
        KV_LORA_RANK,
        ROPE_DIM,
        dtype=DTYPE,
        block_size=TILE_SIZE,
        cos_dtype=inputs[-1].dtype,
        num_warps=AUTOTUNE_WARPS,
        num_stages=AUTOTUNE_STAGES,
        max_num_configs=MAX_NUM_CONFIGS,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    print(f"torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name()}")
    print(
        "scenario,tokens,heads,latent,rope,torch_rope_mean_us,"
        "torch_cache_mean_us,torch_unfused_mean_us,best_num_warps,"
        "best_num_stages,autotuned_mean_us,"
        "speedup_vs_pytorch"
    )

    for scenario, tokens in (("decode", 1), ("concurrent_decode", 10), ("long_context", 2048)):
        inputs = _make_inputs(tokens)

        def rope_only():
            return _torch_rope_only(inputs)

        # Keep the rotated k_pe in a persistent tensor for cache-only timing.
        rotated_k = _torch_rope_only(inputs)[1]

        def cache_only():
            return _torch_cache_only(inputs, rotated_k)

        def unfused():
            return _torch_unfused(inputs)

        def autotuned():
            return ntops.torch.mla_rope_concat_and_cache(
                *inputs,
                block_size=TILE_SIZE,
                num_warps=AUTOTUNE_WARPS,
                num_stages=AUTOTUNE_STAGES,
                max_num_configs=MAX_NUM_CONFIGS,
            )

        # Trigger compilation/autotuning before collecting steady-state times.
        autotuned()
        torch.cuda.synchronize()
        warps, stages = selected_config(_kernel_handle(inputs))
        rope_us = benchmark_mean(rope_only)
        cache_us = benchmark_mean(cache_only)
        unfused_us = benchmark_mean(unfused)
        auto_us = benchmark_mean(autotuned)
        print(
            f"{scenario},{tokens},{NUM_HEADS},{KV_LORA_RANK},{ROPE_DIM},"
            f"{rope_us:.3f},{cache_us:.3f},{unfused_us:.3f},{warps},{stages},"
            f"{auto_us:.3f},{unfused_us / auto_us:.3f}"
        )


if __name__ == "__main__":
    main()
