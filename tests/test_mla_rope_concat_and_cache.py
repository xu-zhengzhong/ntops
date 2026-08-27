import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _tables(max_position, rope_dim, dtype, device):
    # vLLM stores [cos(0:R/2), sin(0:R/2)] in each row.
    return torch.randn(max_position, rope_dim, dtype=dtype, device=device)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("kpe_rank3", (False, True))
def test_mla_rope_concat_and_cache(dtype, kpe_rank3):
    device = "cuda"
    batch, heads, latent, rope = 3, 4, 8, 8
    ql_nope = torch.randn(batch, heads, latent, device=device, dtype=dtype)
    q_pe = torch.randn(batch, heads, rope, device=device, dtype=dtype)
    kv_c = torch.randn(batch, latent, device=device, dtype=dtype)
    k_pe = torch.randn(batch, 1, rope, device=device, dtype=dtype)
    if not kpe_rank3:
        k_pe = k_pe.reshape(batch, rope)
    kv_cache = torch.randn(6, 4, latent + rope, device=device, dtype=dtype)
    reference_cache = kv_cache.clone()
    slot_mapping = torch.tensor((0, -1, 7), device=device, dtype=torch.int64)
    positions = torch.tensor((0, 2, 5), device=device, dtype=torch.int64)
    cos_sin_cache = _tables(16, rope, torch.float32, device)

    output = ntops.torch.mla_rope_concat_and_cache(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        block_size=8,
        num_warps=1,
        num_stages=1,
    )
    reference = ntops.torch.mla_rope_concat_and_cache_reference(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        reference_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
    )

    atol = 2e-3 if dtype != torch.bfloat16 else 2e-2
    assert torch.allclose(output, reference, rtol=atol, atol=atol)
    assert torch.allclose(kv_cache, reference_cache, rtol=atol, atol=atol)


@skip_if_cuda_not_available
def test_mla_rope_autotuned_and_vllm_alias():
    device = "cuda"
    batch, heads, latent, rope = 2, 2, 8, 8
    values = [
        torch.randn(batch, heads, latent, device=device, dtype=torch.float16),
        torch.randn(batch, heads, rope, device=device, dtype=torch.float16),
        torch.randn(batch, latent, device=device, dtype=torch.float16),
        torch.randn(batch, rope, device=device, dtype=torch.float16),
    ]
    cache = torch.zeros(4, 4, latent + rope, device=device, dtype=torch.float16)
    reference_cache = cache.clone()
    slots = torch.tensor((0, 5), device=device, dtype=torch.int64)
    positions = torch.tensor((0, 1), device=device, dtype=torch.int64)
    table = torch.randn(16, rope, device=device, dtype=torch.float32)
    output = ntops.torch.mla_rope_concat_and_cache(
        *values,
        cache,
        slots,
        positions,
        table,
        num_warps=(1, 2, 4, 8),
        num_stages=(1, 2),
        max_num_configs=8,
    )
    reference = ntops.torch.mla_rope_concat_and_cache_reference(
        *values,
        reference_cache,
        slots,
        positions,
        table,
    )
    assert torch.allclose(output, reference, rtol=2e-3, atol=2e-3)
    assert torch.allclose(cache, reference_cache, rtol=2e-3, atol=2e-3)
    assert ntops.torch.fused_mla_decode_q_concat_kv_cache_insert is ntops.torch.mla_rope_concat_and_cache


@skip_if_cuda_not_available
def test_mla_rope_strided_source_and_cache():
    device = "cuda"
    batch, heads, latent, rope = 2, 2, 8, 8
    ql_nope = torch.randn(batch, heads, latent * 2, device=device, dtype=torch.float16)[
        ..., ::2
    ]
    q_pe = torch.randn(batch, heads, rope * 2, device=device, dtype=torch.float16)[
        ..., ::2
    ]
    kv_c = torch.randn(batch, latent * 2, device=device, dtype=torch.float16)[:, ::2]
    k_pe = torch.randn(batch, rope * 2, device=device, dtype=torch.float16)[:, ::2]
    cache = torch.empty(
        4, 4, (latent + rope) * 2, device=device, dtype=torch.float16
    )[..., ::2]
    reference_cache = cache.clone()
    slots = torch.tensor((0, 5), device=device, dtype=torch.int64)
    positions = torch.tensor((0, 1), device=device, dtype=torch.int64)
    table = torch.randn(16, rope, device=device, dtype=torch.float32)

    output = ntops.torch.mla_rope_concat_and_cache(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        cache,
        slots,
        positions,
        table,
        block_size=8,
        num_warps=1,
        num_stages=1,
    )
    reference = ntops.torch.mla_rope_concat_and_cache_reference(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        reference_cache,
        slots,
        positions,
        table,
    )
    assert torch.allclose(output, reference, rtol=2e-3, atol=2e-3)
    assert torch.allclose(cache, reference_cache, rtol=2e-3, atol=2e-3)
