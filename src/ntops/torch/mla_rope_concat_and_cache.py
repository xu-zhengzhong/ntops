"""MLA decode RoPE and paged latent-cache insertion."""

import torch

import ntops
from ntops.torch.utils import _cached_make


def _validate_inputs(
    ql_nope,
    q_pe,
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
):
    if ql_nope.ndim != 3 or q_pe.ndim != 3:
        raise ValueError("ql_nope and q_pe must have shape [B, H, dim]")
    if kv_c.ndim != 2:
        raise ValueError("kv_c must have shape [B, kv_lora_rank]")
    if k_pe.ndim not in (2, 3):
        raise ValueError("k_pe must have shape [B, rope_dim] or [B, 1, rope_dim]")
    if kv_cache.ndim != 3:
        raise ValueError(
            "kv_cache must have shape [num_blocks, block_size, entry_dim]"
        )
    if slot_mapping.ndim != 1 or positions.ndim != 1:
        raise ValueError("slot_mapping and positions must be one-dimensional")
    if cos_sin_cache.ndim != 2:
        raise ValueError("cos_sin_cache must have shape [max_position, rope_dim]")
    batch, heads, rope_dim = q_pe.shape
    if ql_nope.shape[:2] != (batch, heads):
        raise ValueError("ql_nope and q_pe batch/head dimensions must match")
    if ql_nope.shape[2] != kv_c.shape[1]:
        raise ValueError("ql_nope width must equal kv_c latent width")
    if kv_c.shape[0] != batch or k_pe.shape[0] != batch:
        raise ValueError("all token dimensions must match")
    if k_pe.numel() != batch * rope_dim:
        raise ValueError("k_pe must contain one shared rope vector per token")
    if kv_cache.shape[2] != kv_c.shape[1] + rope_dim:
        raise ValueError("kv_cache entry dimension must equal kv_c width + rope width")
    if kv_cache.shape[0] <= 0 or kv_cache.shape[1] <= 0:
        raise ValueError("kv_cache must have positive block and block-size dimensions")
    if slot_mapping.shape[0] != batch or positions.shape[0] != batch:
        raise ValueError("slot_mapping and positions must have one value per token")
    if cos_sin_cache.shape[1] != rope_dim:
        raise ValueError(
            "cos_sin_cache width must equal rope_dim (packed cos[0:R/2]|sin[0:R/2])"
        )
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError("rope_dim must be a positive even number")
    if kv_c.shape[1] <= 0:
        raise ValueError("kv_c must have a positive latent width")
    if slot_mapping.dtype != torch.int64:
        raise TypeError("slot_mapping must be torch.int64")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must be torch.int32 or torch.int64")
    tensors = (ql_nope, q_pe, kv_c, k_pe, kv_cache, slot_mapping, positions, cos_sin_cache)
    if any(t.device != ql_nope.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(t.dtype != ql_nope.dtype for t in (q_pe, kv_c, k_pe, kv_cache)):
        raise TypeError("MLA inputs and cache must share dtype")
    if cos_sin_cache.dtype not in (ql_nope.dtype, torch.float32):
        raise TypeError("cos_sin_cache must use the input dtype or torch.float32")


def mla_rope_concat_and_cache(
    ql_nope,
    q_pe,
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    *,
    block_size=128,
    num_warps=(1, 2, 4, 8),
    num_stages=(1, 2),
    max_num_configs=8,
):
    """Fuse MLA decode RoPE, query concatenation, and latent cache update.

    This matches vLLM's ``fused_mla_decode_q_concat_kv_cache_insert`` for the
    unquantized (``kv_cache_dtype=auto``) path.  ``k_pe`` may be ``[B, R]`` or
    ``[B, 1, R]`` and is broadcast across heads.  ``kv_cache`` is updated
    in-place; slots with ``-1`` are ignored.  The return value is a new
    contiguous tensor with shape ``[B, H, L + R]``.  By default Triton
    autotunes the launch configuration and caches the winner by input key.
    """
    _validate_inputs(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
    )
    if not ql_nope.is_cuda:
        raise RuntimeError("mla_rope_concat_and_cache requires a CUDA device")
    if positions.dtype == torch.int32:
        positions = positions.to(torch.int64)
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    batch, num_heads, rope_dim = q_pe.shape
    kv_lora_rank = kv_c.shape[1]
    entry_dim = kv_lora_rank + rope_dim
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    tile_size = 1 << (block_size - 1).bit_length()
    if batch == 0:
        return torch.empty(
            (0, num_heads, entry_dim), dtype=ql_nope.dtype, device=ql_nope.device
        )
    output = torch.empty(
        (batch, num_heads, entry_dim), dtype=ql_nope.dtype, device=ql_nope.device
    )
    kernel = _cached_make(
        ntops.kernels.mla_rope_concat_and_cache.premake,
        num_heads,
        kv_lora_rank,
        rope_dim,
        dtype=ql_nope.dtype,
        block_size=tile_size,
        cos_dtype=cos_sin_cache.dtype,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )
    kernel(
        output,
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        num_heads,
        kv_lora_rank,
        rope_dim,
        entry_dim,
        tile_size,
        kv_cache.shape[1],
        1 << (kv_lora_rank - 1).bit_length(),
        1 << (rope_dim - 1).bit_length(),
    )
    return output


def mla_rope_concat_and_cache_reference(
    ql_nope,
    q_pe,
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
):
    """PyTorch reference used by correctness tests and the benchmark."""
    _validate_inputs(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
    )
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    rope_dim = q_pe.shape[-1]
    half = rope_dim // 2
    table = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = table[:, :half].to(torch.float32)
    sin = table[:, half:].to(torch.float32)

    def rotate(x):
        x0 = x[..., 0::2].to(torch.float32)
        x1 = x[..., 1::2].to(torch.float32)
        table_cos = cos.unsqueeze(-2) if x.ndim == 3 else cos
        table_sin = sin.unsqueeze(-2) if x.ndim == 3 else sin
        y = torch.empty_like(x, dtype=torch.float32)
        y[..., 0::2] = x0 * table_cos - x1 * table_sin
        y[..., 1::2] = x0 * table_sin + x1 * table_cos
        return y.to(x.dtype)

    q_pe_rot = rotate(q_pe)
    k_pe_rot = rotate(k_pe)
    output = torch.cat((ql_nope, q_pe_rot), dim=-1)
    cache = kv_cache
    for token_idx, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        block_idx, block_offset = divmod(slot, kv_cache.shape[1])
        cache[block_idx, block_offset, : kv_c.shape[1]] = kv_c[token_idx]
        cache[block_idx, block_offset, kv_c.shape[1] :] = k_pe_rot[token_idx]
    return output


# Name used by vLLM's Kimi/MLA wrapper.  Keeping the alias makes it easy to
# replace the custom op at the call site without changing tensor semantics.
fused_mla_decode_q_concat_kv_cache_insert = mla_rope_concat_and_cache
