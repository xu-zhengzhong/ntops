"""NineToothed MLA RoPE + compressed KV-cache write kernel.

The cache entry follows vLLM's MLA layout:

    [compressed kv_c | RoPE(k_pe)]

``kv_c`` is the shared low-rank K/V latent.  It is not expanded into
per-head K/V in this kernel.  ``k_pe`` is the shared rotary key component and
is rotated in-place in the write path.
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(
    driver,
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    entry_dim,
    tile_size,
    num_entry_tiles,
    kv_lora_rank,
    rope_dim,
    cache_block_size,
):
    # ``driver`` is a zero-allocation [tokens, entry_tiles] view used only to
    # make NineToothed launch one program per token/tile.  The actual cache
    # write is performed through the source-root tensors below.
    driver = driver.tile((1, 1))
    # Keep one innermost scalar so application can explicitly touch the
    # source-root pointer.  This prevents Triton autotune from keying on a
    # freshly-created Python view object on every call.
    driver.dtype = driver.dtype.squeeze(0)
    return (
        driver,
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        entry_dim,
        tile_size,
        num_entry_tiles,
        kv_lora_rank,
        rope_dim,
        cache_block_size,
    )


def application(
    driver,
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    entry_dim,
    tile_size,
    num_entry_tiles,
    kv_lora_rank,
    rope_dim,
    cache_block_size,
):
    driver_marker = driver[0]  # noqa: F841
    pid = ntl.program_id(0)
    token_idx = pid // num_entry_tiles
    entry_tile_idx = pid % num_entry_tiles
    feature = entry_tile_idx * tile_size + ntl.arange(0, tile_size)
    feature_valid = feature < entry_dim
    slot = ntl.load(slot_mapping.data_ptr() + token_idx)
    cache_valid = slot >= 0
    block_idx = slot // cache_block_size
    block_offset = slot % cache_block_size
    cache_base = (
        kv_cache.data_ptr()
        + block_idx * kv_cache.stride(0)
        + block_offset * kv_cache.stride(1)
    )

    nope_mask = feature_valid & (feature < kv_lora_rank)
    latent_idx = feature
    latent_values = ntl.load(
        kv_c.data_ptr()
        + token_idx * kv_c.stride(0)
        + latent_idx * kv_c.stride(1),
        mask=cache_valid & nope_mask,
        other=0,
    )
    ntl.store(
        cache_base + latent_idx * kv_cache.stride(2),
        latent_values,
        mask=cache_valid & nope_mask,
    )

    rope_idx = feature - kv_lora_rank
    rope_mask = feature_valid & (feature >= kv_lora_rank) & (rope_idx < rope_dim)
    pair = rope_idx // 2
    even = (rope_idx & 1) == 0
    pair_mask = rope_mask & (pair * 2 + 1 < rope_dim)
    kpe_base = k_pe.data_ptr() + token_idx * k_pe.stride(0)
    kpe0 = ntl.load(
        kpe_base + pair * 2 * k_pe.stride(1),
        mask=cache_valid & pair_mask,
        other=0,
    )
    kpe1 = ntl.load(
        kpe_base + (pair * 2 + 1) * k_pe.stride(1),
        mask=cache_valid & pair_mask,
        other=0,
    )

    position = ntl.load(positions.data_ptr() + token_idx)
    table_base = cos_sin_cache.data_ptr() + position * cos_sin_cache.stride(0)
    table_stride = cos_sin_cache.stride(1)
    cos = ntl.load(table_base + pair * table_stride, mask=pair_mask, other=1)
    sin = ntl.load(
        table_base + (rope_dim // 2 + pair) * table_stride,
        mask=pair_mask,
        other=0,
    )
    rotated = ntl.where(
        even,
        kpe0 * cos - kpe1 * sin,
        kpe0 * sin + kpe1 * cos,
    )
    ntl.store(
        cache_base + (kv_lora_rank + rope_idx) * kv_cache.stride(2),
        rotated.to(kv_c.dtype),
        mask=cache_valid & rope_mask,
    )


def premake(
    kv_lora_rank,
    rope_dim,
    dtype=None,
    block_size=128,
    cache_block_size=16,
    cos_dtype=None,
):
    entry_dim = kv_lora_rank + rope_dim
    num_entry_tiles = (entry_dim + block_size - 1) // block_size
    dynamic = {"constexpr": True, "upper_bound": 2**20}
    tensors = (
        Tensor(
            shape=(None, num_entry_tiles),
            dtype=dtype,
            shape_options=(dynamic, {"constexpr": True}),
        ),
        Tensor(
            shape=(None, kv_lora_rank),
            dtype=dtype,
            shape_options=(dynamic, {"constexpr": True}),
        ),
        Tensor(
            shape=(None, rope_dim),
            dtype=dtype,
            shape_options=(dynamic, {"constexpr": True}),
        ),
        Tensor(
            shape=(None, None, entry_dim),
            dtype=dtype,
            shape_options=(dynamic, dynamic, {"constexpr": True}),
        ),
        Tensor(shape=(None,), dtype=ninetoothed.int64, shape_options=(dynamic,)),
        Tensor(shape=(None,), dtype=ninetoothed.int64, shape_options=(dynamic,)),
        Tensor(
            shape=(None, rope_dim),
            dtype=cos_dtype or ninetoothed.float32,
            shape_options=(dynamic, {"constexpr": True}),
        ),
        Tensor(0, constexpr=True, value=entry_dim),
        Tensor(0, constexpr=True, value=block_size),
        Tensor(0, constexpr=True, value=num_entry_tiles),
        Tensor(0, constexpr=True, value=kv_lora_rank),
        Tensor(0, constexpr=True, value=rope_dim),
        Tensor(0, constexpr=True, value=cache_block_size),
    )
    return arrangement, application, tensors
