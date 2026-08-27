"""NineToothed MLA RoPE + latent KV-cache update kernels.

The layout follows vLLM's decode-side MLA fusion:

* ``ql_nope`` is the absorbed query in ``[B, H, L]``;
* ``q_pe`` is the per-head rotary query in ``[B, H, R]``;
* ``kv_c`` and ``k_pe`` are written as ``[kv_c | RoPE(k_pe)]`` to the
  paged cache at ``slot_mapping``;
* the returned query is ``[ql_nope | RoPE(q_pe)]``.

The RoPE table is vLLM's packed table: each row contains ``R / 2`` cosine
values followed by ``R / 2`` sine values.  Values are interleaved in pairs.
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(
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
    tile_size_value,
    cache_block_size,
    kv_rank_p2,
    rope_dim_p2,
):
    # One program owns one (token, head) row, while the last dimension is a
    # vector tile.  Source tensors stay at their roots so their pointers and
    # strides can be used for the indirect paged-cache write below.
    output = output.tile((1, 1, tile_size_value.value))
    output.dtype = output.dtype.squeeze((0, 1))
    return (
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
        tile_size_value,
        cache_block_size,
        kv_rank_p2,
        rope_dim_p2,
    )


def application(
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
    cache_block_size,
    kv_rank_p2,
    rope_dim_p2,
):
    pid = ntl.program_id(0)
    num_entry_blocks = (entry_dim + tile_size - 1) // tile_size
    token_idx = pid // (num_heads * num_entry_blocks)
    head_idx = (pid // num_entry_blocks) % num_heads
    entry_block_idx = pid % num_entry_blocks
    feature = entry_block_idx * tile_size + ntl.arange(0, tile_size)
    feature_valid = feature < entry_dim
    nope_mask = feature_valid & (feature < kv_lora_rank)
    rope_mask = feature_valid & (feature >= kv_lora_rank)
    rope_feature = feature - kv_lora_rank
    pair = rope_feature // 2
    even = (rope_feature & 1) == 0

    ql_ptr = ql_nope.data_ptr()
    ql_s0 = ql_nope.stride(0)
    ql_s1 = ql_nope.stride(1)
    ql_s2 = ql_nope.stride(2)
    ql_values = ntl.load(
        ql_ptr
        + token_idx * ql_s0
        + head_idx * ql_s1
        + feature * ql_s2,
        mask=nope_mask,
        other=0,
    )

    qpe_ptr = q_pe.data_ptr()
    qpe_s0 = q_pe.stride(0)
    qpe_s1 = q_pe.stride(1)
    qpe_s2 = q_pe.stride(2)
    qpe_base = qpe_ptr + token_idx * qpe_s0 + head_idx * qpe_s1
    qpe_pair_mask = rope_mask & (pair * 2 + 1 < rope_dim)
    qpe0 = ntl.load(qpe_base + pair * 2 * qpe_s2, mask=qpe_pair_mask, other=0)
    qpe1 = ntl.load(
        qpe_base + (pair * 2 + 1) * qpe_s2,
        mask=qpe_pair_mask,
        other=0,
    )

    position = ntl.load(positions.data_ptr() + token_idx)
    table_base = cos_sin_cache.data_ptr() + position * cos_sin_cache.stride(0)
    table_s1 = cos_sin_cache.stride(1)
    cos = ntl.load(table_base + pair * table_s1, mask=qpe_pair_mask, other=1)
    sin = ntl.load(
        table_base + (rope_dim // 2 + pair) * table_s1,
        mask=qpe_pair_mask,
        other=0,
    )
    qrot = ntl.where(even, qpe0 * cos - qpe1 * sin, qpe0 * sin + qpe1 * cos)
    result = ntl.where(nope_mask, ql_values, qrot)
    output = result.to(ql_nope.dtype)  # noqa: F841

    # Only the first head and first entry tile performs the token-level cache
    # insert.  The mask keeps padded (-1) slots and all other heads inactive.
    slot = ntl.load(slot_mapping.data_ptr() + token_idx)
    cache_valid = (slot >= 0) & (head_idx == 0) & (entry_block_idx == 0)
    block_idx = slot // cache_block_size
    block_offset = slot % cache_block_size
    cache_base = (
        kv_cache.data_ptr()
        + block_idx * kv_cache.stride(0)
        + block_offset * kv_cache.stride(1)
    )
    cache_rank = ntl.arange(0, kv_rank_p2)
    rank_mask = cache_rank < kv_lora_rank
    kv_values = ntl.load(
        kv_c.data_ptr()
        + token_idx * kv_c.stride(0)
        + cache_rank * kv_c.stride(1),
        mask=cache_valid & rank_mask,
        other=0,
    )
    ntl.store(
        cache_base + cache_rank * kv_cache.stride(2),
        kv_values,
        mask=cache_valid & rank_mask,
    )

    cache_rope = ntl.arange(0, rope_dim_p2)
    cache_rope_mask = cache_rope < rope_dim
    cache_pair = cache_rope // 2
    cache_even = (cache_rope & 1) == 0
    kpe_base = k_pe.data_ptr() + token_idx * k_pe.stride(0)
    kpe_pair_mask = cache_rope_mask & (cache_pair * 2 + 1 < rope_dim)
    kpe0 = ntl.load(
        kpe_base + cache_pair * 2 * k_pe.stride(1),
        mask=cache_valid & kpe_pair_mask,
        other=0,
    )
    kpe1 = ntl.load(
        kpe_base + (cache_pair * 2 + 1) * k_pe.stride(1),
        mask=cache_valid & kpe_pair_mask,
        other=0,
    )
    kcos = ntl.load(
        table_base + cache_pair * table_s1,
        mask=cache_valid & kpe_pair_mask,
        other=1,
    )
    ksin = ntl.load(
        table_base + (rope_dim // 2 + cache_pair) * table_s1,
        mask=cache_valid & kpe_pair_mask,
        other=0,
    )
    krot = ntl.where(
        cache_even,
        kpe0 * kcos - kpe1 * ksin,
        kpe0 * ksin + kpe1 * kcos,
    )
    ntl.store(
        cache_base + (kv_lora_rank + cache_rope) * kv_cache.stride(2),
        krot.to(kv_c.dtype),
        mask=cache_valid & cache_rope_mask,
    )


def premake(
    num_heads,
    kv_lora_rank,
    rope_dim,
    dtype=None,
    block_size=None,
    cos_dtype=None,
):
    if block_size is None:
        block_size = 128
    entry_dim = kv_lora_rank + rope_dim
    dynamic = {"constexpr": True, "upper_bound": 2**20}
    b_shape = (dynamic, {}, {})
    br_shape = (dynamic, {})
    cache_shape = (dynamic, dynamic, {})
    tensors = (
        Tensor(shape=(None, num_heads, entry_dim), dtype=dtype, shape_options=b_shape),
        Tensor(shape=(None, num_heads, kv_lora_rank), dtype=dtype, shape_options=b_shape),
        Tensor(shape=(None, num_heads, rope_dim), dtype=dtype, shape_options=b_shape),
        Tensor(shape=(None, kv_lora_rank), dtype=dtype, shape_options=br_shape),
        Tensor(
            shape=(None, rope_dim),
            dtype=cos_dtype or ninetoothed.float32,
            shape_options=br_shape,
        ),
        Tensor(shape=(None, None, entry_dim), dtype=dtype, shape_options=cache_shape),
        Tensor(shape=(None,), dtype=ninetoothed.int64, shape_options=(dynamic,)),
        Tensor(shape=(None,), dtype=ninetoothed.int64, shape_options=(dynamic,)),
        Tensor(shape=(None, rope_dim), dtype=dtype, shape_options=br_shape),
        Tensor(0, constexpr=True, value=num_heads),
        Tensor(0, constexpr=True, value=kv_lora_rank),
        Tensor(0, constexpr=True, value=rope_dim),
        Tensor(0, constexpr=True, value=entry_dim),
        Tensor(0, constexpr=True, value=block_size),
        Tensor(0, constexpr=True, value=block_size),
        Tensor(0, constexpr=True, value=1 << (kv_lora_rank - 1).bit_length()),
        Tensor(0, constexpr=True, value=1 << (rope_dim - 1).bit_length()),
    )
    return arrangement, application, tensors
