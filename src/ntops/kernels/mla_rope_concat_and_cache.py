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

import ninetoothed.language as ntl

import ninetoothed
from ninetoothed import Tensor


def _source_view(tensor):
    return tensor.tile((1,) * tensor.ndim)


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
    # vector tile. Unit-tile views retain each input's source root while making
    # indirect source indexing portable across the legacy and SSA frontends.
    output = output.tile((1, 1, tile_size_value.value))
    output.dtype = output.dtype.squeeze((0, 1))
    ql_nope, q_pe, kv_c, k_pe, kv_cache, slot_mapping, positions, cos_sin_cache = (
        _source_view(tensor)
        for tensor in (
            ql_nope,
            q_pe,
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
        )
    )
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
    token_idx = output.offsets(0)
    head_idx = output.offsets(1)
    feature = output.offsets(2)
    feature_valid = feature < entry_dim
    nope_mask = feature_valid & (feature < kv_lora_rank)
    rope_mask = feature_valid & (feature >= kv_lora_rank)
    nope_feature = ntl.where(nope_mask, feature, 0)
    rope_feature = ntl.where(rope_mask, feature - kv_lora_rank, 0)
    pair = rope_feature // 2
    even = (rope_feature & 1) == 0

    ql_values = ql_nope.source[
        token_idx,
        head_idx,
        nope_feature,
    ]
    qpe0 = q_pe.source[
        token_idx,
        head_idx,
        pair * 2,
    ]
    qpe1 = q_pe.source[
        token_idx,
        head_idx,
        pair * 2 + 1,
    ]

    # Canonicalize loaded i64 aliases before mixing them with SSA index values.
    position = positions.source[token_idx].to(ntl.int64)
    cos = cos_sin_cache.source[position, pair]
    sin = cos_sin_cache.source[position, rope_dim // 2 + pair]
    qrot = ntl.where(
        even,
        qpe0 * cos - qpe1 * sin,
        qpe0 * sin + qpe1 * cos,
    )
    output = ntl.where(nope_mask, ql_values, qrot).to(ql_nope.dtype)  # noqa: F841

    latent_value = kv_c.source[token_idx, nope_feature]
    kpe0 = k_pe.source[token_idx, pair * 2]
    kpe1 = k_pe.source[token_idx, pair * 2 + 1]
    krot = ntl.where(
        even,
        kpe0 * cos - kpe1 * sin,
        kpe0 * sin + kpe1 * cos,
    )
    cache_value = ntl.where(nope_mask, latent_value, krot).to(kv_c.dtype)

    # A token's latent cache row is shared by all query heads, so head zero is
    # the sole writer. Map padding and other heads to a negative block; indexed
    # stores carry source-shape bounds masks and therefore cannot touch cache.
    slot = slot_mapping.source[token_idx].to(ntl.int64)
    write_slot = ntl.where((slot >= 0) & (head_idx == 0), slot, -1)
    block_idx = write_slot // cache_block_size
    block_offset = write_slot % cache_block_size
    kv_cache.source[block_idx, block_offset, feature] = cache_value


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
        Tensor(shape=(None, rope_dim), dtype=dtype, shape_options=br_shape),
        Tensor(shape=(None, None, entry_dim), dtype=dtype, shape_options=cache_shape),
        Tensor(shape=(None,), dtype="int64", shape_options=(dynamic,)),
        Tensor(shape=(None,), dtype="int64", shape_options=(dynamic,)),
        Tensor(
            shape=(None, rope_dim),
            dtype=cos_dtype or ninetoothed.float32,
            shape_options=br_shape,
        ),
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
