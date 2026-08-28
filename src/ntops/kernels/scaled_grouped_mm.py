import functools

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size(lower_bound=16)
BLOCK_SIZE_N = ninetoothed.block_size(lower_bound=16)
MICROSCALE_K = 32
PACKED_MICROSCALE_K = MICROSCALE_K // 2


def _arrange_activation(mat_a, output_arranged, block_size_m):
    arranged = mat_a.tile(
        (1, block_size_m, PACKED_MICROSCALE_K),
        strides=(1, block_size_m, MICROSCALE_K),
        dilation=(1, 1, 2),
    )
    arranged = arranged.tile((1, 1, -1))
    arranged = arranged.expand((-1, -1, output_arranged.shape[-1]))
    arranged.dtype = arranged.dtype.squeeze((0, 1))
    arranged.dtype.dtype = arranged.dtype.dtype.squeeze(0)
    return arranged


def arrangement(
    mat_a_even,
    mat_a_odd,
    mat_b,
    scale_b,
    output,
    block_size_m=None,
    block_size_n=None,
):
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    output_arranged = output.tile((1, block_size_m, block_size_n))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    mat_a_even_arranged = _arrange_activation(mat_a_even, output_arranged, block_size_m)
    mat_a_odd_arranged = _arrange_activation(mat_a_odd, output_arranged, block_size_m)

    mat_b_arranged = mat_b.tile((1, PACKED_MICROSCALE_K, block_size_n))
    mat_b_arranged = mat_b_arranged.tile((1, -1, 1))
    mat_b_arranged = mat_b_arranged.expand((-1, output_arranged.shape[-2], -1))
    mat_b_arranged.dtype = mat_b_arranged.dtype.squeeze((0, 2))
    mat_b_arranged.dtype.dtype = mat_b_arranged.dtype.dtype.squeeze(0)

    scale_b_arranged = scale_b.tile((1, 1, block_size_n))
    scale_b_arranged = scale_b_arranged.tile((1, -1, 1))
    scale_b_arranged = scale_b_arranged.expand((-1, output_arranged.shape[-2], -1))
    scale_b_arranged.dtype = scale_b_arranged.dtype.squeeze((0, 2))
    scale_b_arranged.dtype.dtype = scale_b_arranged.dtype.dtype.squeeze(0)

    return (
        mat_a_even_arranged,
        mat_a_odd_arranged,
        mat_b_arranged,
        scale_b_arranged,
        output_arranged,
    )


def arrangement_lookup(
    mat_a_even,
    mat_a_odd,
    mat_b,
    scale_b,
    decode_even,
    decode_odd,
    output,
    block_size_m=None,
    block_size_n=None,
):
    arranged = arrangement(
        mat_a_even,
        mat_a_odd,
        mat_b,
        scale_b,
        output,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )
    return (
        *arranged[:-1],
        decode_even.tile((1,)),
        decode_odd.tile((1,)),
        arranged[-1],
    )


def application(mat_a_even, mat_a_odd, mat_b, scale_b, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(mat_a_even.shape[0]):
        packed = (mat_b[k] + 0).to(ntl.int32)
        scale = ntl.exp2((scale_b[k] + 0).to(ntl.float32) - 127.0)

        even_code = packed & 0xF
        even_magnitude_code = even_code & 0x7
        even_exponent = even_magnitude_code >> 1
        even_mantissa = ((even_magnitude_code & 1) + 0).to(ntl.float32)
        even_normal = (1.0 + 0.5 * even_mantissa) * ntl.exp2(
            (even_exponent + 0).to(ntl.float32) - 1.0
        )
        even_magnitude = ntl.where(
            even_exponent == 0, 0.5 * even_mantissa, even_normal
        )
        even_sign = ntl.where((even_code & 0x8) == 0, 1.0, -1.0)
        weight_even = (even_sign * even_magnitude * scale).to(ntl.bfloat16)

        odd_code = (packed >> 4) & 0xF
        odd_magnitude_code = odd_code & 0x7
        odd_exponent = odd_magnitude_code >> 1
        odd_mantissa = ((odd_magnitude_code & 1) + 0).to(ntl.float32)
        odd_normal = (1.0 + 0.5 * odd_mantissa) * ntl.exp2(
            (odd_exponent + 0).to(ntl.float32) - 1.0
        )
        odd_magnitude = ntl.where(
            odd_exponent == 0, 0.5 * odd_mantissa, odd_normal
        )
        odd_sign = ntl.where((odd_code & 0x8) == 0, 1.0, -1.0)
        weight_odd = (odd_sign * odd_magnitude * scale).to(ntl.bfloat16)

        accumulator += ntl.dot(mat_a_even[k], weight_even)
        accumulator += ntl.dot(mat_a_odd[k], weight_odd)

    output = accumulator


def application_lookup(
    mat_a_even,
    mat_a_odd,
    mat_b,
    scale_b,
    decode_even,
    decode_odd,
    output,
):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(mat_a_even.shape[0]):
        packed = (mat_b[k] + 0).to(ntl.int32)
        scale = ntl.exp2((scale_b[k] + 0).to(ntl.float32) - 127.0)

        # A packed-byte lookup prevents the SSA emitter from duplicating the
        # full E2M1 decode expression inside each block-dot operand.
        weight_even = (decode_even.source[packed] * scale).to(ntl.bfloat16)
        weight_odd = (decode_odd.source[packed] * scale).to(ntl.bfloat16)

        accumulator += ntl.dot(mat_a_even[k], weight_even)
        accumulator += ntl.dot(mat_a_odd[k], weight_odd)

    output = accumulator


def premake(jagged=False, block_size_m=None, block_size_n=None, lookup=False):
    arrangement_ = functools.partial(
        arrangement_lookup if lookup else arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )
    jagged_dim = 1 if jagged else None
    common_tensors = (
        Tensor(3, dtype=torch.bfloat16, jagged_dim=jagged_dim, other=0),
        Tensor(3, dtype=torch.bfloat16, jagged_dim=jagged_dim, other=0),
        Tensor(3, dtype=torch.uint8, other=0),
        Tensor(3, dtype=torch.uint8, other=127),
    )
    lookup_tensors = (
        Tensor(1, dtype=torch.float32, other=0),
        Tensor(1, dtype=torch.float32, other=0),
    )
    output_tensor = (
        Tensor(3, dtype=torch.bfloat16, jagged_dim=jagged_dim),
    )
    tensors = common_tensors + (lookup_tensors if lookup else ()) + output_tensor

    return arrangement_, application_lookup if lookup else application, tensors
