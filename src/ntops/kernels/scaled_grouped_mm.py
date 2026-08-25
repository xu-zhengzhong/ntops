import functools

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor


BLOCK_SIZE_M = ninetoothed.block_size(lower_bound=16)
BLOCK_SIZE_N = ninetoothed.block_size(lower_bound=16)
BLOCK_SIZE_K = ninetoothed.block_size(lower_bound=64)


def arrangement(
    mat_a,
    mat_b,
    scale_b,
    output,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    if block_size_k is None:
        block_size_k = BLOCK_SIZE_K

    output_arranged = output.tile((1, block_size_m, block_size_n))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    mat_a_arranged = mat_a.tile((1, block_size_m, block_size_k))
    mat_a_arranged = mat_a_arranged.tile((1, 1, -1))
    mat_a_arranged = mat_a_arranged.expand(
        (-1, -1, output_arranged.shape[-1])
    )
    mat_a_arranged.dtype = mat_a_arranged.dtype.squeeze((0, 1))
    mat_a_arranged.dtype.dtype = mat_a_arranged.dtype.dtype.squeeze(0)

    mat_b_arranged = mat_b.tile((1, block_size_k // 2, block_size_n))
    mat_b_arranged = mat_b_arranged.tile((1, -1, 1))
    mat_b_arranged = mat_b_arranged.expand(
        (-1, output_arranged.shape[-2], -1)
    )
    mat_b_arranged.dtype = mat_b_arranged.dtype.squeeze((0, 2))
    mat_b_arranged.dtype.dtype = mat_b_arranged.dtype.dtype.squeeze(0)

    scale_b_arranged = scale_b.tile((1, block_size_k // 32, block_size_n))
    scale_b_arranged = scale_b_arranged.tile((1, -1, 1))
    scale_b_arranged = scale_b_arranged.expand(
        (-1, output_arranged.shape[-2], -1)
    )
    scale_b_arranged.dtype = scale_b_arranged.dtype.squeeze((0, 2))
    scale_b_arranged.dtype.dtype = scale_b_arranged.dtype.dtype.squeeze(0)

    return mat_a_arranged, mat_b_arranged, scale_b_arranged, output_arranged


def application(mat_a, mat_b, scale_b, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(mat_a.shape[0]):
        accumulator = ntl.dot_scaled(
            mat_a[k],
            None,
            "bf16",
            mat_b[k],
            scale_b[k],
            "e2m1",
            accumulator,
            True,
            True,
            True,
        )

    output = accumulator


def premake(jagged=False, block_size_m=None, block_size_n=None, block_size_k=None):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )
    jagged_dim = 1 if jagged else None
    tensors = (
        Tensor(3, dtype=torch.bfloat16, jagged_dim=jagged_dim, other=0),
        Tensor(3, dtype=torch.uint8, other=0),
        Tensor(3, dtype=torch.uint8, other=127),
        Tensor(3, dtype=torch.bfloat16, jagged_dim=jagged_dim),
    )

    return arrangement_, application, tensors
