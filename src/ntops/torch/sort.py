import torch

import ntops
from ntops.torch.utils import _cached_make


def sort(input, dim=-1, descending=False, out=None):
    if out is None:
        out = torch.empty_like(input)

    dim_size = input.shape[dim]
    dim_size_padded = 1
    while dim_size_padded < dim_size:
        dim_size_padded *= 2

    kernel = _cached_make(ntops.kernels.sort.premake, input.ndim, dim)

    if dim_size_padded != dim_size:
        out_padded_shape = list(input.shape)
        out_padded_shape[dim] = dim_size_padded
        out_padded = torch.empty(
            out_padded_shape, dtype=input.dtype, device=input.device
        )

        input_padded = torch.empty(
            out_padded_shape, dtype=input.dtype, device=input.device
        )
        input_padded.narrow(dim, 0, dim_size).copy_(input)
        if descending:
            input_padded.narrow(dim, dim_size, dim_size_padded - dim_size).copy_(
                float("-inf")
            )
        else:
            input_padded.narrow(dim, dim_size, dim_size_padded - dim_size).copy_(
                float("inf")
            )

        kernel(input_padded, descending, out_padded)

        out.copy_(out_padded.narrow(dim, 0, dim_size))
    else:
        kernel(input, descending, out)

    return out
