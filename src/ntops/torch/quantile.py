import torch

import ntops
from ntops.torch.utils import _cached_make


def quantile(input, q, dim=-1, keepdim=False, interpolation='linear', out=None):
    # Sort the input tensor along the specified dimension
    sorted = torch.empty_like(input)
    
    dim_size = input.shape[dim]
    dim_size_padded = 1
    while dim_size_padded < dim_size:
        dim_size_padded *= 2
    
    sort_kernel = _cached_make(ntops.kernels.sort.premake, input.ndim, dim)

    if dim_size_padded != dim_size:
        sorted_padded_shape = list(input.shape)
        sorted_padded_shape[dim] = dim_size_padded
        sorted_padded = torch.empty(sorted_padded_shape, dtype=input.dtype, device=input.device)

        input_padded = torch.empty(sorted_padded_shape, dtype=input.dtype, device=input.device)
        input_padded.narrow(dim, 0, dim_size).copy_(input)
        input_padded.narrow(dim, dim_size, dim_size_padded - dim_size).copy_(float("inf"))

        sort_kernel(input_padded, False, sorted_padded)

        sorted.copy_(sorted_padded.narrow(dim, 0, dim_size))
    else:
        sort_kernel(input, False, sorted)
    
    # Compute the quantiles using the sorted tensor
    out_shape = list(input.shape)
    out_shape.insert(0, q.shape[0])
    out_shape[dim + 1] = 1

    if keepdim == False:
        out_shape.pop(dim + 1)

    if out is None:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    
    kernel = _cached_make(ntops.kernels.quantile.premake, input.ndim, out.ndim, dim, interpolation)
    
    kernel(sorted, q, out)
    
    return out