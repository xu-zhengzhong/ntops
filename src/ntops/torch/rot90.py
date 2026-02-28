import torch

import ntops
from ntops.torch.utils import _cached_make


def _pad_to_next_power_of_2(input, dim, pad_value=0.0):
    dim_size = input.shape[dim]
    dim_size_padded = 1
    while dim_size_padded < dim_size:
        dim_size_padded *= 2
    
    if dim_size_padded == dim_size:
        return input
    
    padded_shape = list(input.shape)
    padded_shape[dim] = dim_size_padded
    flattened_size = 1
    for s in padded_shape:
        flattened_size *= s
    # `infinicore.tensor` does not support `full`, so we create a tensor from a list instead.
    # padded_input = torch.tensor([pad_value] * flattened_size, dtype=input.dtype, device=input.device)
    padded_input = torch.from_list([pad_value] * flattened_size, dtype=input.dtype, device=input.device)
    padded_input = padded_input.view(padded_shape)
    padded_input.narrow(dim, dim_size_padded - dim_size, dim_size).copy_(input)
    
    return padded_input

def rot90(input, k=1, dims=(0, 1), *, out=None):
    if out is None:
        if k % 2 == 0:
            out = torch.empty_like(input)
        else:
            dims_permute = list(range(input.ndim))
            dims_permute[dims[0]], dims_permute[dims[1]] = dims_permute[dims[1]], dims_permute[dims[0]]
            out = torch.empty(input.permute(dims_permute).shape, dtype=input.dtype, device=input.device)

    if k % 4 == 1:
        input_prepared = _pad_to_next_power_of_2(input, dims[1])
    elif k % 4 == 2:
        input_prepared = _pad_to_next_power_of_2(_pad_to_next_power_of_2(input, dims[0]), dims[1])
    elif k % 4 == 3:
        input_prepared = _pad_to_next_power_of_2(input, dims[0])
    else:  # k % 4 == 0
        input_prepared = input
    
    kernel = _cached_make(ntops.kernels.rot90.premake, input.ndim, k, dims)

    kernel(input_prepared, out)

    return out