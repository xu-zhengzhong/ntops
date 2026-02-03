import torch

import ntops
from ntops.torch.utils import _cached_make


def rot90(input, k=1, dims=(0, 1), *, out=None):
    if out is None:
        if k % 2 == 0:
            out = torch.empty_like(input)
        else:
            dims_permute = list(range(input.ndim))
            dims_permute[dims[0]], dims_permute[dims[1]] = dims_permute[dims[1]], dims_permute[dims[0]]
            out = torch.empty(input.permute(dims_permute).shape, dtype=input.dtype, device=input.device)

    kernel = _cached_make(ntops.kernels.rot90.premake, input.ndim, k, tuple(dims))
    kernel(input, out)

    return out
