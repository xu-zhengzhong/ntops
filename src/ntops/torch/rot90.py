import torch

import ntops
from ntops.torch.utils import _cached_make


def rot90(input, dims, k, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    if k % 4 == 1 or k % 4 == 3:
        out = out.transpose(dims[0], dims[1])
    kernel = _cached_make(ntops.kernels.rot90.premake, input.ndim, dims, k)
    kernel(input, out)

    return out
