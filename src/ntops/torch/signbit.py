import torch

import ntops
from ntops.torch.utils import _cached_make


def signbit(input, *, out=None):
    if out is None:
        output = torch.empty_like(input)
    else:
        output = out

    kernel = _cached_make(ntops.kernels.signbit.premake, input.ndim)

    kernel(input, output)

    return output