import torch

import ntops
from ntops.torch.utils import _cached_make


def sgn(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    input = torch.view_as_real(input)
    out_rm = torch.view_as_real(out)

    kernel = _cached_make(ntops.kernels.sgn.premake, input.ndim)

    kernel(input, input, out_rm)

    return out
