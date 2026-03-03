import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def _get_padding(tensor, dims, padding_right=True):
    if isinstance(dims, int):
        target_dims = [dims]
    elif isinstance(dims, (list, tuple)):
        target_dims = list(dims)
    else:
        raise ValueError("dims must be an int or a list/tuple of ints")

    for i, d in enumerate(target_dims):
        if d < 0:
            d += tensor.ndim

        if d < 0 or d >= tensor.ndim:
            raise ValueError(f"Invalid dims: {dims}")

        target_dims[i] = d

    padding = [0] * (tensor.ndim * 2)

    for d in target_dims:
        current_len = tensor.size(d)

        if (current_len & (current_len - 1)) == 0:
            continue
        else:
            exponent = current_len.bit_length()
            target_len = 1 << exponent

        pad_len = target_len - current_len

        pad_idx = (tensor.ndim - 1 - d) * 2 + (1 if padding_right else 0)
        padding[pad_idx] = pad_len

    return padding


def arrangement(input, descending, output, dim, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = input.ndim
    if dim < 0:
        dim += ndim

    non_target_dims = tuple(i for i in range(input.ndim) if i != dim)

    def _arrangement(input):
        arranged = input.permute(non_target_dims + (dim,))
        if ndim == 1:
            arranged = arranged.unsqueeze(0)
        arranged = arranged.flatten(end_dim=-1)
        arranged = arranged.pad(_get_padding(arranged, 1))
        arranged = arranged.tile((1, -1))
        arranged.dtype = arranged.dtype.squeeze(0)

        return arranged

    return _arrangement(input), descending, _arrangement(output)


def application(input, descending, output):
    output = ntl.sort(input, descending=descending)  # noqa: F841


def premake(ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
        Tensor(0, constexpr=True),
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
    )

    return arrangement_, application, tensors


import torch
dtype = torch.float32
device = torch.device("cuda")

torch.manual_seed(42)
dim = -1
descending = False

x = torch.randn(4, 2, dtype=dtype, device=device)
ref, _ = torch.sort(x, dim=dim, descending=descending)
y = torch.empty_like(ref)

from ntops.torch.utils import _cached_make
kernel = _cached_make(premake, y.dim(), dim)

kernel(x, descending, y)
print("Input:", x)
print("Output:", y)
print("Reference:", ref)
