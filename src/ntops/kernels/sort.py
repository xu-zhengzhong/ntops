import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


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


# import torch
# dtype = torch.float32
# device = torch.device("cuda")

# torch.manual_seed(42)
# dim = -1
# descending = False

# x = torch.randn(3, 7, dtype=dtype, device=device)
# length = x.shape[dim]
# a = 1
# while a < length:
#     a *= 2
# new_length = a
# padded_shape = list(x.shape)
# padded_shape[dim] = new_length
# padded_x = torch.empty(padded_shape, dtype=dtype, device=device)
# padded_x[..., :length] = x
# padded_x[..., length:] = float("inf")
# ref = torch.sort(padded_x, dim=dim, descending=descending).values[..., :length]
# y = torch.empty_like(padded_x)

# from ntops.torch.utils import _cached_make
# kernel = _cached_make(premake, y.dim(), dim)

# print("Input:")
# print(x)
# print("Padded Input:")
# print(padded_x)
# print("Output:")
# kernel(padded_x, descending, y)
# print(y)
# print("Reference:")
# print(ref)
