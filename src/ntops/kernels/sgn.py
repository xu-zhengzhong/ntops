import functools
import ninetoothed

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice


def arrangement(input, other, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    output_arranged = output.flatten(end_dim=-1)
    output_arranged = output_arranged.tile((block_size,  1))

    input_arranged = input.flatten(end_dim=-1)
    input_arranged = input_arranged.tile((block_size, 1))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    other_arranged = other.flatten(end_dim=-1)
    other_arranged = other_arranged.tile((block_size, 1))

    return input_arranged, other_arranged, output_arranged

def application(input, other, output):
    denominators = ntl.sqrt(libdevice.pow(input[0], 2) + libdevice.pow(input[1], 2))
    output = other / ntl.where(denominators == 0.0, 1.0, denominators)  # noqa: F841

def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors

# kernel = ninetoothed.make(arrangement, application, (Tensor(3), Tensor(3), Tensor(3)))

# import torch
# dtype = torch.complex64
# device = torch.device("cuda")

# x = torch.tensor([[3+4j, 7-24j, 0, 1+2j],[3+4j, 7-24j, 0, 1+2j]], dtype=dtype, device=device)
# x_rm = torch.view_as_real(x)
# y = torch.empty_like(x)
# y_rm = torch.view_as_real(y)

# print(y_rm)
# kernel(x_rm, x_rm, y_rm)

# reference = torch.sgn(x)

# print(x_rm)
# print(y_rm)
# # assert torch.allclose(y, x)
# assert torch.allclose(y, reference)
