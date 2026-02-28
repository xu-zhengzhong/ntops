import functools
import ninetoothed

import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    def _arrange(input):
        arranged = input.flatten(end_dim=-1)
        arranged = arranged.tile((block_size,  1))
        arranged = arranged.tile((1, -1))
        arranged.dtype = arranged.dtype.squeeze(0)

        return arranged

    return _arrange(input), _arrange(output)

def application(input, output):
    denominators = ntl.sqrt(input[0] * input[0] + input[1] * input[1])
    denominators = ntl.where(denominators == 0.0, 1.0, denominators)
    for i in range(input.shape[0]):
        output[i] = input[i] / denominators  # noqa: F841

def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors

# import torch
# dtype = torch.complex64
# device = torch.device("cuda")

# input = torch.tensor([[3+4j, 7-24j, 0, 1+2j],[3+4j, 7-24j, 0, 1+2j]], dtype=dtype, device=device)
# input_rm = torch.view_as_real(input)
# ref = torch.sgn(input)
# output = torch.empty_like(ref)
# output_rm = torch.view_as_real(output)

# from ntops.torch.utils import _cached_make
# kernel = _cached_make(premake, input_rm.dim())

# kernel(input_rm, output_rm)

# print("Input:", input)
# print("Output:", output)
# print("Reference:", ref)