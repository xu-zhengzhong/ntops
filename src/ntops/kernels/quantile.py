import functools
import ninetoothed

import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(input, q, output, dim, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()
    
    ndim = input.ndim
    if dim < 0:
        dim += ndim

    non_target_dims = tuple(i for i in range(input.ndim) if i != dim)

    output_arranged = output.flatten(start_dim=1)
    output_arranged = output_arranged.tile((1, block_size))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    input_arranged = input.permute(non_target_dims + (dim,))
    input_arranged = input_arranged.flatten(end_dim=-1)
    input_arranged = input_arranged.tile((block_size, 1))
    input_arranged.dtype = input_arranged.dtype.squeeze(1)
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.unsqueeze(0)
    input_arranged = input_arranged.expand((output_arranged.shape[0], -1))

    q_arranged = q.tile((1,))
    q_arranged.dtype = q_arranged.dtype.squeeze(0)
    q_arranged = q_arranged.unsqueeze(1)
    q_arranged = q_arranged.expand((-1, output_arranged.shape[1]))

    return input_arranged, q_arranged, output_arranged

def linear_application(input, q, output):
    n = ntl.cast(input.shape[0], ntl.int32)
    pos = ntl.cast(q * (n - 1), ntl.float32)
    pos_int = ntl.cast(pos, ntl.int32)
    pos_int_float = ntl.cast(pos_int, ntl.float32)

    if pos == pos_int_float:
        output = input[pos_int] # noqa: F841
    else:
        i = ntl.cast(ntl.floor(pos), ntl.int32)
        frac = ntl.cast(pos - i, ntl.float32)
        
        if i + 1 < n:
            j = i + 1
        else:
            j = i

        output = input[i] + frac * (input[j] - input[i]) # noqa: F841

def lower_application(input, q, output):
    n = ntl.cast(input.shape[0], ntl.int32)
    pos = ntl.cast(q * (n - 1), ntl.float32)
    pos_int = ntl.cast(pos, ntl.int32)
    pos_int_float = ntl.cast(pos_int, ntl.float32)

    if pos == pos_int_float:
        output = input[pos_int] # noqa: F841
    else:
        i = ntl.cast(ntl.floor(pos), ntl.int32)

        output = input[i] # noqa: F841

def higher_application(input, q, output):
    n = ntl.cast(input.shape[0], ntl.int32)
    pos = ntl.cast(q * (n - 1), ntl.float32)
    pos_int = ntl.cast(pos, ntl.int32)
    pos_int_float = ntl.cast(pos_int, ntl.float32)

    if pos == pos_int_float:
        output = input[pos_int] # noqa: F841
    else:
        i = ntl.cast(ntl.floor(pos), ntl.int32)
        
        if i + 1 < n:
            j = i + 1
        else:
            j = i

        output = input[j] # noqa: F841

def nearest_application(input, q, output):
    n = ntl.cast(input.shape[0], ntl.int32)
    pos = ntl.cast(q * (n - 1), ntl.float32)
    i = ntl.cast(ntl.floor(pos), ntl.int32)
    frac = ntl.cast(pos - i, ntl.float32)
    if frac > 0.5:
        if i + 1 < n:
            i = i + 1
    elif frac == 0.5:
        # round half to even
        if i + 1 < n and (i % 2 == 1):
            i = i + 1

    output = input[i] # noqa: F841

def midpoint_application(input, q, output):
    n = ntl.cast(input.shape[0], ntl.int32)
    pos = ntl.cast(q * (n - 1), ntl.float32)
    pos_int = ntl.cast(pos, ntl.int32)
    pos_int_float = ntl.cast(pos_int, ntl.float32)

    if pos == pos_int_float:
        output = input[pos_int] # noqa: F841
    else:
        i = ntl.cast(ntl.floor(pos), ntl.int32)
        frac = ntl.cast(pos - i, ntl.float32)
        
        if i + 1 < n:
            j = i + 1
        else:
            j = i

        output = (input[i] + input[j]) / 2 # noqa: F841

def premake(in_ndim, out_ndim, dim, interpolation,  dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(in_ndim, dtype=dtype),
        Tensor(1, dtype=dtype),
        Tensor(out_ndim, dtype=dtype),
    )

    if interpolation == 'linear':
        application = linear_application
    elif interpolation == 'lower':
        application = lower_application
    elif interpolation == 'higher':
        application = higher_application
    elif interpolation == 'nearest':
        application = nearest_application
    elif interpolation == 'midpoint':
        application = midpoint_application
    else:
        raise ValueError(f"Unsupported interpolation method: {interpolation}")
    
    return arrangement_, application, tensors

# dim = -1
# interpolation = 'nearest'
# keepdim=False

# import torch
# dtype = torch.float32
# device = torch.device("cuda")

# torch.manual_seed(42)
# x = torch.rand(3, dtype=dtype, device=device)
# x, _ = torch.sort(x, dim=dim)
# q = torch.tensor([0.25, 0.5, 0.75], dtype=dtype, device=device)
# ref = torch.quantile(x, q, dim=dim, interpolation=interpolation, keepdim=keepdim)
# y = torch.empty_like(ref)

# from ntops.torch.utils import _cached_make
# kernel = _cached_make(premake, x.dim() + 1, y.dim() + 1, dim, interpolation)

# kernel(x.unsqueeze(0), q, y.unsqueeze(-1))

# print(x)
# print(y)
# print(ref)
# # assert torch.allclose(y, reference)
