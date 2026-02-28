import functools
import ninetoothed

import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, output, k, dims, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = input.ndim
    dims = tuple(dim if dim >= 0 else dim + ndim for dim in dims)
    non_target_dims = tuple(i for i in range(ndim) if i not in dims)

    def _arrange_0(tensor):
        arranged = tensor.flatten()
        arranged = arranged.tile((block_size, ))

        return arranged
    
    def _arrange_1_or_3(tensor, dims):
        arranged = tensor.permute(non_target_dims + dims)
        arranged = arranged.flatten(end_dim=-1)
        arranged = arranged.tile((1, -1))
        arranged.dtype = arranged.dtype.squeeze(0)
        
        return arranged

    def _arrange_2(tensor, dims):
        arranged = tensor.permute(non_target_dims + dims)
        if ndim == 2: 
            arranged = arranged.unsqueeze(0)
        arranged = arranged.flatten(end_dim=-2)
        arranged = arranged.tile((1, -1, -1))
        arranged.dtype = arranged.dtype.squeeze(0)
        
        return arranged
    
    if k % 4 == 0:
        input_arranged = _arrange_0(input)
        output_arranged = _arrange_0(output)
    elif k % 4 == 1:
        input_arranged = _arrange_1_or_3(input, dims)
        output_arranged = _arrange_1_or_3(output, tuple(reversed(dims)))
    elif k % 4 == 3:
        input_arranged = _arrange_1_or_3(input, tuple(reversed(dims)))
        output_arranged = _arrange_1_or_3(output, dims)
    else:  # k % 4 == 2
        input_arranged = _arrange_2(input, dims)
        output_arranged = _arrange_2(output, dims)

    return input_arranged, output_arranged

def application_0(input, output):
    output = input # noqa: F841

def application_1_or_3(input, output):
    if input.shape[0] == 1:
        output = input # noqa: F841
    else:
        output = ntl.flip(input, 0) # noqa: F841
    output = ntl.flip(input, 0) # noqa: F841

def application_2(input, output):
    output = ntl.flip(ntl.flip(input, 0), 1) # noqa: F841

def premake(ndim, k, dims, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, k=k, dims=dims, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
    )

    if k % 4 == 0:
        application = application_0
    elif k % 4 == 2:
        application = application_2
    else:  # k % 4 == 1 or 3
        application = application_1_or_3

    return arrangement_, application, tensors


# def _pad_to_next_power_of_2(input, dim, pad_value=float("inf")):
#     dim_size = input.shape[dim]
#     dim_size_padded = 1
#     while dim_size_padded < dim_size:
#         dim_size_padded *= 2
    
#     if dim_size_padded == dim_size:
#         return input
    
#     padded_shape = list(input.shape)
#     padded_shape[dim] = dim_size_padded
#     flattened_size = 1
#     for s in padded_shape:
#         flattened_size *= s
#     # `infinicore.tensor` does not support `full`, so we create a tensor from a list instead.
#     padded_input = torch.tensor([pad_value] * flattened_size, dtype=input.dtype, device=input.device)
#     # padded_input = torch.from_list([pad_value] * flattened_size, dtype=input.dtype, device=input.device)
#     padded_input = padded_input.view(padded_shape)
#     padded_input.narrow(dim, dim_size_padded - dim_size, dim_size).copy_(input)
    
#     return padded_input


# import torch
# max_display = 20
# dims = (0, 1)
# k = 1
# shape = (12, 1, 3, 54)
# ndim = len(shape)
# dtype = torch.float32
# device = torch.device("cuda")

# size = 1
# for dim in shape:
#     size *= dim
# input = torch.arange(size, dtype=dtype, device=device).view(shape)
# ref = torch.rot90(input, k=k, dims=dims)
# output = torch.empty_like(ref)

# from ntops.torch.utils import _cached_make
# kernel = _cached_make(premake, input.dim(), k=k, dims=dims)

# if k == 1:
#     input_prepared = _pad_to_next_power_of_2(input, dims[1])
# elif k == 2:
#     input_prepared = _pad_to_next_power_of_2(_pad_to_next_power_of_2(input, dims[0]), dims[1])
# elif k == 3:
#     input_prepared = _pad_to_next_power_of_2(input, dims[0])
# else:  # k == 0
#     input_prepared = input

# kernel(input_prepared, output)

# if size <= max_display:
#     print("Input:", input)
#     print("Output:", output)
#     print("Reference:", ref)
# else:
#     # print only the first max_display elements for large tensors
#     print("Input:", input.flatten()[:max_display])
#     print("Output:", output.flatten()[:max_display])
#     print("Reference:", ref.flatten()[:max_display])
#     if not torch.allclose(output, ref):
#         print("Output does not match reference!")