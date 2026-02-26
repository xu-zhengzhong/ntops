import torch

import ntops
from ntops.torch.utils import _cached_make

def _pad_to_next_power_of_2(input, dim, pad_value=float("inf")):
    dim_size = input.shape[dim]
    dim_size_padded = 1
    while dim_size_padded < dim_size:
        dim_size_padded *= 2
    
    if dim_size_padded == dim_size:
        return input
    
    padded_shape = list(input.shape)
    padded_shape[dim] = dim_size_padded
    flattened_size = 1
    for s in padded_shape:
        flattened_size *= s
    # `infinicore.tensor` does not support `full`, so we create a tensor from a list instead.
    # padded_input = torch.tensor([pad_value] * flattened_size, dtype=input.dtype, device=input.device)
    padded_input = torch.from_list([pad_value] * flattened_size, dtype=input.dtype, device=input.device)
    padded_input = padded_input.view(padded_shape)
    padded_input.narrow(dim, 0, dim_size).copy_(input)
    
    return padded_input

def quantile(input, q, dim=None, keepdim=False, interpolation='linear', out=None):
    if isinstance(q, float):
        # q = torch.tensor([q], dtype=input.dtype, device=input.device)
        q = torch.from_list([q], dtype=input.dtype, device=input.device)
        is_scalar = True
    elif q.ndim == 0:
        q = q.unsqueeze(0)
    
    # If dim is None, input tensor will be flattened before computation.
    ndim = None
    if dim == None:
        ndim = input.ndim
        # `flatten` is not supported in `infinicore.tensor`, use `view` instead.
        flattened_size = 1
        for s in input.shape:
            flattened_size *= s
        input = input.contiguous().view([flattened_size])
        dim = 0
    
    # Pad the input and q tensors to the next power of 2 along the specified dimensions.
    input_padded = _pad_to_next_power_of_2(input, dim)
    q_padded = _pad_to_next_power_of_2(q, 0, pad_value=0.0)
    
    copy_back = False
    if out is None:
        out_shape = list(input.shape)
        out_shape[dim] = 1
        if keepdim == False:
            out_shape.pop(dim)
        elif ndim is not None:
            out_shape.extend([1] * (ndim - 1))
        out_shape.insert(0, q.shape[0])
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    elif is_scalar:
        # If `q` is a scalar, the corresponding `output` will also be a scalar,
        # but the application uses `gather` to get the sorted values, which requires
        # the `output` to have at least 1 dimension. We can unsqueeze the `output`
        # to make it compatible with the application.
        if out.is_contiguous():
            out = out.unsqueeze(0)
        else:
            # `unsqueeze` for non-contiguous `infinicore.tensor` does not work right,
            # so we create a new contiguous tensor and copy back the result after computation.
            original_out = out
            copy_back = True
            out = out.contiguous()
            out = out.unsqueeze(0)
    
    if keepdim == True:
        out_adjust = out.squeeze(dim + 1)
    else:
        out_adjust = out
    
    kernel = _cached_make(ntops.kernels.quantile.premake, input.ndim, out_adjust.ndim, dim, interpolation)
    
    kernel(input_padded, q_padded, input.shape[dim], out_adjust)
    
    if copy_back:
        original_out.copy_(out.squeeze(0))
        out = original_out
    
    return out