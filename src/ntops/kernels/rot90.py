'''
Docstring for ntops.src.ntops.kernels.rot90

rot90(input, k=1, dims=(0,1)) -> Tensor
Rotate a matrix by 90 degrees in the plane specified by dims axes.

- for k % 4 = 1: output = input.flip(dims[1]).transpose(dims[0], dims[1])
- for k % 4 = 2: output = input.flip(dims[0]).flip(dims[1])
- for k % 4 = 3: output = input.transpose(dims[0], dims[1]).flip(dims[1])
'''
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(input, output, k, dims, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = input.ndim

    dims     = tuple(dim if dim >= 0 else dim + ndim for dim in dims)

    non_target_dims = tuple(i for i in range(ndim) if i not in dims)

    def _arrange_0(tensor):
        arranged = tensor.flatten()
        arranged = arranged.tile((block_size, ))

        return arranged
    
    def _arrange_1_or_3(tensor, dims):
        arranged = tensor.permute(non_target_dims + dims)
        arranged = arranged.flatten(end_dim=-1)
        arranged = arranged.tile((1, 1))
        arranged = arranged.tile((block_size, -1))
        
        return arranged

    def _arrange_2(tensor, dims):
        arranged = tensor.permute(non_target_dims + dims)
        if ndim == 2: 
            arranged = arranged.unsqueeze(0)
        arranged = arranged.flatten(end_dim=-2)
        arranged = arranged.tile((1, 1, 1))
        arranged = arranged.tile((block_size, -1, -1))
        
        return arranged
    
    k = k % 4

    
    if k == 0:
        input_arranged = _arrange_0(input)
        output_arranged = _arrange_0(output)
    elif k == 1:
        input_arranged = _arrange_1_or_3(input, dims)
        output_arranged = _arrange_1_or_3(output, tuple(reversed(dims)))
    elif k == 3:
        input_arranged = _arrange_1_or_3(input, tuple(reversed(dims)))
        output_arranged = _arrange_1_or_3(output, dims)
    else:  # k == 2
        input_arranged = _arrange_2(input, dims)
        output_arranged = _arrange_2(output, dims)

    return input_arranged, output_arranged

def application_0(input, output):
    output = input # noqa: F841

def application_1_or_3(input, output):
    m, n = input.shape
    for i in range(m):
        for j in range(n):
            output[i, j] = input[i, n - 1 - j]  # noqa: F841

def application_2(input, output):
    x, y, z = input.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                output[i, j, k] = input[i, y - 1 - j, z - 1 - k]  # noqa: F841

def premake(ndim, k=1, dims=(0, 1), dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, k=k, dims=dims, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    if k % 4 == 0:
        application = application_0
    elif k % 4 == 2:
        application = application_2
    else:  # k % 4 == 1 or 3
        application = application_1_or_3

    return arrangement_, application, tensors

# dims = (0, 1)
# k = 3
# ndim = 4
# # kernel = ninetoothed.make(functools.partial(arrangement, dims=dims, k=k), application_2, (Tensor(ndim), Tensor(ndim)))
# kernel = ninetoothed.make(functools.partial(arrangement, dims=dims, k=k), application_1_or_3, (Tensor(ndim), Tensor(ndim)))

# import torch
# dtype = torch.float32
# device = torch.device("cuda")

# x = torch.arange(138*191*1*229, dtype=dtype, device=device).view(138, 191, 1, 229)
# reference = torch.rot90(x, k=k, dims=dims)
# y = torch.empty_like(reference)

# # print(y)
# kernel(x, y)

# print(reference[0, 0, 0, :5])
# print(y[0, 0, 0, :5])
# # for i in range(y.shape[3]):
# #     if not torch.allclose(y[:, :, :, i], reference[:, :, :, i]):
# #         print(f"Mismatch at index {i}:")
# #         print("y:", y[:, :, :, i])
# #         print("reference:", reference[:, :, :, i])
# #         break
# # assert torch.allclose(y, x)
# assert torch.allclose(y, reference)
