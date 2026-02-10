import pytest
import torch
import random

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize("keepdim", (False, True))
@pytest.mark.parametrize("interpolation", ('linear', 'lower', 'higher', 'nearest', 'midpoint'))
@pytest.mark.parametrize(*generate_arguments())
def test_quantile(shape, keepdim, interpolation, dtype, device, rtol, atol):
    # quantile 不支持 float16
    if dtype is torch.float16:
        return
    
    input = torch.randn(shape, dtype=dtype, device=device)
    q_size = random.randint(1, 10)
    q = torch.rand(q_size, dtype=dtype, device=device)
    dim = random.randint(0, input.ndim - 1)

    ninetoothed_output = ntops.torch.quantile(input, q, dim=dim, keepdim=keepdim, interpolation=interpolation)
    reference_output = torch.quantile(input, q, dim=dim, keepdim=keepdim, interpolation=interpolation)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
