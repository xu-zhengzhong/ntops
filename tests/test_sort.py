import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize("descending", (False, True))
@pytest.mark.parametrize(*generate_arguments())
def test_sort(shape, descending, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.sort(input, descending=descending)
    reference_output = torch.sort(input, descending=descending)[0]

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
