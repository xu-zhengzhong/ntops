import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_signbit(shape, dtype, device, rtol, atol):
    def generate_tensor(shape, dtype, device):
        x = torch.randn(shape, dtype=dtype, device=device)

        # inject +/-inf and -0.0 to cover corner cases
        probs = (0.2, 0.4, 0.6)
        p = torch.rand(shape, device=device)

        mask = p < probs[0]
        x[mask] = float("inf")

        mask = (probs[0] <= p) & (p < probs[1])
        x[mask] = float("-inf")

        mask = (probs[1] <= p) & (p < probs[2])
        x[mask] = torch.tensor(-0.0, dtype=dtype, device=device)

        return x

    input = generate_tensor(shape, dtype, device)

    ninetoothed_output = ntops.torch.signbit(input)
    reference_output = torch.signbit(input)

    assert torch.equal(ninetoothed_output, reference_output)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_signbit_out(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    out = torch.empty_like(input)

    ret = ntops.torch.signbit(input, out=out)
    reference = torch.signbit(input)

    assert ret is out
    assert torch.equal(out, reference)