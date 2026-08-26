import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


def _reference(input, z, weight, group_size, norm_before_gate, activation):
    x = input.float()
    gate = None if z is None else z.float()
    if gate is not None:
        gate = torch.sigmoid(gate) if activation == "sigmoid" else F.silu(gate)
    if gate is not None and not norm_before_gate:
        x = x * gate

    if group_size is None:
        output = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
    else:
        x_group = x.reshape(*x.shape[:-1], -1, group_size)
        output = (
            x_group * torch.rsqrt(x_group.square().mean(-1, keepdim=True) + 1e-5)
        ).reshape_as(x)
    output = output * weight.float()
    if gate is not None and norm_before_gate:
        output = output * gate
    return output.to(input.dtype)


@skip_if_cuda_not_available
@pytest.mark.parametrize("group_size", (None, 32))
@pytest.mark.parametrize("norm_before_gate", (False, True))
@pytest.mark.parametrize("activation", ("swish", "sigmoid"))
def test_rms_norm_gated_optimized(group_size, norm_before_gate, activation):
    input = torch.randn((3, 2, 128), device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(input)
    weight = torch.randn((128,), device="cuda", dtype=torch.float32)

    output = ntops.torch.rms_norm_gated_optimized(
        input,
        z,
        weight,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )
    reference = _reference(
        input, z, weight, group_size, norm_before_gate, activation
    )
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)


@skip_if_cuda_not_available
def test_rms_norm_gated_optimized_without_gate():
    input = torch.randn((4, 128), device="cuda", dtype=torch.float16)
    weight = torch.randn((128,), device="cuda", dtype=torch.float16)
    output = ntops.torch.rms_norm_gated_optimized(input, weight=weight)
    reference = _reference(input, None, weight, None, False, "swish")
    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)


def test_rms_norm_gated_optimized_validates_arguments():
    input = torch.randn((2, 128))
    with pytest.raises(ValueError, match="activation"):
        ntops.torch.rms_norm_gated_optimized(input, activation="gelu")
    with pytest.raises(ValueError, match="positive integer"):
        ntops.torch.rms_norm_gated_optimized(input, group_size=0)
    with pytest.raises(ValueError, match="divisible"):
        ntops.torch.rms_norm_gated_optimized(input, group_size=3)
