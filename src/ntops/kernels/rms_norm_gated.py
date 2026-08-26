import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement as reduction_arrangement


def _silu_gate(input, gate):
    return input * (gate / (1 + ntl.exp(-gate)))


def _sigmoid_gate(input, gate):
    return input * ntl.sigmoid(gate)


def _application(
    input,
    gate,
    weight,
    eps,
    output,
    num_normalized_elements,
    has_gate,
    norm_before_gate,
    gate_fn,
    rms,
):
    if has_gate and not norm_before_gate:
        for i in range(input.shape[0]):
            input_i = gate_fn(
                ntl.cast(input[i], ntl.float32),
                ntl.cast(gate[i], ntl.float32),
            )
            rms += input_i * input_i
    else:
        for i in range(input.shape[0]):
            input_i = ntl.cast(input[i], ntl.float32)
            rms += input_i * input_i

    rms_value = ntl.sqrt(ntl.sum(rms) / num_normalized_elements + eps)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        if has_gate and not norm_before_gate:
            input_i = gate_fn(input_i, ntl.cast(gate[i], ntl.float32))

        output_i = input_i / rms_value * ntl.cast(weight[i], ntl.float32)
        if has_gate and norm_before_gate:
            output_i = gate_fn(output_i, ntl.cast(gate[i], ntl.float32))
        output[i] = output_i


def application_silu(
    input,
    gate,
    weight,
    eps,
    output,
    num_normalized_elements,
    has_gate,
    norm_before_gate,
):
    rms = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    _application(
        input,
        gate,
        weight,
        eps,
        output,
        num_normalized_elements,
        has_gate,
        norm_before_gate,
        _silu_gate,
        rms,
    )


def application_sigmoid(
    input,
    gate,
    weight,
    eps,
    output,
    num_normalized_elements,
    has_gate,
    norm_before_gate,
):
    rms = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    _application(
        input,
        gate,
        weight,
        eps,
        output,
        num_normalized_elements,
        has_gate,
        norm_before_gate,
        _sigmoid_gate,
        rms,
    )


def application_silu_group(
    input,
    gate,
    weight,
    eps,
    output,
    num_normalized_elements,
    has_gate,
    norm_before_gate,
):
    rms = ntl.zeros((1,), dtype=ntl.float32)
    _application(
        input,
        gate,
        weight,
        eps,
        output,
        num_normalized_elements,
        has_gate,
        norm_before_gate,
        _silu_gate,
        rms,
    )


def application_sigmoid_group(
    input,
    gate,
    weight,
    eps,
    output,
    num_normalized_elements,
    has_gate,
    norm_before_gate,
):
    rms = ntl.zeros((1,), dtype=ntl.float32)
    _application(
        input,
        gate,
        weight,
        eps,
        output,
        num_normalized_elements,
        has_gate,
        norm_before_gate,
        _sigmoid_gate,
        rms,
    )


def _arrange_group(tensor, group_size):
    return tensor.flatten().tile((group_size,))


def arrangement(
    input,
    gate,
    weight,
    eps,
    output,
    num_normalized_elements,
    has_gate,
    norm_before_gate,
    group_size=None,
    block_size=None,
):
    if group_size is None:
        input, gate, weight, output = reduction_arrangement(
            input,
            gate,
            weight,
            output,
            dim=-1,
            block_size=block_size,
        )
    else:
        input, gate, weight, output = (
            _arrange_group(tensor, group_size)
            for tensor in (input, gate, weight, output)
        )

    return (
        input,
        gate,
        weight,
        eps,
        output,
        num_normalized_elements,
        has_gate,
        norm_before_gate,
    )


def premake(
    ndim,
    group_size=None,
    norm_before_gate=False,
    activation="swish",
    dtype=None,
    input_dtype=None,
    gate_dtype=None,
    weight_dtype=None,
    output_dtype=None,
    block_size=None,
    has_gate=True,
):
    if activation not in ("silu", "sigmoid", "swish"):
        raise ValueError(
            "activation must be one of 'silu', 'sigmoid', or 'swish'"
        )

    if dtype is not None:
        input_dtype = input_dtype or dtype
        gate_dtype = gate_dtype or dtype
        weight_dtype = weight_dtype or dtype
        output_dtype = output_dtype or dtype

    arrangement_ = functools.partial(
        arrangement,
        group_size=group_size,
        block_size=block_size,
    )

    tensors = (
        Tensor(ndim, other=0, dtype=input_dtype),
        Tensor(ndim, other=0, dtype=gate_dtype),
        Tensor(ndim, dtype=weight_dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(ndim, dtype=output_dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(0, constexpr=True, value=has_gate),
        Tensor(0, constexpr=True, value=norm_before_gate),
    )

    if activation == "sigmoid":
        application = (
            application_sigmoid_group
            if group_size is not None
            else application_sigmoid
        )
    else:
        application = (
            application_silu_group if group_size is not None else application_silu
        )

    return arrangement_, application, tensors
