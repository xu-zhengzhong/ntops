import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def _silu(gate):
    return gate / (1 + ntl.exp(-gate))


def _arrange_last_dim(tensor, block_size):
    # Keep rows outside the hierarchy and reduce only the hidden dimension.
    non_target_dims = tuple(range(tensor.ndim - 1))
    inner_shape = (1,) * len(non_target_dims) + (block_size,)
    outer_shape = (1,) * len(non_target_dims) + (-1,)
    arranged = tensor.tile(inner_shape)
    arranged = arranged.tile(outer_shape)
    arranged.dtype = arranged.dtype.squeeze(non_target_dims)
    arranged.dtype.dtype = arranged.dtype.dtype.squeeze(non_target_dims)
    return arranged


def _arrange_group(tensor, group_size):
    return tensor.flatten().tile((group_size,))


def arrangement(
    input,
    gate,
    weight,
    eps,
    output,
    num_normalized_elements,
    group_size=None,
    block_size=128,
):
    if group_size is None:
        input, gate, weight, output = (
            _arrange_last_dim(tensor, block_size)
            for tensor in (input, gate, weight, output)
        )
    else:
        input, gate, weight, output = (
            _arrange_group(tensor, group_size)
            for tensor in (input, gate, weight, output)
        )

    return input, gate, weight, eps, output, num_normalized_elements


def arrangement_no_gate(
    input,
    weight,
    eps,
    output,
    num_normalized_elements,
    group_size=None,
    block_size=128,
):
    if group_size is None:
        input, weight, output = (
            _arrange_last_dim(tensor, block_size)
            for tensor in (input, weight, output)
        )
    else:
        input, weight, output = (
            _arrange_group(tensor, group_size)
            for tensor in (input, weight, output)
        )

    return input, weight, eps, output, num_normalized_elements


def application_silu_before(
    input, gate, weight, eps, output, num_normalized_elements
):
    rms = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        rms += input_i * input_i

    rms_value = ntl.sqrt(ntl.sum(rms) / num_normalized_elements + eps)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        gate_i = ntl.cast(gate[i], ntl.float32)
        normed_i = input_i / rms_value * ntl.cast(weight[i], ntl.float32)
        gated_i = _silu(gate_i)
        output[i] = normed_i * gated_i


def application_silu_after(
    input, gate, weight, eps, output, num_normalized_elements
):
    rms = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        gate_i = ntl.cast(gate[i], ntl.float32)
        gated_i = input_i * _silu(gate_i)
        rms += gated_i * gated_i

    rms_value = ntl.sqrt(ntl.sum(rms) / num_normalized_elements + eps)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        gate_i = ntl.cast(gate[i], ntl.float32)
        gated_i = input_i * _silu(gate_i)
        output_i = gated_i / rms_value * ntl.cast(weight[i], ntl.float32)
        output[i] = output_i


def application_sigmoid_before(
    input, gate, weight, eps, output, num_normalized_elements
):
    rms = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        rms += input_i * input_i

    rms_value = ntl.sqrt(ntl.sum(rms) / num_normalized_elements + eps)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        gate_i = ntl.cast(gate[i], ntl.float32)
        normed_i = input_i / rms_value * ntl.cast(weight[i], ntl.float32)
        gated_i = ntl.sigmoid(gate_i)
        output[i] = normed_i * gated_i


def application_sigmoid_after(
    input, gate, weight, eps, output, num_normalized_elements
):
    rms = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        gate_i = ntl.cast(gate[i], ntl.float32)
        gated_i = input_i * ntl.sigmoid(gate_i)
        rms += gated_i * gated_i

    rms_value = ntl.sqrt(ntl.sum(rms) / num_normalized_elements + eps)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        gate_i = ntl.cast(gate[i], ntl.float32)
        gated_i = input_i * ntl.sigmoid(gate_i)
        output_i = gated_i / rms_value * ntl.cast(weight[i], ntl.float32)
        output[i] = output_i


def application_no_gate(input, weight, eps, output, num_normalized_elements):
    rms = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        rms += input_i * input_i

    rms_value = ntl.sqrt(ntl.sum(rms) / num_normalized_elements + eps)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        output_i = input_i / rms_value * ntl.cast(weight[i], ntl.float32)
        output[i] = output_i


def application_silu_before_group(
    input, gate, weight, eps, output, num_normalized_elements
):
    rms = ntl.zeros((1,), dtype=ntl.float32)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        rms += input_i * input_i
    rms_value = ntl.sqrt(ntl.sum(rms) / num_normalized_elements + eps)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        normed_i = input_i / rms_value * ntl.cast(weight[i], ntl.float32)
        gated_i = _silu(ntl.cast(gate[i], ntl.float32))
        output[i] = normed_i * gated_i


def application_silu_after_group(
    input, gate, weight, eps, output, num_normalized_elements
):
    rms = ntl.zeros((1,), dtype=ntl.float32)
    for i in range(input.shape[0]):
        gated_i = ntl.cast(input[i], ntl.float32) * _silu(
            ntl.cast(gate[i], ntl.float32)
        )
        rms += gated_i * gated_i
    rms_value = ntl.sqrt(ntl.sum(rms) / num_normalized_elements + eps)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        gated_i = input_i * _silu(ntl.cast(gate[i], ntl.float32))
        output_i = gated_i / rms_value * ntl.cast(weight[i], ntl.float32)
        output[i] = output_i


def application_sigmoid_before_group(
    input, gate, weight, eps, output, num_normalized_elements
):
    rms = ntl.zeros((1,), dtype=ntl.float32)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        rms += input_i * input_i
    rms_value = ntl.sqrt(ntl.sum(rms) / num_normalized_elements + eps)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        normed_i = input_i / rms_value * ntl.cast(weight[i], ntl.float32)
        gated_i = ntl.sigmoid(ntl.cast(gate[i], ntl.float32))
        output[i] = normed_i * gated_i


def application_sigmoid_after_group(
    input, gate, weight, eps, output, num_normalized_elements
):
    rms = ntl.zeros((1,), dtype=ntl.float32)
    for i in range(input.shape[0]):
        gated_i = ntl.cast(input[i], ntl.float32) * ntl.sigmoid(
            ntl.cast(gate[i], ntl.float32)
        )
        rms += gated_i * gated_i
    rms_value = ntl.sqrt(ntl.sum(rms) / num_normalized_elements + eps)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        gated_i = input_i * ntl.sigmoid(ntl.cast(gate[i], ntl.float32))
        output_i = gated_i / rms_value * ntl.cast(weight[i], ntl.float32)
        output[i] = output_i


def application_no_gate_group(
    input, weight, eps, output, num_normalized_elements
):
    rms = ntl.zeros((1,), dtype=ntl.float32)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        rms += input_i * input_i
    rms_value = ntl.sqrt(ntl.sum(rms) / num_normalized_elements + eps)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        output_i = input_i / rms_value * ntl.cast(weight[i], ntl.float32)
        output[i] = output_i


def premake(
    ndim,
    hidden_size,
    group_size=None,
    norm_before_gate=True,
    activation="silu",
    dtype=None,
    input_dtype=None,
    gate_dtype=None,
    weight_dtype=None,
    output_dtype=None,
    block_size=128,
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
    if block_size is None:
        block_size = 128

    if has_gate:
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
        )
    else:
        arrangement_ = functools.partial(
            arrangement_no_gate,
            group_size=group_size,
            block_size=block_size,
        )
        tensors = (
            Tensor(ndim, other=0, dtype=input_dtype),
            Tensor(ndim, dtype=weight_dtype),
            Tensor(0, dtype=ninetoothed.float64),
            Tensor(ndim, dtype=output_dtype),
            Tensor(0, dtype=ninetoothed.float64),
        )

    if not has_gate:
        application = (
            application_no_gate_group
            if group_size is not None
            else application_no_gate
        )
    elif activation == "sigmoid":
        application = (
            application_sigmoid_before_group
            if group_size is not None and norm_before_gate
            else application_sigmoid_after_group
            if group_size is not None
            else application_sigmoid_before
            if norm_before_gate
            else application_sigmoid_after
        )
    else:
        application = (
            application_silu_before_group
            if group_size is not None and norm_before_gate
            else application_silu_after_group
            if group_size is not None
            else application_silu_before
            if norm_before_gate
            else application_silu_after
        )

    return arrangement_, application, tensors
