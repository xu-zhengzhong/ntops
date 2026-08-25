import contextlib
import enum
import threading

import torch
import torch.nn.functional as F

import ntops
from ntops.torch.utils import _cached_make


class _ScalingType(enum.IntEnum):
    TensorWise = 0
    RowWise = 1
    BlockWise1x16 = 2
    BlockWise1x32 = 3
    BlockWise1x128 = 4
    BlockWise128x128 = 5


class _SwizzleType(enum.IntEnum):
    NO_SWIZZLE = 0
    SWIZZLE_32_4_4 = 1


ScalingType = getattr(F, "ScalingType", _ScalingType)
SwizzleType = getattr(F, "SwizzleType", _SwizzleType)


def _dtype_if_available(name):
    return getattr(torch, name, None)


def _is_dtype(dtype, expected):
    return dtype is expected or (expected is not None and dtype == expected)


def _enum_matches(value, enum_type, member_name):
    member = getattr(enum_type, member_name)

    if value == member:
        return True

    return getattr(value, "name", None) == member_name


def _require_tensor(name, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_contiguous(name, value):
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _as_uint8(value):
    if value.dtype == torch.uint8:
        return value

    return value.view(torch.uint8)


_LOWERING_LOCK = threading.RLock()


@contextlib.contextmanager
def _enable_dot_scaled_lowering():
    from ninetoothed.backends.emitters import ssa as ssa_emitter
    from ninetoothed.frontend import python as python_frontend

    def lower_linalg_call(self, name, node, operands, operations):
        if name != "dot_scaled":
            return original_lower(self, name, node, operands, operations)

        if len(operands) != 10:
            raise python_frontend.LoweringError(
                "ntops dot_scaled expects the full Triton argument list"
            )

        return self._emit(
            operations,
            "linalg.dot",
            operands=(
                operands[0].name,
                operands[3].name,
                operands[4].name,
                operands[6].name,
            ),
            attrs={"ntops_dot_scaled": True},
            result_type=operands[6].type,
        )

    def emit_linalg_dot(operation, context, coords=None):
        if not operation.attrs.get("ntops_dot_scaled"):
            return original_emit(operation, context, coords=coords)

        if not (context.block_program or context.native_block_program):
            raise RuntimeError("dot_scaled requires a block-program lowering")

        lhs, rhs, rhs_scale, accumulator = operation.operands
        lhs_axes = ssa_emitter._value_axes(lhs, context)
        rhs_axes = ssa_emitter._value_axes(rhs, context)
        scale_axes = ssa_emitter._value_axes(rhs_scale, context)
        lhs_value = ssa_emitter._emit_element(
            lhs, context.target.block_coords(lhs_axes), context
        )
        rhs_value = ssa_emitter._emit_element(
            rhs, context.target.block_coords(rhs_axes), context
        )
        scale_value = ssa_emitter._emit_element(
            rhs_scale, context.target.block_coords(scale_axes), context
        )
        accumulator_value = ssa_emitter._emit_value(accumulator, context)

        return (
            f'tl.dot_scaled({lhs_value}, None, "bf16", {rhs_value}, '
            f'{scale_value}, "e2m1", acc={accumulator_value}, '
            "fast_math=True, rhs_k_pack=True)"
        )

    with _LOWERING_LOCK:
        original_lower = (
            python_frontend._ApplicationSSABuilder._lower_linalg_call
        )
        original_emit = ssa_emitter._emit_linalg_dot
        python_frontend._ApplicationSSABuilder._lower_linalg_call = (
            lower_linalg_call
        )
        ssa_emitter._emit_linalg_dot = emit_linalg_dot
        try:
            yield
        finally:
            python_frontend._ApplicationSSABuilder._lower_linalg_call = (
                original_lower
            )
            ssa_emitter._emit_linalg_dot = original_emit


def _validate_optional_arguments(
    scale_a,
    scale_recipe_a,
    scale_recipe_b,
    swizzle_a,
    swizzle_b,
    bias,
    output_dtype,
    contraction_dim,
    use_fast_accum,
):
    if scale_a is not None:
        raise NotImplementedError("scale_a is not supported for W4A16")

    if scale_recipe_a is not None:
        raise NotImplementedError("scale_recipe_a is not supported for W4A16")

    if isinstance(scale_recipe_b, (tuple, list)):
        raise NotImplementedError("multi-level scale recipes are not supported")

    if not _enum_matches(scale_recipe_b, ScalingType, "BlockWise1x32"):
        raise ValueError("scale_recipe_b must be ScalingType.BlockWise1x32")

    if swizzle_a is not None or swizzle_b is not None:
        raise NotImplementedError("swizzled inputs are not supported")

    if bias is not None:
        raise NotImplementedError("bias is not supported")

    if output_dtype != torch.bfloat16:
        raise ValueError("output_dtype must be torch.bfloat16")

    if contraction_dim not in (None, ()):
        raise NotImplementedError("contraction_dim is not supported")

    if use_fast_accum:
        raise NotImplementedError("use_fast_accum=True is not supported")


def _validate_inputs(mat_a, mat_b, scale_b, offs):
    _require_tensor("mat_a", mat_a)
    _require_tensor("mat_b", mat_b)
    _require_tensor("scale_b", scale_b)

    if mat_a.dtype != torch.bfloat16:
        raise TypeError("mat_a must have dtype torch.bfloat16")

    packed_dtype = _dtype_if_available("float4_e2m1fn_x2")
    if mat_b.dtype != torch.uint8 and not _is_dtype(mat_b.dtype, packed_dtype):
        raise TypeError(
            "mat_b must have dtype torch.uint8 or torch.float4_e2m1fn_x2"
        )

    scale_dtype = _dtype_if_available("float8_e8m0fnu")
    if (
        scale_b.dtype != torch.uint8
        and not _is_dtype(scale_b.dtype, scale_dtype)
    ):
        raise TypeError(
            "scale_b must have dtype torch.uint8 or torch.float8_e8m0fnu"
        )

    if mat_b.ndim != 3:
        raise ValueError("mat_b must have shape (G, K // 2, N)")

    if scale_b.ndim != 3:
        raise ValueError("scale_b must have shape (G, K // 32, N)")

    _require_contiguous("mat_a", mat_a)
    _require_contiguous("mat_b", mat_b)
    _require_contiguous("scale_b", scale_b)

    group_count, packed_k, n = mat_b.shape
    if group_count == 0:
        raise ValueError("mat_b must contain at least one group")

    k = packed_k * 2
    if k == 0 or k % 32 != 0:
        raise ValueError("the logical K dimension must be a positive multiple of 32")

    if scale_b.shape != (group_count, k // 32, n):
        raise ValueError(
            f"scale_b must have shape {(group_count, k // 32, n)}, "
            f"but got {tuple(scale_b.shape)}"
        )

    device = mat_a.device
    if mat_b.device != device or scale_b.device != device:
        raise ValueError("mat_a, mat_b, and scale_b must be on the same device")

    if offs is None:
        if mat_a.ndim != 3:
            raise ValueError("mat_a must have shape (G, M, K) when offs is None")

        if mat_a.shape[0] != group_count or mat_a.shape[2] != k:
            raise ValueError(
                f"mat_a must have shape (G, M, K) with G={group_count} and K={k}"
            )

        return False, group_count, mat_a.shape[1], k, n

    _require_tensor("offs", offs)
    if mat_a.ndim != 2:
        raise ValueError("mat_a must have shape (total_M, K) when offs is provided")

    if mat_a.shape[1] != k:
        raise ValueError(f"mat_a must have K={k}, but got K={mat_a.shape[1]}")

    if offs.ndim != 1 or offs.numel() != group_count:
        raise ValueError(f"offs must have shape ({group_count},)")

    if offs.dtype != torch.int32:
        raise TypeError("offs must have dtype torch.int32")

    if offs.device != device:
        raise ValueError("offs must be on the same device as the inputs")

    _require_contiguous("offs", offs)

    if bool(torch.any(offs < 0).item()):
        raise ValueError("offs must contain non-negative cumulative row counts")

    if group_count > 1 and bool(torch.any(offs[1:] < offs[:-1]).item()):
        raise ValueError("offs must be nondecreasing")

    if int(offs[-1].item()) != mat_a.shape[0]:
        raise ValueError("offs[-1] must equal mat_a.shape[0]")

    return True, group_count, mat_a.shape[0], k, n


def scaled_grouped_mm(
    mat_a,
    mat_b,
    scale_a,
    scale_recipe_a,
    scale_b,
    scale_recipe_b,
    swizzle_a=None,
    swizzle_b=None,
    bias=None,
    offs=None,
    output_dtype=torch.bfloat16,
    contraction_dim=(),
    use_fast_accum=False,
):
    """Compute BF16 activations times block-scaled MXFP4 grouped weights."""
    _validate_optional_arguments(
        scale_a,
        scale_recipe_a,
        scale_recipe_b,
        swizzle_a,
        swizzle_b,
        bias,
        output_dtype,
        contraction_dim,
        use_fast_accum,
    )
    jagged, group_count, m, _, n = _validate_inputs(
        mat_a, mat_b, scale_b, offs
    )

    output_shape = (group_count, m, n) if not jagged else (m, n)
    output = torch.empty(
        output_shape, dtype=torch.bfloat16, device=mat_a.device
    )
    with _enable_dot_scaled_lowering():
        kernel = _cached_make(ntops.kernels.scaled_grouped_mm.premake, jagged)
    mat_b_uint8 = _as_uint8(mat_b)
    scale_b_uint8 = _as_uint8(scale_b)

    if not jagged:
        kernel(mat_a, mat_b_uint8, scale_b_uint8, output)
        return output

    offsets = torch.cat((offs.new_zeros(1), offs))
    mat_a_jagged = torch.nested.nested_tensor_from_jagged(mat_a, offsets)
    output_jagged = torch.nested.nested_tensor_from_jagged(output, offsets)
    kernel(mat_a_jagged, mat_b_uint8, scale_b_uint8, output_jagged)

    return output
