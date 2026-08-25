import pytest
import torch

import ninetoothed
import ntops


skip_if_dot_scaled_is_unavailable = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Triton MXFP4 dot_scaled is unavailable",
)


def _make_mxfp4_weight(group_count, k, n, device):
    codes = torch.randint(
        0, 16, (group_count, k, n), dtype=torch.uint8, device=device
    )
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = torch.randint(
        124,
        128,
        (group_count, k // 32, n),
        dtype=torch.uint8,
        device=device,
    )

    return codes, packed.contiguous(), scales.contiguous()


def _decode_mxfp4(codes, scales):
    magnitude_code = codes & 0x7
    exponent = (magnitude_code >> 1).to(torch.int32)
    mantissa = (magnitude_code & 1).to(torch.float32)
    normal = (1.0 + 0.5 * mantissa) * torch.exp2(
        exponent.to(torch.float32) - 1.0
    )
    magnitude = torch.where(exponent == 0, 0.5 * mantissa, normal)
    sign = torch.where((codes & 0x8) == 0, 1.0, -1.0)
    block_scales = torch.exp2(scales.to(torch.float32) - 127.0)
    block_scales = block_scales.repeat_interleave(32, dim=1)

    return (sign * magnitude * block_scales).to(torch.bfloat16)


def _scaled_grouped_mm(mat_a, mat_b, scale_b, offs=None):
    return ntops.torch.scaled_grouped_mm(
        mat_a,
        mat_b,
        None,
        None,
        scale_b,
        ntops.torch.ScalingType.BlockWise1x32,
        offs=offs,
    )


@skip_if_dot_scaled_is_unavailable
@pytest.mark.parametrize("native_dtypes", (False, True))
def test_scaled_grouped_mm_uniform(native_dtypes):
    torch.manual_seed(0)
    group_count, m, k, n = 2, 17, 96, 19
    mat_a = (
        0.25
        * torch.randn(
            (group_count, m, k), dtype=torch.bfloat16, device="cuda"
        )
    ).contiguous()
    codes, mat_b, scale_b = _make_mxfp4_weight(
        group_count, k, n, mat_a.device
    )

    if native_dtypes:
        mat_b = mat_b.view(torch.float4_e2m1fn_x2)
        scale_b = scale_b.view(torch.float8_e8m0fnu)

    output = _scaled_grouped_mm(mat_a, mat_b, scale_b)
    weight = _decode_mxfp4(codes, scale_b.view(torch.uint8))
    expected = torch.bmm(mat_a.float(), weight.float()).to(torch.bfloat16)

    assert output.dtype == torch.bfloat16
    assert output.shape == (group_count, m, n)
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.03)


@skip_if_dot_scaled_is_unavailable
@pytest.mark.parametrize("native_dtypes", (False, True))
def test_scaled_grouped_mm_jagged_with_zero_token_expert(native_dtypes):
    torch.manual_seed(1)
    group_count, k, n = 3, 96, 23
    expert_rows = (4, 0, 7)
    total_m = sum(expert_rows)
    mat_a = (
        0.25
        * torch.randn((total_m, k), dtype=torch.bfloat16, device="cuda")
    ).contiguous()
    codes, mat_b, scale_b = _make_mxfp4_weight(
        group_count, k, n, mat_a.device
    )
    offs = torch.tensor((4, 4, 11), dtype=torch.int32, device=mat_a.device)

    if native_dtypes:
        mat_b = mat_b.view(torch.float4_e2m1fn_x2)
        scale_b = scale_b.view(torch.float8_e8m0fnu)

    output = _scaled_grouped_mm(mat_a, mat_b, scale_b, offs=offs)
    weight = _decode_mxfp4(codes, scale_b.view(torch.uint8))
    expected_parts = []
    start = 0
    for expert, end in enumerate(offs.cpu().tolist()):
        expected_parts.append(
            (mat_a[start:end].float() @ weight[expert].float()).to(
                torch.bfloat16
            )
        )
        start = end
    expected = torch.cat(expected_parts)

    assert output.dtype == torch.bfloat16
    assert output.shape == (total_m, n)
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.03)


def test_scaled_grouped_mm_lowers_to_direct_dot_scaled():
    from ntops.torch.scaled_grouped_mm import _enable_dot_scaled_lowering

    with _enable_dot_scaled_lowering():
        kernel = ninetoothed.make(
            *ntops.kernels.scaled_grouped_mm.premake(False),
            max_num_configs=1,
        )

    sources = kernel._compilation.artifact.sources.values()
    source = "\n".join(str(value) for value in sources)

    assert source.count("tl.dot_scaled(") == 1


def _cpu_inputs(group_count=3, total_m=None):
    k, n = 32, 5
    if total_m is None:
        mat_a = torch.zeros((group_count, 2, k), dtype=torch.bfloat16)
    else:
        mat_a = torch.zeros((total_m, k), dtype=torch.bfloat16)
    mat_b = torch.zeros((group_count, k // 2, n), dtype=torch.uint8)
    scale_b = torch.full((group_count, k // 32, n), 127, dtype=torch.uint8)

    return mat_a, mat_b, scale_b


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    (
        ("scale_a", torch.ones(1), NotImplementedError),
        ("scale_recipe_a", ntops.torch.ScalingType.TensorWise, NotImplementedError),
        ("scale_recipe_b", [ntops.torch.ScalingType.BlockWise1x32], NotImplementedError),
        ("swizzle_a", ntops.torch.SwizzleType.NO_SWIZZLE, NotImplementedError),
        ("bias", torch.ones(1), NotImplementedError),
        ("output_dtype", torch.float16, ValueError),
        ("contraction_dim", (0,), NotImplementedError),
        ("use_fast_accum", True, NotImplementedError),
    ),
)
def test_scaled_grouped_mm_rejects_unsupported_options(argument, value, error):
    mat_a, mat_b, scale_b = _cpu_inputs()
    arguments = {
        "scale_a": None,
        "scale_recipe_a": None,
        "scale_recipe_b": ntops.torch.ScalingType.BlockWise1x32,
    }
    arguments[argument] = value

    with pytest.raises(error):
        ntops.torch.scaled_grouped_mm(
            mat_a,
            mat_b,
            scale_b=scale_b,
            **arguments,
        )


def test_scaled_grouped_mm_validates_shapes_and_offsets():
    mat_a, mat_b, scale_b = _cpu_inputs()

    with pytest.raises(TypeError, match="mat_a must have dtype"):
        _scaled_grouped_mm(mat_a.float(), mat_b, scale_b)

    with pytest.raises(ValueError, match="scale_b must have shape"):
        _scaled_grouped_mm(
            mat_a, mat_b, scale_b[:, :, :-1].contiguous()
        )

    jagged_a, mat_b, scale_b = _cpu_inputs(total_m=5)

    with pytest.raises(TypeError, match="offs must have dtype"):
        _scaled_grouped_mm(jagged_a, mat_b, scale_b, torch.tensor((2, 2, 5)))

    with pytest.raises(ValueError, match="offs must be nondecreasing"):
        _scaled_grouped_mm(
            jagged_a,
            mat_b,
            scale_b,
            torch.tensor((3, 2, 5), dtype=torch.int32),
        )

    with pytest.raises(ValueError, match="offs\\[-1\\]"):
        _scaled_grouped_mm(
            jagged_a,
            mat_b,
            scale_b,
            torch.tensor((2, 2, 4), dtype=torch.int32),
        )
