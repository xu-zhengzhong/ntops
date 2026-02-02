# kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)))

# dtype = torch.complex64
# device = torch.device("cuda")

# x = torch.tensor([3+4j, 7-24j, 0, 1+2j], dtype=dtype, device=device)
# x_rm = torch.view_as_real(x)
# y = torch.empty_like(x)
# y_rm = torch.view_as_real(y)

# print(y_rm)
# kernel(x_rm, x_rm, y_rm)

# reference = torch.sgn(x)

# print(x_rm)
# print(y_rm)
# # assert torch.allclose(y, x)
# assert torch.allclose(y, reference)

import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def generate_arguments():
    arguments = []

    # 定义需要测试的张量类型：分为「非复数张量」和「复数张量」两大类
    # 注意：复数张量对应 cfloat32/cfloat16，与非复数张量一一对应
    tensor_type_groups = [
        # 非复数张量（float系列，等价于torch.sign()）
        (torch.float32, False, 0.001, 0.001),
        (torch.float16, False, 0.01, 0.01),
        # 复数张量（cfloat系列，核心测试场景，保持角度、模长为1）
        (torch.complex64, True, 0.001, 0.001),
    ]

    for dtype, is_complex, rtol, atol in tensor_type_groups:
        device = "cuda"  # 测试CUDA设备上的功能

        # 生成随机张量形状的辅助函数（合理范围，避免显存溢出）
        def generate_random_size():
            return random.randint(1, 256)

        # 生成多种张量形状：1D/2D/3D，覆盖常见使用场景
        for shape_dim in (1, 2, 3):
            if shape_dim == 1:
                shape = (generate_random_size(),)
            elif shape_dim == 2:
                shape = (generate_random_size(), generate_random_size())
            else:
                shape = (generate_random_size(), generate_random_size(), generate_random_size())

            # 加入参数列表：张量形状、是否复数、数据类型、设备、误差阈值
            arguments.append((
                shape,
                is_complex,
                dtype,
                device,
                rtol,
                atol
            ))

    # 返回参数化的名称和参数列表（与测试函数的入参对应）
    return "shape, is_complex, dtype, device, rtol, atol", arguments


@skip_if_cuda_not_available  # 跳过CUDA不可用的环境
@pytest.mark.parametrize(*generate_arguments())  # 传入自动生成的测试参数
def test_sgn(shape, is_complex, dtype, device, rtol, atol):
    """测试ntops.torch.sgn与原生torch.sgn的输出一致性，覆盖实数和复数张量场景"""
    # 1. 生成符合要求的随机输入张量（区分实数和复数场景）
    if not is_complex:
        # 非复数张量：用randn生成正负分布的随机数（包含零附近的值）
        input_tensor = torch.randn(shape, dtype=dtype, device=device)
    else:
        # 复数张量：实部和虚部分别用randn生成，组合为复数张量
        # 兼容 cfloat32/cfloat16 类型，覆盖 torch.sgn() 的核心扩展场景
        real_part = torch.randn(shape, dtype=dtype.to_real(), device=device)
        imag_part = torch.randn(shape, dtype=dtype.to_real(), device=device)
        input_tensor = torch.complex(real_part, imag_part).to(dtype=dtype)

    # 2. 分别调用待测函数和原生参考函数执行sgn操作（默认不指定out参数）
    ninetoothed_output = ntops.torch.sgn(input_tensor)
    reference_output = torch.sgn(input_tensor)

    # 3. 断言两者输出结果在误差阈值范围内一致
    # 注意：复数张量的allclose会同时验证实部和虚部的一致性
    assert torch.allclose(
        ninetoothed_output,
        reference_output,
        rtol=rtol,
        atol=atol,
        equal_nan=False
    ), f"sgn测试失败：形状{shape}，{'复数' if is_complex else '非复数'}，数据类型{dtype}"

    # 可选扩展测试：指定out参数的场景（验证out参数的功能正确性）
    # 预先分配输出张量（与输入张量同形状、同类型、同设备）
    out_ninetoothed = torch.empty_like(input_tensor)
    out_reference = torch.empty_like(input_tensor)

    # 显式传入out参数调用函数
    ntops.torch.sgn(input_tensor, out=out_ninetoothed)
    torch.sgn(input_tensor, out=out_reference)

    # 断言out参数接收的结果也一致
    assert torch.allclose(
        out_ninetoothed,
        out_reference,
        rtol=rtol,
        atol=atol,
        equal_nan=False
    ), f"sgn(out参数)测试失败：形状{shape}，{'复数' if is_complex else '非复数'}，数据类型{dtype}"