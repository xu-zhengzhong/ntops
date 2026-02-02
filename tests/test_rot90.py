import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def generate_arguments():
    arguments = []

    # 遍历支持的张量数据类型
    for dtype in (torch.float32, torch.float16):
        device = "cuda"  # 测试CUDA设备上的功能

        # 针对不同精度设置合理的误差阈值（和原代码保持一致）
        if dtype is torch.float32:
            atol = 0.001
            rtol = 0.001
        else:
            atol = 0.01
            rtol = 0.01

        # 生成随机张量形状的辅助函数（保证形状在合理范围内，避免显存溢出）
        def generate_random_size():
            return random.randint(1, 256)  # 适当缩小范围，适配rot90多维度场景

        # rot90的关键参数1：旋转次数（k=1,2,3对应90/180/270度逆时针旋转）
        for rot_k in (1, 2, 3):
            # rot90的关键参数2：旋转维度（dims），覆盖常见场景
            # 注意：dims必须是两个不同的整数，且符合张量维度范围
            for rot_dims in ((-2, -1), (0, 1)):  # 默认最后两个维度、前两个维度
                # 生成4维张量形状 [b, c, h, w]，兼容大部分场景
                b = generate_random_size()
                c = generate_random_size()
                h = generate_random_size()
                w = generate_random_size()

                # 场景1：2D张量 [h, w]（基础场景，仅支持(0,1)或(-2,-1)，效果一致）
                arguments.append((
                    (h, w),       # 张量形状
                    rot_k,        # 旋转次数
                    rot_dims,     # 旋转维度
                    dtype,        # 数据类型
                    device,       # 设备
                    rtol,         # 相对误差阈值
                    atol          # 绝对误差阈值
                ))

                # 场景2：4D张量 [b, c, h, w]（实际业务常用场景）
                arguments.append((
                    (b, c, h, w), # 张量形状
                    rot_k,        # 旋转次数
                    rot_dims,     # 旋转维度
                    dtype,        # 数据类型
                    device,       # 设备
                    rtol,         # 相对误差阈值
                    atol          # 绝对误差阈值
                ))

    # 返回参数化的名称和参数列表（与测试函数的入参对应，新增rot_dims）
    return "shape, rot_k, rot_dims, dtype, device, rtol, atol", arguments


@skip_if_cuda_not_available  # 跳过CUDA不可用的环境
@pytest.mark.parametrize(*generate_arguments())  # 传入自动生成的测试参数
def test_rot90(shape, rot_k, rot_dims, dtype, device, rtol, atol):
    """测试ntops.torch.rot90与原生torch.rot90的输出一致性"""
    # 1. 生成随机输入张量（符合指定形状、数据类型和设备）
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    # 2. 分别调用待测函数和原生参考函数执行rot90操作
    # 关键修正：为ntops.torch.rot90显式传入rot_dims参数，与原生函数对齐
    ninetoothed_output = ntops.torch.rot90(input_tensor, k=rot_k, dims=rot_dims)
    reference_output = torch.rot90(input_tensor, k=rot_k, dims=rot_dims)

    # 3. 断言两者输出结果在误差阈值范围内一致
    assert torch.allclose(
        ninetoothed_output,
        reference_output,
        rtol=rtol,
        atol=atol,
    ), f"rot90测试失败：形状{shape}，旋转次数{rot_k}，旋转维度{rot_dims}，数据类型{dtype}"

# dims = (0, 1)
# k = 3
# ndim = 4
# dtype = torch.float32
# device = torch.device("cuda")

# x = torch.arange(138*191*1*229, dtype=dtype, device=device).view(138, 191, 1, 229)
# reference = torch.rot90(x, k=k, dims=dims)
# y = ntops.torch.rot90(x, k=k, dims=dims)

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