import math
import random

import torch


def generate_arguments(use_float=True, use_complex=False):
    arguments = []
    dtype_arr = []

    # 1. 保留原有逻辑：根据 use_float 添加浮点型/布尔整型
    if use_complex:
        # 优先级最高：如果指定 use_complex=True，仅生成复数类型（满足当前测试需求）
        dtype_arr = (torch.complex64, torch.complex128)
    else:
        # 保留原有的非复数类型逻辑
        if use_float:
            dtype_arr = (torch.float32, torch.float16)
        else:
            dtype_arr = (torch.bool, torch.int8, torch.int16, torch.int32)

    for ndim in range(1, 5):
        for dtype in dtype_arr:
            device = "cuda"

            # 2. 补充复数类型的误差容限，同时保留原有类型的容限逻辑
            if dtype is torch.float32:
                atol = 0.001
                rtol = 0.001
            elif dtype is torch.complex64:
                # complex64 对应 float32 精度，沿用 float32 的容限
                atol = 0.001
                rtol = 0.001
            elif dtype is torch.complex128:
                # complex128 对应 float64 精度，设置更高的精度容限
                atol = 1e-08
                rtol = 1e-05
            else:
                # 其他类型（float16、布尔、整型）沿用原有容限
                atol = 0.01
                rtol = 0.01

            arguments.append((_random_shape(ndim), dtype, device, rtol, atol))

    return "shape, dtype, device, rtol, atol", arguments


def gauss(mu=0.0, sigma=1.0):
    return random.gauss(mu, sigma)


def _random_shape(ndim, min_num_elements=2**8, max_num_elements=2**10):
    num_elements = random.randint(min_num_elements, max_num_elements)

    shape = []
    remaining = num_elements

    for _ in range(ndim - 1):
        size = random.randint(1, max(1, math.isqrt(remaining)))
        shape.append(size)
        remaining //= size

    shape.append(remaining)
    random.shuffle(shape)

    return shape
