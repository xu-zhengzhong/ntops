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

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments(use_complex=True))
def test_sgn(shape, dtype, device, rtol, atol):
    # 生成复数类型的随机张量（匹配 dtype 精度）
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real_part = torch.randn(shape, dtype=real_dtype, device=device)
    imag_part = torch.randn(shape, dtype=real_dtype, device=device)
    input = torch.complex(real_part, imag_part)
    
    # 验证自定义 sgn 函数与原生 torch.sgn 输出一致性
    ninetoothed_output = ntops.torch.sgn(input)
    reference_output = torch.sgn(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)