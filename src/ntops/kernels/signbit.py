import functools

from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    # signbit(x) should be True for:
    # 1) x < 0
    # 2) x is -0.0 (important: -0.0 has sign bit set but x < 0 is False)
    neg = input < 0

    is_zero = input == 0
    neg_zero = is_zero & ((1 / input) == float("-inf"))

    # output = neg | neg_zero  # noqa: F841
    output = output.ndim()


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors