"""Shared Triton benchmark and autotune reporting helpers."""

import torch
from triton.testing import do_bench


def benchmark_mean(function):
    """Return mean latency in microseconds using Triton's benchmark policy."""
    with torch.inference_mode():
        return do_bench(
            function,
            warmup=25,
            rep=100,
            return_mode="mean",
        ) * 1000.0


def selected_config(kernel):
    """Return the most recently cached ``(num_warps, num_stages)`` pair."""
    globals_ = kernel._kernel.__globals__
    autotuners = (
        value
        for name, value in globals_.items()
        if name.endswith("_with_auto_tuning") and hasattr(value, "cache")
    )
    autotuner = next(autotuners, None)
    if autotuner is None or not autotuner.cache:
        raise RuntimeError("the kernel has not completed autotuning")
    config = next(reversed(autotuner.cache.values()))
    return config.num_warps, config.num_stages
