"""Benchmark BlockWise1x128 x BlockWise128x128 FP8 projection shapes."""

import torch
from _benchmark import benchmark_mean

import ntops


def _column_major(value):
    return value.t().contiguous().t()


def _make_inputs(m, n, k):
    mat_a = (
        torch.randn((m, k), device="cuda")
        .clamp(-3, 3)
        .to(torch.float8_e4m3fn)
    )
    weight = (
        torch.randn((n, k), device="cuda")
        .clamp(-3, 3)
        .to(torch.float8_e4m3fn)
    )
    k_blocks = k // 128
    padded_k_blocks = ((k_blocks + 3) // 4) * 4
    scale_a = _column_major(
        torch.ones((m, k_blocks), device="cuda", dtype=torch.float32)
    )
    scale_b = _column_major(
        torch.ones(
            (padded_k_blocks, n // 128),
            device="cuda",
            dtype=torch.float32,
        )
    )
    return mat_a, weight.t(), scale_a, scale_b


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-compatible accelerator is required")

    print(f"torch={torch.__version__}")
    print(f"hip={torch.version.hip}")
    print(f"device={torch.cuda.get_device_name()}")
    print("scenario,M,N,K,mean_us,tflops,max_abs_error")
    scenarios = (
        ("attention_decode", 1, 4096, 4096),
        ("moe_expert", 32, 14336, 4096),
        ("linear_prefill", 128, 4096, 4096),
    )
    for name, m, n, k in scenarios:
        mat_a, mat_b, scale_a, scale_b = _make_inputs(m, n, k)

        def run():
            return ntops.torch.scaled_mm(
                mat_a,
                mat_b,
                scale_a,
                ntops.torch.ScalingType.BlockWise1x128,
                scale_b,
                ntops.torch.ScalingType.BlockWise128x128,
            )

        with torch.inference_mode():
            output = run()
            reference = (mat_a.float() @ mat_b.float()).to(torch.bfloat16)
            max_abs_error = (
                (output.float() - reference.float()).abs().max().item()
            )
            torch.cuda.synchronize()

        mean_us = benchmark_mean(run)
        tflops = 2.0 * m * n * k / (mean_us * 1.0e6)
        print(
            f"{name},{m},{n},{k},{mean_us:.3f},{tflops:.3f},"
            f"{max_abs_error:.6f}"
        )


if __name__ == "__main__":
    main()
