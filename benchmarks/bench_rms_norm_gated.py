import statistics

import torch
import torch.nn.functional as F

import ntops

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN_SIZE = 128
LOCAL_VALUE_HEADS = 8  # Qwen3-Next: 32 value heads with TP=4.
EPS = 1e-5
AUTOTUNE_WARPS = (1, 2, 4, 8)
AUTOTUNE_STAGES = (1, 2)


def pytorch_reference(input, z, weight):
    x = input.float()
    z = z.float()
    weight = weight.float()
    gate = F.silu(z)
    output = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS)
    output = output * weight
    output = output * gate
    return output.to(input.dtype)


def measure(function, warmup=50, repeats=30, iterations=10):
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()

        samples_ms = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                function()
            end.record()
            end.synchronize()
            samples_ms.append(start.elapsed_time(end) / iterations)

    samples_ms.sort()
    p50 = statistics.median(samples_ms)
    p90 = samples_ms[int(0.9 * (len(samples_ms) - 1))]
    return p50, p90


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print(f"torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name()}")
    print(
        "scenario,tokens,rows,hidden,dtype,ntops_p50_us,ntops_p90_us,"
        "optimized_p50_us,optimized_p90_us,autotuned_p50_us,autotuned_p90_us,"
        "pytorch_p50_us,pytorch_p90_us,optimized_speedup_vs_ntops,"
        "optimized_speedup_vs_pytorch,autotuned_speedup_vs_optimized,"
        "autotuned_speedup_vs_ntops,autotuned_speedup_vs_pytorch,"
        "max_abs_error_ntops,max_abs_error_optimized,max_abs_error_autotuned"
    )

    # The vLLM Qwen3-Next recipe uses 2048-token random prompts and a
    # maximum concurrency of 10. Decode is also represented by one token.
    scenarios = (
        ("decode", 1),
        ("concurrent_decode", 10),
        ("prefill_2048", 2048),
    )

    for name, num_tokens in scenarios:
        rows = num_tokens * LOCAL_VALUE_HEADS
        input = torch.randn(
            (rows, HIDDEN_SIZE), device=DEVICE, dtype=DTYPE
        )
        z = torch.randn_like(input)
        weight = torch.randn(
            HIDDEN_SIZE, device=DEVICE, dtype=torch.float32
        )

        def ntops_function():
            return ntops.torch.rms_norm_gated(
                input,
                z,
                weight,
                eps=EPS,
                group_size=None,
                norm_before_gate=True,
                activation="silu",
            )

        def pytorch_function():
            return pytorch_reference(input, z, weight)

        def optimized_function():
            return ntops.torch.rms_norm_gated_optimized(
                input,
                z,
                weight,
                eps=EPS,
                group_size=None,
                norm_before_gate=True,
                activation="silu",
            )

        def autotuned_function():
            return ntops.torch.rms_norm_gated_optimized(
                input,
                z,
                weight,
                eps=EPS,
                group_size=None,
                norm_before_gate=True,
                activation="silu",
                num_warps=AUTOTUNE_WARPS,
                num_stages=AUTOTUNE_STAGES,
            )

        with torch.inference_mode():
            ntops_output = ntops_function()
            optimized_output = optimized_function()
            autotuned_output = autotuned_function()
            pytorch_output = pytorch_function()
            torch.cuda.synchronize()
            max_abs_error_ntops = (
                (ntops_output.float() - pytorch_output.float())
                .abs()
                .max()
                .item()
            )
            max_abs_error_optimized = (
                (optimized_output.float() - pytorch_output.float())
                .abs()
                .max()
                .item()
            )
            max_abs_error_autotuned = (
                (autotuned_output.float() - pytorch_output.float())
                .abs()
                .max()
                .item()
            )

        ntops_p50, ntops_p90 = measure(ntops_function)
        optimized_p50, optimized_p90 = measure(optimized_function)
        autotuned_p50, autotuned_p90 = measure(autotuned_function)
        pytorch_p50, pytorch_p90 = measure(pytorch_function)
        optimized_speedup_vs_ntops = ntops_p50 / optimized_p50
        optimized_speedup_vs_pytorch = pytorch_p50 / optimized_p50
        autotuned_speedup_vs_optimized = optimized_p50 / autotuned_p50
        autotuned_speedup_vs_ntops = ntops_p50 / autotuned_p50
        autotuned_speedup_vs_pytorch = pytorch_p50 / autotuned_p50
        print(
            f"{name},{num_tokens},{rows},{HIDDEN_SIZE},{DTYPE},"
            f"{ntops_p50 * 1000:.3f},{ntops_p90 * 1000:.3f},"
            f"{optimized_p50 * 1000:.3f},{optimized_p90 * 1000:.3f},"
            f"{autotuned_p50 * 1000:.3f},{autotuned_p90 * 1000:.3f},"
            f"{pytorch_p50 * 1000:.3f},{pytorch_p90 * 1000:.3f},"
            f"{optimized_speedup_vs_ntops:.2f},{optimized_speedup_vs_pytorch:.2f},"
            f"{autotuned_speedup_vs_optimized:.2f},{autotuned_speedup_vs_ntops:.2f},"
            f"{autotuned_speedup_vs_pytorch:.2f},"
            f"{max_abs_error_ntops:.6f},{max_abs_error_optimized:.6f},"
            f"{max_abs_error_autotuned:.6f}"
        )


if __name__ == "__main__":
    main()
