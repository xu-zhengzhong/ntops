# ntops Block-scaled FP8 矩阵乘实现报告

> 日期：2026-08-28
>
> 当前验证平台：海光 DCU gfx936。天数平台保留同一公共 kernel 路径，尚需在
> 目标设备上完成端到端复验。

## 1. 接口与支持范围

公开入口与 PyTorch 主线 torch.nn.functional.scaled_mm 参数顺序对齐：

    ntops.torch.scaled_mm(
        mat_a,
        mat_b,
        scale_a,
        scale_recipe_a,
        scale_b,
        scale_recipe_b,
        swizzle_a=None,
        swizzle_b=None,
        bias=None,
        output_dtype=torch.bfloat16,
        contraction_dim=(),
        use_fast_accum=False,
    )

当前实现针对线性层、单专家 MoE GEMM 和 Attention Q/K/V/O 投影所需的
DeepSeek-style FP8 子集：

| 项目 | 支持值 |
| --- | --- |
| mat_a / mat_b | 2D torch.float8_e4m3fn |
| activation scale | BlockWise1x128，FP32 |
| weight scale | BlockWise1x128 或 BlockWise128x128，FP32 |
| swizzle | None 或 NO_SWIZZLE |
| bias | 可选的 [N] FP16/BF16/FP32 |
| output | FP16、BF16 或 FP32；None 等价于 BF16 |
| fast accumulation | 关闭 |
| contraction dimensions | None、() 或 [] |

mat_a 使用 row-major contiguous 布局；mat_b 使用 weight.t() 产生的
column-major dense 布局。K 必须为正的 128 倍数，N 必须为 128 倍数。该约束
覆盖主流 Transformer projection hidden size，并避免 wrapper 在推理热路径中创建
padding/copy。

当前不支持 batched/grouped 输入。MoE 场景由 router 得到的每个专家连续 token
切片分别调用；需要单次 grouped launch 时应另行扩展 scaled_grouped_mm 的 FP8
recipe，而不改变本接口的 2D 语义。

## 2. Scale 布局

设 A[M,K] @ B[K,N]，KB=K/128，NB=N/128。

- A 的 BlockWise1x128 scale 逻辑 shape 为 [M, KB]；
- B 的 BlockWise1x128 scale 逻辑 shape 为 [N, KB]；
- B 的 BlockWise128x128 scale shape 为 [KB, NB]；
- 同时接受 cuBLAS/PyTorch 使用的 K-block padding：
  [round_up(KB, 4), NB]，padding 行不会参与计算。

PyTorch CUDA 测试对 1x128 scale 使用 column-major stride，对 128x128 scale
使用 K-block-major column-major stride。本实现同时接受 dense row-major 和
column-major scale，不在 wrapper 中强制复制。

第 p 个 128-K block 的数学贡献为：

    C[m, n] += dot_fp32(A_fp8[m, p], B_fp8[p, n])
                 * scale_a[m, p] * scale_b[p, n_block_or_column]

所有 partial result 和 bias addition 均使用 FP32，最后一次性转换到输出 dtype。

## 3. 跨平台 kernel 设计

### 3.1 只使用公共 lowering 子集

实现不调用 tl.dot_scaled，不 patch NineToothed 私有 SSA/emitter，也不调用平台
专用 intrinsic。HIP kernel 使用 float8_e4m3fn 普通 ntl.dot；CoreX kernel
先把加载的 E4M3FN tile 精确转换为 BF16，再使用普通 BF16 ntl.dot。两条路径
都只使用普通 masked load/store、cast、乘法和 FP32 累加。

cast 使用 method-style .to(ntl.bfloat16) / .to(ntl.float32)。
generated source 测试会检查不存在未解析的 ntl. 和 dot_scaled。

### 3.2 分平台 dot 与 128-wide scale

海光 gfx936 上，16x128x128 以及 K tile 128 的 FP8 dot 会在 AMD LLVM codegen
阶段触发进程级崩溃。固定实现因此使用：

    BLOCK_M=16, BLOCK_N=16
    scale K block=128
    num_stages=1
    HIP BLOCK_K=32, num_warps=1
    CoreX BLOCK_K=16, num_warps=4

HIP 每四个 32-wide FP8 dot、CoreX 每八个 16-wide BF16 dot 使用同一个
128-K scale。E4M3FN 的所有有限值都能由 BF16 精确表示，因此 CoreX 转换不改变
输入值。MR-V100/Triton 3.1.0 的 32-wide FP8 dot 只累计前 16 个 K lane，
16-wide FP8 dot 则破坏 M/N tile lane 布局，因此原生 FP8 dot 不能作为该平台的
正确性路径。上述 tile 已覆盖 M=1、M 尾块、多 K block 和多个 N block。未把进程
级失败的候选加入 autotuner，避免一次新 shape 请求终止服务进程。

### 3.3 无分配 view 映射

权重的每个 128-N block 被映射为一个逻辑 group：

    B[K,N] -> B_blocks[NB,K,128]
    output[M,N] -> output_blocks[NB,M,128]

A 和 A scale 通过 stride-0 view 在 N block 间共享；B、B scale、bias 和 output
均只做 reshape/transpose/expand view。wrapper 不反量化 FP8、不展开 block
scale，也不复制 column-major weight，适合 decode、prefill 和专家 projection
热路径。

## 4. 验证

定向测试：

    PYTHONPATH=src pytest -q tests/test_scaled_mm.py

CoreX MR-V100、PyTorch 2.7.1、Triton 3.1.0 的结果为 15 passed。
海光 BW/gfx936、PyTorch 2.9.0、HIP 6.3.26093 的既有结果为 14 passed；该结果
记录于新增完整 K/N lane 回归测试之前，本次未在海光设备复验，但保留的 HIP
K=32 FP8 路径未修改。

覆盖内容：

- BlockWise1x128 x BlockWise1x128；
- 主要推理路径 BlockWise1x128 x BlockWise128x128；
- M 尾块、两个 K scale block、两个 N block；
- 单 token Attention decode；
- bias 融合、BF16 与 FP32 output；
- PyTorch 单值/list API 形式；
- dtype、shape、layout、recipe、swizzle 和 fast-accum 拒绝路径；
- generated source 中只有普通 FP8/BF16 dot，无 dot_scaled 和 ntl. 泄漏。

PyTorch reference 独立地把 FP8 转为 FP32，按 1x128/128x128 scale block
反量化后执行 FP32 matmul，并在最后转换到目标 dtype。

主要推理 shape 的性能入口为：

    PYTHONPATH=src python benchmarks/bench_scaled_mm.py

benchmark 覆盖单 token Attention、MoE 单专家和 linear prefill。性能数字必须在
海光与天数分别记录，不能把一端配置或加速比外推到另一端。

本次海光固定配置实测如下，时间为共享 benchmark_mean 方法的 mean latency：

| 场景 | M | N | K | mean latency | TFLOP/s | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention decode | 1 | 4096 | 4096 | 648.409 us | 0.052 | 0.000000 |
| MoE expert | 32 | 14336 | 4096 | 1185.789 us | 3.169 | 0.000977 |
| Linear prefill | 128 | 4096 | 4096 | 1223.433 us | 3.511 | 0.031250 |

这些数字用于记录当前稳定基线，不代表天数结果，也不与原生硬件 block-scaled
kernel 声称加速比。当前海光 Triton 对更宽 N tile 和更多 wave 的候选会在 AMD
LLVM 阶段进程级崩溃，因此未以不稳定候选换取表面吞吐。

## 5. 已知限制与天数复验项

- 天数当前仅完成公共 source 路径设计，尚未报告设备数值和性能结果；
- float8_e4m3fnuz 在当前海光 Triton 上无法合法转换，不作为公开支持 dtype；
- 暂不支持 A 侧 BlockWise128x128、swizzled scale、多级 scale、batch 和
  use_fast_accum=True；
- N 尾块暂不接受，避免为 128-N weight scale 引入 padding output；
- 当前固定保守配置以跨平台稳定为优先，完成天数复验后再以独立子进程筛选候选。

天数验收至少需要：相同 commit 的完整定向测试、三个 benchmark shape、实际
NineToothed/PyTorch/Triton 路径与版本、generated source 检查和最佳配置记录。

## 6. 上游参考

- PyTorch public API：
  <https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_mm.html>
- PyTorch functional implementation：
  <https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py>
- PyTorch block-scaled tests：
  <https://github.com/pytorch/pytorch/blob/main/test/test_scaled_matmul_cuda.py>
