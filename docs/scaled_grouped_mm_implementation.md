# ntops `scaled_grouped_mm` (MXFP4 W4A16) technical report

> Status: portable implementation passes end-to-end tests on Iluvatar; the same
> revision still requires a fresh DCU run.
>
> Date: 2026-08-28

## 1. Conclusion

The first implementation did not satisfy the two-platform requirement. It used
`ntl.dot_scaled`, patched private NineToothed SSA classes at runtime, and skipped
all four numerical tests whenever `torch.version.hip` was set. Therefore the DCU
result `10 passed, 4 skipped` only proved Python validation and source generation;
it did not launch the grouped GEMM kernel.

The current implementation removes that private compiler patch and avoids
`tl.dot_scaled`. It decodes packed MXFP4 values inside the kernel. CoreX uses
ordinary BF16 `dot` operations; HIP uses explicit FP32 multiply reductions so
that its generated IR contains no AMD BF16 MMAC operation. The numerical tests
now run on HIP/DCU instead of being skipped.

This operator matches the public argument order of PyTorch
`torch.nn.functional.scaled_grouped_mm`, but deliberately implements its W4A16
MoE subset rather than every scaling recipe supported by upstream PyTorch.

## 2. Supported semantics

### 2.1 Public signature

```python
scaled_grouped_mm(
    mat_a,
    mat_b,
    scale_a,
    scale_recipe_a,
    scale_b,
    scale_recipe_b,
    swizzle_a=None,
    swizzle_b=None,
    bias=None,
    offs=None,
    output_dtype=torch.bfloat16,
    contraction_dim=(),
    use_fast_accum=False,
)
```

Supported arguments are:

| Item | Supported value |
| --- | --- |
| activation | contiguous BF16 |
| weight | contiguous packed MXFP4, exposed as `uint8` or `float4_e2m1fn_x2` |
| activation scale | `scale_a=None`, `scale_recipe_a=None` |
| weight scale | one E8M0 tensor, directly or in a one-element list |
| weight recipe | `BlockWise1x32`, directly or in a one-element list |
| output | BF16; `output_dtype=None` also selects BF16 |
| bias | not supported |
| swizzle | `None` or `NO_SWIZZLE` |
| contraction dimensions | `None`, `()` or `[]` |
| fast accumulation | disabled |

Unsupported combinations fail explicitly. They are not silently interpreted as
W4A16.

### 2.2 Grouped-M modes

The kernel supports the two expert layouts needed by MoE inference:

| Mode | `mat_a` | `mat_b` storage | `scale_b` | Output |
| --- | --- | --- | --- | --- |
| uniform | `[G, M, K]` | `[G, K/2, N]` | `[G, K/32, N]` | `[G, M, N]` |
| routed | `[total_M, K]` | `[G, K/2, N]` | `[G, K/32, N]` | `[total_M, N]` |

For routed mode, `offs[g]` is the cumulative row end of expert `g`.
Nondecreasing offsets are allowed, so equal adjacent values represent a zero-token
expert. `offs` is contiguous device `int32`, and `offs[-1] == total_M`.

The implementation does not provide PyTorch's grouped-K 2D-by-2D mode. That is a
different use of `offs` and is outside this W4A16 expert kernel.

## 3. MXFP4 encoding

Each byte stores two E2M1 values. The even K element is in the low nibble and the
odd K element is in the high nibble, matching vLLM's portable MXFP4 quantizer:

```text
packed = even_code | (odd_code << 4)
```

For a nibble `c`:

```text
sign = -1 when bit 3 is set, otherwise +1
exponent = (c & 0x7) >> 1
mantissa = c & 1

abs(c) = 0.5 * mantissa                         when exponent == 0
abs(c) = (1 + 0.5 * mantissa) * 2^(exponent-1) otherwise
```

The representable magnitudes are `0, 0.5, 1, 1.5, 2, 3, 4, 6`. An E8M0 scale byte
`s` decodes to `2^(s-127)`. One scale is applied to each 32-element logical K
block and each output column.

Consequently, the mathematical operation for expert `g` is:

\[
C_g = \operatorname{BF16}\left(
  \operatorname{FP32Accumulate}(A_g \times \operatorname{dequant}(B_g,S_g))
\right).
\]

The kernel consumes vLLM-compatible nibble and E8M0 values. Its matrix storage is
the logical `mat_b` orientation required by the PyTorch interface; callers holding
vLLM weights as `[G, N, K/2]` must provide the corresponding contiguous
`[G, K/2, N]` representation.

## 4. Portable kernel design

### 4.1 Even/odd K decomposition

Triton `dot` accepts BF16 blocks but not packed nibbles on every backend. A
32-element MX scale group is therefore split into two 16-element K matrices:

```text
A_even = A[..., 0, 2, ..., 30]
A_odd  = A[..., 1, 3, ..., 31]
B_even = decode(low_nibble(B_packed)) * scale
B_odd  = decode(high_nibble(B_packed)) * scale

CoreX:
  acc += dot(A_even, B_even)
  acc += dot(A_odd,  B_odd)

HIP:
  acc += sum(A_even[:, :, None] * B_even[None, :, :], axis=1)
  acc += sum(A_odd[:, :, None] * B_odd[None, :, :], axis=1)
```

The public operator selects one of two single-kernel schedules without changing
its mathematical interface. CoreX retains the faster block-dot schedule. HIP
uses a branchless integer E2M1 decoder followed by explicit FP32 reductions.
Neither path materializes the full dequantized `[G, K, N]` weight.

The separate HIP schedule is required for DCU compiler stability. The SSA
emitter expands elementwise values inside a block-dot operand, producing a large
Triton AST whose LLIR contains two
`llvm.amdgcn.mmac.f32.16x16x16bf16` calls. AMD `make_amdgcn` segfaulted even after
reducing the tile to `16x16` and the launch to one wave.

An intermediate table-lookup decoder compiled without MMAC, but was not
numerically valid: a tensor-valued `.source[...]` index was typed as a scalar by
the SSA frontend, so the reduction coordinate never reached the packed-weight
load. A direct arithmetic revision exposed the same coordinate-loss behavior in
`where` predicates. The current decoder therefore uses only integer bitwise and
arithmetic operations. It constructs twice the E2M1 magnitude exactly, applies
the sign as `1 - 2 * sign_bit`, converts once to FP32, and multiplies by `0.5`.

The resulting HIP IR contains no gather, `where`, `dot`, or MMAC operation and
uses two explicit FP32 reductions. Weights are rounded to BF16 before conversion
back to FP32 for accumulation, preserving the reference dequantization semantics.

### 4.2 Arrangement

The activation arguments passed to the kernel are `mat_a[..., :-1]` and
`mat_a[..., 1:]`. Both have logical K length `K-1`. A dilation-2 tile of width 16
then produces exactly `K/32` blocks without floor-tiling the M dimension. This is
important: applying `floor_mode` to the complete 3D tile drops the final M tile
and can leave output rows or experts unwritten.

The current platform configurations are fixed:

```text
CoreX: BLOCK_M=16, BLOCK_N=64, num_warps=4, num_stages=1
HIP:   BLOCK_M=1,  BLOCK_N=16, num_warps=1, num_stages=1
```

The CoreX arithmetic configuration was selected from a bounded
fixed-configuration comparison. HIP combines the branchless decoder with a
single output row and a small N tile to keep the explicit reduction bounded.
Autotuning is intentionally not part of this revision: a candidate
that crashes an AMD compiler process cannot be caught by the Python autotuner.
Platform-specific tuning should be added only after both backends have a known
correct fixed configuration.

The HIP descriptors additionally mark dense G/K/N dimensions as compile-time
specializations. Triton compiles and caches a specialization for each encountered
shape, allowing repeated shape predicates to fold before AMD LLVM code generation.
Routed M dimensions deliberately remain runtime values. Making the per-expert row
count constexpr specializes it to the maximum sequence length, which lets padding
programs for shorter experts overwrite later expert rows.

The HIP tile uses one 64-thread wave and does not emit dot-operand layout
conversions or `llvm.amdgcn.mmac` calls. CoreX retains its independently measured
four-warp configuration.

On Iluvatar, the MoE-shaped screening case `G=8, M=16, K=4096, N=4096` produced:

| `BLOCK_M` | `BLOCK_N` | warps | mean latency |
| ---: | ---: | ---: | ---: |
| 16 | 16 | 4 | 3.3159 ms |
| 16 | 32 | 4 | 1.9367 ms |
| 16 | 64 | 4 | 1.3689 ms |
| 32 | 32 | 4 | 3.7424 ms |
| 32 | 64 | 4 | 2.2697 ms |

These arithmetic-decoder numbers are retained as the tile-selection record. A
fresh run of the current dispatched CoreX path on the same shape measured
`1.4627 ms` mean (`warmup=25`, `rep=100`). An experimental unconditional
table-lookup path with `BLOCK_N=64` measured `4.2387 ms`, so the DCU compiler
workaround is intentionally not applied to CoreX. HIP uses `1x16` until its
compile/run stability and performance have been measured on the target device.

### 4.3 Removed private compiler dependency

The old wrapper imported and replaced these private symbols:

```text
ninetoothed.frontend.python._ApplicationSSABuilder._lower_linalg_call
ninetoothed.backends.emitters.ssa._emit_linalg_dot
```

That failed on the installed stable legacy compiler because
`ninetoothed.backends` did not exist, and tied the operator to one internal SSA
revision. Both dispatched implementations use normal DSL operations already
used elsewhere in ntops: bitwise arithmetic, `exp2`, `zeros`, dtype conversion,
`dot` on CoreX, and `sum` on HIP.

One frontend-compatibility detail is also intentional:

- casts use method-style `.to(ntl.dtype)`: the DCU SSA emitter lowers this to
  `.to(tl.dtype)`, while `ntl.cast(value, ntl.dtype)` can leak an undefined
  runtime `ntl.dtype` value into generated Triton.

## 5. PyTorch alignment and limits

Upstream PyTorch `scaled_grouped_mm` is a general API. Current upstream tests also
cover FP8, MXFP8, MXFP4/MXFP4, NVFP4, grouped-K, grouped-M, scale swizzles, bias,
and several output modes. This ntops operator is not a complete replacement for
that entire API.

The alignment claim is specifically:

1. public positional argument order and option names match PyTorch;
2. routed 2D-by-3D `offs` uses cumulative expert row ends;
3. output and accumulation semantics match the supported W4A16 subset;
4. unsupported PyTorch modes raise explicit errors.

This distinction matters because the requested W4A16 path has an unquantized BF16
activation (`scale_a=None`), whereas PyTorch's current MXFP4 benchmark quantizes
both operands. vLLM's current SM100 MXFP4 grouped MoE kernel is likewise W4A4,
not the W4A16 operation implemented here. vLLM is used as the encoding reference,
not as a claim that the kernels have identical operand types.

## 6. Validation

### 6.1 Reference implementation

The tests independently unpack all nibbles, decode E2M1 and E8M0 in PyTorch,
materialize a BF16 reference weight, and use FP32 `torch.bmm`/`torch.mm`
accumulation before converting to BF16.

Coverage includes:

- uniform `G=2, M=17, K=96, N=19`, exercising M/N tails;
- routed rows `(4, 0, 7)`, exercising a zero-token expert;
- all 16 E2M1 codes in both nibble positions using an identity activation;
- three independent K scale blocks;
- raw byte storage and native packed dtypes when the installed PyTorch has them;
- dtype, shape, recipe, option, and offset failures;
- equivalent one-element list/default API forms and an all-zero-token return;
- generated CoreX Python source containing exactly two ordinary dots and no
  `dot_scaled`;
- generated HIP reduction source containing no dot/MMAC input, exactly two FP32
  reductions, no `tl.where`, no decode-table arguments, and no unresolved
  `ntl.` namespace.

### 6.2 Iluvatar result

Environment:

| Item | Value |
| --- | --- |
| device | Iluvatar MR-V100 |
| Python | 3.12.3 |
| PyTorch | 2.7.1+corex.4.4.0 |
| Triton | 3.1.0+corex.4.4.0 |
| NineToothed | 0.26.0 installed stable package |

Command:

```bash
PYTHONPATH=src pytest -q tests/test_scaled_grouped_mm.py
```

Expected result for this environment is `15 passed, 2 skipped`. The skipped
cases are only the native `float4_e2m1fn_x2`/`float8_e8m0fnu` dtype variants,
because PyTorch 2.7.1 does not define the former. The raw-byte uniform, routed,
and exhaustive encoding tests all launch the accelerator kernel.

### 6.3 DCU verification command

```bash
git pull
pytest -q tests/test_scaled_grouped_mm.py
```

On DCU, the old `10 passed, 4 skipped` output is no longer sufficient. At least
the raw-byte uniform, routed, and exhaustive encoding tests must report `PASSED`.
If native packed dtypes are absent, exactly two dtype variants may still be
skipped.

No DCU performance result is reported until this exact portable revision has
completed the numerical suite. Likewise, no speedup is claimed against PyTorch's
native operator because the local PyTorch build does not provide the same W4A16
path; a dequantize-plus-BF16-matmul reference is a correctness baseline, not a
hardware-equivalent performance baseline.

## 7. Remaining work

- Run this revision on DCU and archive the complete output.
- After correctness, benchmark several fixed `BLOCK_M/BLOCK_N/warps` candidates
  independently on both platforms; do not use an autotune set containing a
  backend-crashing candidate.
- Add an explicit benchmark against end-to-end PyTorch dequantization plus grouped
  BF16 matmul, clearly including dequantization time.
- Consider accepting native scale swizzles and common `[G, N, K/2]` weight storage
  in a separate extension without changing current semantics.
- Add a native backend path only behind capability dispatch; retain the CoreX
  pair-of-dots path and the HIP reduction path as correctness fallbacks.

## 8. Upstream references

- PyTorch public interface:
  <https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_grouped_mm.html>
- PyTorch implementation:
  <https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py>
- vLLM MXFP4 quantization utilities:
  <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py>
- vLLM MXFP4 grouped MoE test:
  <https://github.com/vllm-project/vllm/blob/main/tests/kernels/quantization/test_mxfp4_moe.py>
