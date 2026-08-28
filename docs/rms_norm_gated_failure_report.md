# `rms_norm_gated` Failure Analysis and Repair Report

Date: 2026-08-27

## Executive summary

`ntops.torch.rms_norm_gated` was unusable with the current NineToothed SSA/Triton
toolchain. Every CUDA execution path failed during kernel compilation; only the
Python-side argument-validation test passed. The public Python function, its
signature, defaults, accepted inputs, and mathematical behavior were not the
cause.

The failure came from a legacy private kernel layout and application style that
the current compiler no longer lowers correctly. The kernel was converted to a
supported row-vector reduction: one normalization group is handled by one
program, values are converted into distinct FP32 locals, and RMS is calculated
with a vector reduction. The public wrapper was left unchanged.

After the repair, all 27 focused tests pass across FP32, FP16, BF16, grouped and
ungrouped normalization, both gate orders, Swish and Sigmoid activation,
the no-gate path, validation, and a hidden dimension larger than the requested
block size.

## Failure symptoms

The original focused run produced:

```text
25 failed, 1 passed
```

The passing case only checked argument validation and did not launch a kernel.
The execution failures occurred before an output tensor could be computed.

The generated code exposed several related compiler errors:

```text
SyntaxError: range(0, (%0), 1)
NameError: ntl is not defined
AssertionError: loop-carried variable ... changed from scalar fp32 to vector fp32
NameError: sigmoid is not defined
```

## Root cause

The legacy implementation used two nested tile levels and Python loops over a
hierarchical tensor. That representation depended on behavior from an older
NineToothed lowering path.

With the current SSA path:

1. A symbolic hierarchy size was emitted directly into generated Python as
   `%0`, producing invalid `range` syntax.
2. `ntl.cast(value, ntl.float32)` was not recognized as the supported tensor
   cast form and leaked `ntl.float32` into generated Triton code.
3. Loop-based accumulation initialized a scalar but accumulated program-lane
   vectors, violating Triton's loop-carried type invariant.
4. `ntl.sigmoid` was emitted as an undefined bare `sigmoid` call.
5. Assigning casts back to kernel parameter names was interpreted as a store to
   the source parameter, causing FP16/BF16 values to be reloaded before `exp`.

These were private kernel/compiler compatibility problems, not numerical errors
in RMSNorm or gating.

## Implemented repair

The repair is confined to `src/ntops/kernels/rms_norm_gated.py` plus regression
coverage in `tests/test_rms_norm_gated.py`.

### Layout

- The last dimension is specialized in the internal symbolic tensor shape.
- Each normalization group is kept inside one Triton program.
- The ungrouped tile is at least the hidden size, so RMS is never incorrectly
  split across independently reduced programs when `hidden_size > block_size`.
- Grouped RMSNorm continues to flatten and tile by `group_size`.

### Computation

- Inputs, gates, and weights use the supported `.to(ntl.float32)` cast form.
- Cast results use distinct local names rather than overwriting parameters.
- Sum-of-squares is calculated with `ntl.sum` over the local group vector.
- Swish and Sigmoid use supported `ntl.exp` expressions.
- Gate-before-normalization, gate-after-normalization, and no-gate paths remain
  separately selected at kernel-construction time, without runtime branching.

### Compatibility guarantee

The following public interface remains unchanged:

```python
ntops.torch.rms_norm_gated(
    input,
    z=None,
    weight=None,
    eps=1e-5,
    group_size=None,
    norm_before_gate=False,
    activation="swish",
    block_size=128,
    num_warps=(1, 2, 4, 8),
    num_stages=(1, 2),
    max_num_configs=8,
)
```

An AST comparison against `HEAD` confirmed that the public function signature
is identical, and the wrapper file has no diff.

## Verification

The final focused verification was:

```bash
pytest -q tests/test_rms_norm_gated.py
```

Result:

```text
27 passed in 20.03s
```

Coverage includes:

- FP32, FP16, and BF16;
- Swish and Sigmoid;
- `norm_before_gate=True` and `False`;
- `group_size=None` and grouped normalization;
- missing gate and missing weight;
- invalid activation and group sizes;
- `hidden_size=257` with `block_size=128`.

Additional static checks passed:

```text
python -m py_compile ...
git diff --check
```

`ruff` was not installed in the environment, so its check could not be run.

## Unrelated test-suite state

The two neighboring MLA test files were also executed as a broader check. Their
16 tests currently fail in untouched code because the installed NineToothed SSA
frontend cannot promote `i64` and `index` values. This is independent of
`rms_norm_gated`; none of those failing paths import or execute the repaired
kernel.

## Follow-up considerations

- Re-run `benchmarks/bench_rms_norm_gated.py` before publishing performance
  claims. The kernel layout changed, so historical latency and autotuning
  results should not be treated as measurements of the repaired implementation.
- Fix the separate NineToothed `i64`/`index` promotion issue before using the
  neighboring MLA tests as a repository-wide release gate.
