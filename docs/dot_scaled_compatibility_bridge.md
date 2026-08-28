# Archived `dot_scaled` compatibility bridge

> Status: removed from `scaled_grouped_mm` on 2026-08-28.

The first W4A16 implementation temporarily patched private NineToothed SSA
builder and emitter methods so `ntl.dot_scaled` would generate
`tl.dot_scaled`. That approach is no longer used.

It was removed for three concrete reasons:

1. the installed stable legacy NineToothed package has no
   `ninetoothed.backends` module, so importing the bridge fails;
2. the DCU backend cannot rely on the same `dot_scaled` intrinsic and the old
   HIP test marker skipped every numerical launch;
3. modifying process-global private compiler methods, even under a lock, is not
   a stable operator-level contract.

The replacement decodes low/high MXFP4 nibbles in the generated kernel and uses
two ordinary BF16 dots per 32-element scale group. It depends only on common DSL
operations and has numerical tests enabled on both CUDA-compatible and HIP
platforms.

See [scaled_grouped_mm_implementation.md](scaled_grouped_mm_implementation.md)
for the current design, supported PyTorch subset, and validation procedure.
