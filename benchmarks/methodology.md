# Benchmark methodology

1. **Correctness first.** `check()` against a trusted reference (PyTorch / NumPy / cuBLAS) before timing.
2. **Warm-up.** Default 10 runs to exclude JIT compilation, allocation, and clock ramp-up.
3. **Timing.** CUDA events around each launch, one pair per iteration, then synchronize.
4. **Cache state.** L2 is flushed before each timed run by default (cold-cache numbers). Pass `l2_flush=False` when warm-cache behaviour is what matters, and say so in `notes`.
5. **Repetition.** Default 100 iterations. Report median plus p10/p90 and stdev, never a single best run.
6. **Fixed inputs.** Shapes, dtypes, layouts, and strides are recorded per row. Compare only equivalent precision and functionality.
7. **Environment.** GPU, compute capability, driver, CUDA, torch, triton, and nvcc are recorded automatically.
8. **Quiet GPU.** Check `nvidia-smi` for other processes before benchmarking. If clocks are locked or unlocked, note it.
9. **Derived metrics.** Pass `bytes_moved` (compulsory traffic) or `flops` to get GB/s or TFLOP/s. Compare against the GPU's peak.
10. **Launch overhead.** For tiny kernels, the measured time may be mostly launch overhead. Note this, or time a batch of launches / a CUDA graph.
