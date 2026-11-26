# Plan — locked roadmap

15 weeks × (5 core sessions + 1 Saturday buffer/review) = **90 sessions**, plus a
3-session Week 0. Destination is locked; only the next task adapts.
Rules: [RULEBOOK.md](RULEBOOK.md).

Legend: 🔧 build · 📖 read/trace · 🧪 measure/profile · ⭐ heavy day — adapt existing
code, finish on Saturday if needed.

Every Saturday: catch up → blank-page test → closed-book quiz → 10–15 pages of
philosophy → weekly review in `worklog.md`.

---

## Week 0 — Setup (3 sessions)
- [x] S1 🔧 Confirm approved GPU. Record `nvidia-smi`, `nvcc --version`, torch/triton versions.
- [x] S2 🔧 Verify `compute-sanitizer`, `nsys`, `ncu` all run (cloud instances often block counters). Run `python benchmarks/benchmark.py` smoke test.
- [x] S3 🔧 Push repo to GitHub. Pick the fixed daily slot. Do one practice `./log.sh`.

## Week 1 — Performance thinking & CUDA execution model
- [x] Mon 📖 Roofline (already known — skipped). Arithmetic intensity of vector add and GEMM by hand.
- [x] Tue 📖 Grids, blocks, threads, warps, SMs, SIMT, warp divergence. Draw a 1-D launch.
- [x] Wed 🔧 Vector-add kernel with bounds check; compare vs PyTorch; run `compute-sanitizer`.
- [ ] Thu 📖 CUDA C++ → PTX → SASS. Inspect output for vector add.
- [ ] Fri 🧪 Benchmark vector add with the harness; compute achieved bandwidth vs peak.
- [ ] Sat Review. Deliverable: roofline sheet, execution-model diagram, vector-add baseline.

## Week 2 — Memory hierarchy & coalescing
- [ ] Mon 📖 Registers, local, shared, L1, L2, HBM; coalescing and 32-byte sectors.
- [ ] Tue 🔧 Naive transpose. Predict read/write access patterns first.
- [ ] Wed 🔧 Shared-memory tiled transpose.
- [ ] Thu 🔧 Bank conflicts; padded tile; verify effect.
- [ ] Fri 🧪 First `nsys` trace; `ncu` on all 3 variants (sectors/request, bank conflicts, occupancy). Note why higher occupancy ≠ faster.
- [ ] Sat Review. Deliverable: 3 transpose versions + profiling note.

## Week 3 — Warp primitives & reductions
- [ ] Mon 📖 Warp shuffles (`__shfl_*_sync`), masks, warp-level reduction.
- [ ] Tue 🔧 Parallel sum: naive → shared memory → warp shuffle.
- [ ] Wed 🔧 Block + grid reduction (two-pass vs atomics). Test odd sizes.
- [ ] Thu 🔧 Row-wise softmax in CUDA (max + sum reductions).
- [ ] Fri 🔧 RMSNorm in CUDA (foreshadows the capstone). Benchmark vs PyTorch.
- [ ] Sat Review. Deliverable: reduction family + softmax + RMSNorm with tests.

## Week 4 — GEMM I: first principles
- [ ] Mon 📖 C[M,N]=A[M,K]·B[K,N]: indexing, FLOPs, compulsory traffic.
- [ ] Tue 🔧 Naive GEMM (one thread per output).
- [ ] Wed 🔧 Coalesced access; check row/column-major and strides explicitly.
- [ ] Thu 🔧 Shared-memory tiling (BM, BN, BK).
- [ ] Fri 🧪 Tail masks + sync; odd shapes; benchmark vs cuBLAS; first `ncu` on GEMM.
- [ ] Sat Review. Deliverable: naive + tiled GEMM + benchmark table.

## Week 5 — GEMM II: optimization worklog (Boehm-style)
- [ ] Mon ⭐🔧 1-D register tiling (multiple outputs per thread).
- [ ] Tue ⭐🔧 2-D register tiling; track accumulator/register footprint.
- [ ] Wed 🔧 Vectorized `float4` loads; verify in SASS; alignment + tails.
- [ ] Thu 📖 Warp tiling: map lanes/warps → output fragments.
- [ ] Fri 🧪 Bounded sweep of BM/BN/BK/threads, one family at a time; note wave quantization (odd M/N); write the worklog.
- [ ] Sat Review. Deliverable: GEMM kernel sequence + reproducible worklog.

## Week 6 — Profiling, debugging, trustworthy benchmarks
- [ ] Mon 📖 Nsight Systems vs Compute; triage workflow on a multi-kernel program.
- [ ] Tue 🧪 Read an `ncu` report top-to-bottom; source correlation with SASS.
- [ ] Wed 🔧 Sanitizer racecheck / initcheck / synccheck; plant and fix an OOB and a shared-memory race.
- [ ] Thu 🧪 Audit the harness: warm-up, sync, variance, L2 state, clocks, launch overhead.
- [ ] Fri 🧪 One slow kernel → bottleneck → one change → measured outcome.
- [ ] Sat Review. Deliverable: profiling/debugging checklist in `resources/`.

## Week 7 — Triton
- [ ] Mon 🔧 Vector add tutorial; `program_id`, `arange`, masks, grids.
- [ ] Tue 🔧 Fused softmax; block size effect. Compare with your Week 3 CUDA version.
- [ ] Wed ⭐🔧 Matmul tutorial; explain its tiling and grouped ordering.
- [ ] Thu 🧪 Autotuning: `num_warps`, `num_stages`, tile sizes (small search space).
- [ ] Fri 🔧 One fused op (bias + GELU) vs eager **and `torch.compile`**; register it via `torch.library`.
- [ ] Sat Review. Deliverable: 4 Triton kernels with tests + benchmarks.

## Week 8 — Tensor cores & numerical formats
- [ ] Mon 📖 FP32 / TF32 / FP16 / BF16 / FP8: range, precision, accumulation.
- [ ] Tue 📖 WMMA: fragments, tile constraints, accumulator types.
- [ ] Wed ⭐🔧 Adapt a WMMA GEMM; compare vs best CUDA-core GEMM.
- [ ] Thu 🔧 Numerical tests: abs/rel error, value distributions, awkward shapes. `mma.sync`/`ldmatrix`/swizzle conceptually.
- [ ] Fri 🧪 Profile tensor-core GEMM; compare vs cuBLAS at equivalent precision.
- [ ] Sat Review. Deliverable: tensor-core GEMM + numerical/perf comparison.

## Week 9 — Attention I: math & online softmax
- [ ] Mon 🔧 SDPA from first principles in PyTorch (QKᵀ, scale, mask, softmax, ·V).
- [ ] Tue 📖 Memory cost of materializing scores across sequence lengths.
- [ ] Wed 🔧 Stable softmax; adversarial logits.
- [ ] Thu 🔧 Derive + implement online softmax (running max, sum, rescale).
- [ ] Fri 🔧 Tiled attention reference (Q/K/V blocks, causal mask); validate vs FP64 reference incl. fully-masked row (no NaN), seq len 1, non-block-multiple lengths; draw dataflow.
- [ ] Sat Review. Deliverable: online-softmax reference + dataflow diagram.

## Week 10 — Attention II: FlashAttention
- [ ] Mon 📖 FlashAttention paper: IO complexity, SRAM/HBM traffic.
- [ ] Tue 📖 Forward pass → annotated pseudocode.
- [ ] Wed ⭐🔧 Triton FA forward, non-causal.
- [ ] Thu ⭐🔧 Add causal masking; full correctness tests (head_dim 64/80/96/128).
- [ ] Fri 🧪 FA2 reading (work partitioning) + benchmark vs PyTorch SDPA (**pin the backend** with `sdpa_kernel`) and FlexAttention across seq lens / head dims.
- [ ] Sat Review. Deliverable: correct FA prototype + FA1→FA2 note.

## Week 11 — Inference-serving kernels
- [ ] Mon 📖 Prefill vs decode: shapes and arithmetic intensity.
- [ ] Tue 📖 KV-cache layouts, paged/block allocation, decode traffic vs context length.
- [ ] Wed 📖 Trace a paged decode-attention kernel; FlashInfer paper.
- [ ] Thu 🧪 Launch overhead, streams, CUDA graphs on a small decode loop.
- [ ] Fri 🧪 Profile an inference microbenchmark; separate framework vs kernel time. Write the serving problem statement.
- [ ] Sat Review. **Start looking for one small open-source issue to contribute to.**

## Week 12 — Frontier literacy & AI-generated kernels (light, reading-heavy)
- [ ] Mon 🔧📖 `cp.async` double-buffering experiment (bridge from Week 4 tiling), then Hopper: TMA, WGMMA, async pipelines, warp specialization; FA3.
- [ ] Tue 📖 CuTe **Python DSL** (one deep path, not Gluon too): trace one GEMM (gmem → smem → MMA → store); verify whether FA4 lives in `flash_attn/cute`.
- [ ] Wed 🔧 Blackwell (tcgen05, TMEM) overview + NVFP4 quantize/dequantize reference; measure error.
- [ ] Thu 📖 KernelBench paper + harness; skim DeepGEMM or FlashMLA kernel path.
- [ ] Fri 🧪 Ask an LLM to optimize a known kernel; red-team it (shapes, strides, dtypes, tolerances, shortcuts).
- [ ] Sat Review. Deliverable: hardware note + AI-kernel evaluation report. (FA4 → reading list.)

## Weeks 13–15 — Capstone

**Scope (technically coherent fusions, not a forced single kernel):**
- **Kernel A:** residual add + RMSNorm + quantization (FP8 on sm89+; INT8 otherwise) — the pre-GEMM path.
- **Kernel B:** RoPE + KV-cache write + quantization — the post-QKV-projection path.
- **Benchmark:** both against the unfused PyTorch pipeline, for a fixed model shape (e.g., hidden 4096, head_dim 128).

### Week 13 — Define & build A
- [ ] Mon Freeze scope: GPU, shapes, dtypes, layouts, **RoPE convention** (half-split vs interleaved, position offsets), FP8 scaling (per-token / per-128-group, consumed by `torch._scaled_mm`), baselines (**eager + `torch.compile`**), metric → `capstone/README.md`.
- [ ] Tue Trusted references + full test matrix for A and B.
- [ ] Wed Simplest correct Kernel A (Triton or CUDA). No tuning.
- [ ] Thu Expand tests; sanitizer; fix every correctness issue.
- [ ] Fri Stable baseline for A vs unfused.
- [ ] Sat Review + capstone worklog.

### Week 14 — Optimize A, build B
- [ ] Mon Profile A; name the limiting resource with evidence.
- [ ] Tue One memory/layout optimization for A; retest + rebench.
- [ ] Wed Simplest correct Kernel B.
- [ ] Thu Tests + sanitizer for B.
- [ ] Fri Baseline + profile B.
- [ ] Sat Review.

### Week 15 — Optimize & defend
- [ ] Mon One optimization for B; keep or revert.
- [ ] Tue Bounded tuning matrix for A and B.
- [ ] Wed Full correctness suite; confirm no lucky-input specialization.
- [ ] Thu Measure in a representative layer micro-workload; report kernel vs end-to-end separately.
- [ ] Fri Publish: README, source, tests, scripts, profiler evidence, worklog, what didn't work. **Reproducibility test:** a fresh clone runs the README end-to-end in ≤ 30 min.
- [ ] Sat Final review. Write the next 15-week plan.

---

**After Week 15:** land one merged open-source PR; repeat profile → hypothesis →
implement → measure on harder workloads (paged decode attention, grouped MoE GEMM).
