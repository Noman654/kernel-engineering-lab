# What I've learned

Every concept I've covered, and how well I own it. Updated at the end of each session.
Saturdays: re-test 🟢 items closed-book; promote to ✅ or send back to `NEXT.md`.

**Status:** 🟡 learned · 🟢 practised in code · ✅ explained closed-book / blank-page test passed

Details for each concept are in [notes/](notes/). Mistakes I made are listed so I don't repeat them.

---

## Week 1 — Performance thinking & CUDA execution model
Note: [notes/week01-h100-architecture.md](notes/week01-h100-architecture.md)

| Concept | Status | Session | Key point in one line |
|---|---|---|---|
| Roofline, arithmetic intensity | ✅ | before plan | Already known |
| SM, sub-partitions, cores = lanes | 🟡 | 1 | 132 SMs × 4 sub-partitions × 32 FP32 lanes |
| Thread → warp → block → grid | 🟢 | 1 | I write code for one thread; hardware runs warps |
| Block lives on one SM, never split | 🟢 | 1 | One SM holds many blocks |
| Resident vs executing | 🟡 | 1 | 2048 resident, 128 executing per cycle; warp switch is free |
| Where 2048 comes from; occupancy limits | 🟡 | 1 | Min of threads, 32 blocks, 65,536 registers, 228 KB shared |
| Registers are a count, not KB | 🟡 | 1 | 65,536 registers × 4 B = 256 KB |
| Memory levels: registers, shared/L1, L2, HBM | 🟡 | 1 | Load from HBM once, reuse from fast memory |
| GPC / TPC | 🟡 | 1 | Chip-building detail; matters only for clusters (Week 12) |
| Global index + bounds check | 🟢 | 1 | `i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n)` |
| Ceil-division for grid size | 🟢 | 1 | `(n + b - 1) / b`; integer only (float32 breaks above 2²⁴) |
| compute-sanitizer memcheck | 🟢 | 1 | PASS ≠ correct; sanitizer found out-of-bounds the test missed |
| Timing with CUDA events (warm-up, median) | 🟢 | 1 | Benchmark only on an idle GPU from `srun` |
| Block size effect | 🟢 | 1 | Tiny blocks cost ~0.62 ns each to start (hypothesis); 128–512 is the sweet spot |
| Warp divergence | 🟢 | 1 | Costs only inside a warp; hidden in memory-bound kernels (1.03x) |
| CUDA → PTX → SASS | ⬜ | — | Next session |

### Mistakes I made (so I don't repeat them)
- Thought a block could be split across 2 SMs. It can't.
- Thought 256 threads need 2 SMs. One SM holds 8 such blocks.
- Thought `ceil(n/block)` works in C++. Integer division drops the remainder first.
- Predicted divergence would slow down vector add. It didn't, because the kernel is memory-bound.
- Benchmarked on another job's GPU and got an impossible 4.5 TB/s.

### Open questions
- Does `add_lane_div` compile to a branch or predication? (→ Session 2)
- Is block start-up really the limit for small blocks? (confirm with a profiler, Week 2)
