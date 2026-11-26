# Week 1 — H100 architecture & the CUDA execution model

**Question this answers:** when I launch a kernel, what physical hardware runs my code, and how does my thread/block/grid choice map onto it?

## Mental model: the GPU is a factory

| Hardware | H100 SXM | Factory analogy |
|---|---|---|
| Chip | 1 die (GH100; 132 of 144 SMs enabled) | The factory |
| **SM** (Streaming Multiprocessor) | **132** | A workshop — the real compute unit |
| Sub-partition per SM | 4, each with 1 warp scheduler, 32 FP32 lanes, 1 tensor core | 4 supervisors, each with a squad's worth of workers |
| FP32 lanes ("CUDA cores") per SM | 128 | Workers. A "core" is a lane/ALU, *not* a CPU-style core |
| Tensor cores per SM | 4 | Specialist matrix-math machines |
| Registers per SM | 65,536 × 32-bit = 256 KB (≤ 255 per thread) | Private notepads |
| Shared memory / L1 per SM | 256 KB combined, ≤ 228 KB shared | Team whiteboard (one block only) |
| L2 cache | 50 MB, shared by all SMs | Loading dock shared by all workshops |
| HBM3 | 80 GB, ~3.35 TB/s | The warehouse — big, far, slow (~500+ cycles) |

```
H100 chip
┌───────────────────────────────────────────┐
│ [SM 0] [SM 1] [SM 2] ...        [SM 131]  │
│            L2 cache (50 MB)               │
└──────────────────┬────────────────────────┘
                HBM3 80 GB

One SM
┌───────────────────────────────────────────┐
│ Sub-part 0 │ Sub-part 1 │ Sub-part 2 │ 3  │  each: 1 scheduler,
│ 32 FP32    │ 32 FP32    │ 32 FP32    │ 32 │  32 FP32 lanes,
│ 1 TC       │ 1 TC       │ 1 TC       │ 1  │  1 tensor core
│ Register file (256 KB)                    │
│ L1 / shared memory (256 KB)               │
└───────────────────────────────────────────┘
```

## Software side

| Term | Meaning | Analogy |
|---|---|---|
| Thread | One unit of my work, e.g. `c[517] = a[517] + b[517]` | A job ticket |
| Warp | 32 threads issued together, one instruction at a time — the unit hardware runs | A squad marching in step |
| Block (CTA) | Threads I group (≤ 1024); lives on **one** SM until done; can share memory and `__syncthreads()` | A team assigned to one workshop |
| Grid | All blocks of one launch | The whole order |

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;   // my global index
if (i < n) c[i] = a[i] + b[i];                    // grid usually overshoots n
// gridDim = (n + blockDim - 1) / blockDim
```

## How a launch runs

1. `add<<<65536, 256>>>` → 65,536 blocks × 8 warps.
2. Block scheduler places blocks on SMs; several per SM if resources allow (limits: 2048 threads = 64 warps, 32 blocks, registers, shared memory).
3. Each cycle, each of the 4 schedulers picks a **ready** warp and issues one instruction for its 32 lanes.
4. A warp waiting on HBM is skipped; another ready warp issues → **latency hiding**. More resident warps = more to choose from (that's why occupancy matters — as a means, not a goal).
5. Blocks run in **no guaranteed order**.

Example: n = 2²⁴, B = 256 → 65,536 blocks; 8 resident blocks/SM → 1,056 at once → ~62 waves.

## Rules of thumb
- Block size: multiple of 32; 128–256 default.
- Divergence costs at **warp** level: lanes of one warp taking different branches serialize; whole warps taking different branches cost nothing extra.
- I write code for one thread; hardware groups threads into warps, places blocks on SMs, and runs thousands concurrently.

## Open doubts
- (add here)
