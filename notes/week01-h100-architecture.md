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

## Resident vs executing (why 2048 threads but only 128 cores?)

| | Per SM | Meaning |
|---|---|---|
| Executing per cycle | 4 warps = 128 threads | 4 schedulers × 1 warp each × 32 lanes |
| Resident | ≤ 64 warps = 2048 threads | loaded, registers kept on-chip, waiting their turn |

- Most resident warps are waiting on HBM (~500 cycles). Each cycle a scheduler issues from a warp that is **ready**.
- Switching warps costs **0 cycles**: every resident thread's registers stay on the SM (65,536 regs / 2048 threads = 32 each). No save/restore like a CPU.
- More registers per thread → fewer resident threads (64 regs → 1024 threads). That is **occupancy**.
- One line: 128 compute, up to 2048 wait their turn, so cores never sit idle while data travels.

## Memory hierarchy (close/fast → far/slow)

| Level | Where | Visible to | Controlled by | ~Latency |
|---|---|---|---|---|
| Registers | in SM | one thread | compiler (local vars) | ~1 cycle |
| Shared memory | in SM (same SRAM as L1, 256 KB split) | one block | **me** (`__shared__`) | ~30 cycles |
| L1 cache | in SM | that SM | hardware | ~30 cycles |
| L2 cache (50 MB) | on chip | all SMs | hardware | ~200+ cycles |
| HBM (80 GB, ~3.35 TB/s) | stacked DRAM beside the die | everyone | me (`cudaMalloc`) | ~500+ cycles |

- **HBM** = High Bandwidth Memory: the GPU's main memory. Software name: **global memory**.
- Lookup order for a load: L1 → L2 → HBM.
- Core trick of kernel engineering: **load from HBM once, reuse many times from shared memory/registers.**
- Vector add reuses nothing → HBM-bandwidth-bound → my baseline 2743 GB/s.

## Corrected misconceptions (Session 1 quiz)
- **A block is never split across SMs.** It runs entirely on one SM — required for shared memory and `__syncthreads()`.
- **One SM holds many blocks**, not the reverse: 2048 threads/SM → 8 blocks of 256, or 2 blocks of 1024 (thread limit; registers/shared memory can lower it).
- A warp = 32 **threads** (tickets), not 32 workers. Lanes (cores) execute the warp's instruction.
- `if (i < n)` exists because the grid has more threads than elements (n=1000, B=256 → 1024 threads); the extra 24 would access out of bounds.
- Divergence is **inside a warp**: `threadIdx.x % 2` diverges; `(threadIdx.x / 32) % 2` does not.
- Shared memory is **per block**, even for two blocks on the same SM. Blocks communicate only via global memory (atomics or separate launches).

## Open doubts
- (add here)
