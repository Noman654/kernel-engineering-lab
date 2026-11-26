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

## Where does 2048 come from?

Hardware bookkeeping: each resident warp needs a **scheduler slot** (PC + ready/waiting status) and its **registers**.
H100: 4 schedulers × 16 warp slots = 64 warps × 32 = **2048 threads**. Register file 65,536 / 2048 = 32 regs/thread.
Queried on my GPU: `max_threads_per_multi_processor = 2048`, `regs_per_multiprocessor = 65536`, `shared_memory_per_multiprocessor = 233472` (228 KB).

Blocks per SM = **minimum** of: threads (2048), blocks (32), registers (65,536), shared memory (228 KB).
- 256-thread blocks, 32 regs → 8 blocks (thread-limited)
- 256-thread blocks, 64 regs → 4 blocks (register-limited)
- 100 KB shared/block → 2 blocks (shared-memory-limited)
Resident warps / 64 = **occupancy** (Week 2).

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

## Subtle points (easy to miss)

**Warps**
- Warps are formed from consecutive threads: lane = `threadIdx.x % 32`, warp = `threadIdx.x / 32`. In 2-D blocks, x varies fastest, then y.
- Threads in a warp each have their own program counter (Volta+). Don't assume lockstep; use `__syncwarp()` when lanes exchange data.

**Launch shape**
- Blocks and grids can be 1-D, 2-D, or 3-D (`dim3`). Block: ≤ 1024 threads total (x, y ≤ 1024, z ≤ 64). Grid: x ≤ 2³¹−1, y and z ≤ 65,535.
- For huge n, compute the index in 64-bit: `size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;` (int overflows past ~2.1 B).

**Synchronization**
- `__syncthreads()` syncs one block only. There is no barrier across blocks inside a normal kernel; the kernel boundary is the global sync.
- Kernel launches are asynchronous: the CPU continues immediately. Time with CUDA events plus sync; check errors with `cudaGetLastError()` after the launch.

**Registers**
- Max 255 registers per thread. If a kernel needs more, the compiler **spills** to "local memory", which physically lives in HBM (cached in L1/L2), so spills are slow. Check with `nvcc --resource-usage`.

**Hopper-specific exception**
- H100 adds **thread block clusters**: up to 8 blocks (16 opt-in) scheduled together on one GPC that **can** read each other's shared memory (distributed shared memory). The "shared memory is per block" rule is exact only without clusters (Week 12).
- SMs are grouped: 2 SMs per TPC, ~16–18 SMs per GPC. Clusters are placed within one GPC.

**Tail effect**
- If blocks ÷ (132 × blocks per SM) leaves a partial last wave, some SMs sit idle at the end. It matters for small grids.

## Corrected misconceptions (Session 1 quiz)
- **A block is never split across SMs.** It runs entirely on one SM — required for shared memory and `__syncthreads()`.
- **One SM holds many blocks**, not the reverse: 2048 threads/SM → 8 blocks of 256, or 2 blocks of 1024 (thread limit; registers/shared memory can lower it).
- A warp = 32 **threads** (tickets), not 32 workers. Lanes (cores) execute the warp's instruction.
- `if (i < n)` exists because the grid has more threads than elements (n=1000, B=256 → 1024 threads); the extra 24 would access out of bounds.
- Divergence is **inside a warp**: `threadIdx.x % 2` diverges; `(threadIdx.x / 32) % 2` does not.
- Shared memory is **per block**, even for two blocks on the same SM. Blocks communicate only via global memory (atomics or separate launches).

## Open doubts
- (add here)
