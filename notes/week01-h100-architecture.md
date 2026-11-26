# Week 1 — How the H100 runs my code

My questions, in the order I asked them.

---

### 1. What physically runs my code?

The H100 is one chip with **132 SMs** (Streaming Multiprocessors). An SM is the real compute unit. The GPU is fast because all 132 work at the same time.

Think of a factory with 132 workshops.

Inside one SM:
- **128 cores**: they do the math. A "core" is really a lane, one slot of a 32-wide instruction.
- **4 schedulers**: each picks which warp runs next.
- **Registers**: each thread's private notepad.
- **Shared memory**: a whiteboard for one block.

---

### 2. What do I control?

I write code for **one thread**. Then I group threads:

- **Warp**: 32 threads that run the same instruction together. The hardware runs warps, not single threads.
- **Block**: threads I group together (e.g. 256 = 8 warps).
- **Grid**: all the blocks of one launch.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i] + b[i];
```

`if (i < n)` is there because the grid usually has more threads than data. With n = 1000 and 256-thread blocks I launch 4 blocks = 1024 threads, so the last 24 must do nothing.

---

### 3. How do blocks and SMs relate? (I got this wrong first)

- **A block runs on one SM and is never split.** Its threads share that SM's shared memory and can `__syncthreads()`. That only works if they are in the same place.
- **One SM holds many blocks.** For example, 8 blocks of 256 threads at once.
- Each block has its **own** shared memory. Two blocks on the same SM still cannot see each other's.

Teams and workshops: one workshop hosts several teams, and each team has its own whiteboard.

---

### 4. Why can an SM hold 2048 threads if it only has 128 cores?

Because **holding** a thread and **running** it are different.

- **Running each cycle:** 4 schedulers × 1 warp × 32 lanes = **128 threads**.
- **Held (resident):** up to **2048 threads** (64 warps), mostly waiting on memory.

A memory load takes about 500 cycles. While one warp waits, the scheduler runs another warp that is ready. With many warps waiting, the cores always have something to do. This is **latency hiding**.

Switching warps is free because every resident thread's registers stay on the SM. Nothing is saved or restored like on a CPU.

---

### 5. Where does 2048 come from?

It is a hardware design limit. Each resident warp needs a slot in a scheduler and space for its registers.

- 4 schedulers × 16 slots = 64 warps × 32 = **2048 threads**.
- My GPU reports `max_threads_per_multi_processor = 2048`.

---

### 6. What else limits how many blocks fit?

Registers. An SM has **65,536 registers**. That is a *count*, not KB; each register is 4 bytes, so 256 KB in total.

Example: 512-thread blocks, 64 registers per thread.
- Thread limit: 2048 ÷ 512 = 4 blocks
- Register limit: 65,536 ÷ (64 × 512) = **2 blocks** ← the smaller one wins

So heavy register use means fewer warps to hide latency with. This is called **occupancy** (Week 2).

---

### 7. Where does data live? L1, L2, HBM?

From close and fast to far and slow:

1. **Registers**: inside the SM, one thread. About 1 cycle.
2. **Shared memory / L1**: the same memory inside each SM. I control shared memory; the hardware controls L1. About 30 cycles.
3. **L2 (50 MB)**: on the chip, shared by all SMs. About 200+ cycles.
4. **HBM (80 GB)**: memory chips stacked next to the GPU chip. `cudaMalloc` puts my arrays here; in code it is called **global memory**. About 500+ cycles.

Vector add reads every value once from HBM and never reuses it, so HBM speed is the limit. That is why my baseline got 2743 GB/s, close to the ~3350 GB/s peak.

**The main idea of kernel engineering:** load from HBM once, reuse many times from shared memory or registers.

---

### Come back later
- **Spills:** past 255 registers per thread, data spills to slow "local memory", which is actually HBM.
- **Clusters (Week 12):** on H100, blocks in a cluster *can* share shared memory.
- **Kernel launches don't wait:** the CPU continues immediately, which matters for timing.

### My open doubts
-
