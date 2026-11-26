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

### 1b. What's inside the H100, drawn from the outside in

**Level 1: the whole chip**

```
┌──────────────────────────── H100 SXM chip ────────────────────────────┐
│                                                                        │
│   GigaThread Engine  ← hands out my blocks to SMs                      │
│                                                                        │
│  ┌─ GPC 0 ─┐ ┌─ GPC 1 ─┐ ┌─ GPC 2 ─┐ ┌─ GPC 3 ─┐                       │
│  │ ~16 SMs │ │ ~16 SMs │ │ ~16 SMs │ │ ~16 SMs │                       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                       │
│  ┌──────────────── L2 cache 50 MB (shared by all SMs) ──────────────┐  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌─ GPC 4 ─┐ ┌─ GPC 5 ─┐ ┌─ GPC 6 ─┐ ┌─ GPC 7 ─┐                       │
│  │ ~16 SMs │ │ ~16 SMs │ │ ~16 SMs │ │ ~16 SMs │    8 GPCs = 132 SMs   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                       │
│                                                                        │
│  Memory controllers ──────────────────────────────── NVLink / PCIe    │
└──────┬─────────┬─────────┬─────────┬─────────┬─────────────────────────┘
   ┌───┴──┐  ┌───┴──┐  ┌───┴──┐  ┌───┴──┐  ┌───┴──┐
   │ HBM3 │  │ HBM3 │  │ HBM3 │  │ HBM3 │  │ HBM3 │   5 stacks = 80 GB
   │ 16 GB│  │ 16 GB│  │ 16 GB│  │ 16 GB│  │ 16 GB│   ~3.35 TB/s total
   └──────┘  └──────┘  └──────┘  └──────┘  └──────┘
```

- **GPC** (Graphics Processing Cluster): a group of SMs. Matters later for clusters (Week 12).
- **HBM** stacks sit *beside* the chip in the same package, not inside it. That's why going there is slow.

**Level 2: one GPC**

```
┌──────────────── GPC ────────────────┐
│ ┌─ TPC ──────┐  ┌─ TPC ──────┐       │
│ │ [SM] [SM]  │  │ [SM] [SM]  │  ...  │   TPC = 2 SMs
│ └────────────┘  └────────────┘       │   ~8–9 TPCs per GPC
└──────────────────────────────────────┘
```

**Level 3: one SM** (132 of these)

```
┌──────────────────────────────── SM ────────────────────────────────┐
│                       L1 instruction cache                          │
│ ┌─ Sub-part 0 ─┐ ┌─ Sub-part 1 ─┐ ┌─ Sub-part 2 ─┐ ┌─ Sub-part 3 ─┐ │
│ │  (see below) │ │  (see below) │ │  (see below) │ │  (see below) │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │
│                                                                     │
│ ┌──────── L1 data cache + Shared memory: 256 KB together ────────┐ │
│ │   shared memory (I control, per block) │ L1 (hardware cache)    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│   TMA (bulk copy engine, Week 12)          Texture units            │
└─────────────────────────────────────────────────────────────────────┘
```

**Level 4: one sub-partition** (4 per SM, this is where warps actually run)

```
┌────────────────── Sub-partition ──────────────────┐
│ L0 instruction cache                               │
│ Warp scheduler   → picks 1 ready warp per cycle    │
│ Dispatch unit    → sends its instruction to lanes  │
│                                                    │
│ Register file: 16,384 registers (64 KB)            │
│                                                    │
│ ┌────────────────────────────────────────────────┐ │
│ │ FP32 × 32   ← the "32 cores" (one warp wide)   │ │
│ │ INT32 × 16                                     │ │
│ │ FP64 × 16                                      │ │
│ │ Tensor core × 1  (matrix math, Week 8)         │ │
│ │ LD/ST × 8   (load/store to memory)             │ │
│ │ SFU × 4     (exp, sin, sqrt)                   │ │
│ └────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

**Check the numbers add up:**
- 4 sub-partitions × 32 FP32 = **128 cores per SM**
- 4 sub-partitions × 16,384 = **65,536 registers per SM**
- 4 schedulers × 1 warp per cycle = **4 warps running per cycle**
- 132 SMs × 128 = **16,896 FP32 cores on the chip**

**Where my code lands:**
```
my grid ─→ GigaThread Engine ─→ each block goes to one SM
                                   └─→ its warps are split across the 4 sub-partitions
                                          └─→ each cycle, a scheduler runs one warp on its 32 FP32 lanes
```

---

### 1c. What is a GPC, and why does it exist?

A **GPC** (Graphics Processing Cluster) is a neighbourhood of SMs on the chip.

```
chip ─→ 8 GPCs ─→ each has ~8–9 TPCs ─→ each TPC has 2 SMs
```

- The full GH100 design is 8 GPCs × 9 TPCs × 2 SMs = **144 SMs**. My H100 SXM has **132** enabled. Chips are never perfect, so NVIDIA switches off a few SMs that have defects (this is called *yield*). As a result, my GPCs don't all have the same number of SMs.
**Why have GPC and TPC at all? Why not just 132 SMs in a row?**

1. **The chip is too big to wire flat.** The H100 is about 814 mm² with 80 billion transistors. If every SM were wired directly to everything, the wires would be long, slow, and power-hungry. So the chip is built like a city: SMs sit in pairs (TPC), pairs form neighbourhoods (GPC), and neighbourhoods connect to the shared L2 and HBM. Nearby things are wired closely; far things go through shared roads.
2. **Design once, copy many times.** Engineers design one TPC, copy it to build a GPC, then copy the GPC 8 times. That's how you build an 80-billion-transistor chip without designing every piece by hand.
3. **Yield.** Some SMs always come out broken. Because of the grouping, NVIDIA can switch off a broken TPC and still sell the chip. The same design ships as different products: H100 SXM has 132 SMs, H100 PCIe has 114.
4. **Shared hardware.** The names come from graphics. Each **G**raphics **P**rocessing **C**luster has its own raster engine for drawing triangles, and each **T**exture **P**rocessing **C**luster pair shares some graphics hardware. For compute (my kernels), the useful part is that SMs in one GPC are wired together, which is what makes clusters possible.
5. **Simpler work distribution.** The GigaThread engine hands blocks to GPCs, and GPCs hand them to SMs. Two small steps are easier to build in hardware than one giant one.

So: **TPC and GPC exist for building and wiring the chip. As a kernel writer I mostly only see the SM, except when I use clusters.**

- **Normally I don't care about GPCs.** My blocks are spread over all SMs, and I can't choose where they go.
- **Why it matters on H100: clusters.** I can ask for a group of blocks (a *cluster*, up to 8, or 16 with opt-in) to be placed **inside one GPC at the same time**. SMs in one GPC are wired together, so blocks in a cluster can read each other's shared memory. Without clusters, blocks can't share at all (Week 12).

```
┌──────────────── one GPC ────────────────┐
│  [SM]  [SM]  [SM]  [SM]  [SM]  [SM] ... │
│   ↑      ↑     ↑     ↑                  │
│  blk0  blk1  blk2  blk3   ← one cluster │
│   └──────┴─────┴─────┘                  │
│   can read each other's shared memory   │
└─────────────────────────────────────────┘
```

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
