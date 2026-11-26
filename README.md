# Kernel Engineering Lab

A 15-week, 60-minute-a-day practice of GPU kernel engineering: understand a workload,
write a correct kernel, profile it, find the real bottleneck, optimize it, and defend
the measured result.

> *Depth over novelty, truth over ego, execution over appearance, long-term excellence
> over short-term recognition.*

Sessions logged: **1** / 90 (target ≥ 80)

## How this repo works

| File | Purpose |
|---|---|
| [NEXT.md](NEXT.md) | Today's first action. Open it and start. |
| [PLAN.md](PLAN.md) | The locked 15-week roadmap. |
| [RULEBOOK.md](RULEBOOK.md) | Philosophy, habit rules, engineering rules, AI rules. |
| [LEARNED.md](LEARNED.md) | Every concept learned, its status, mistakes made, open questions. |
| [worklog.md](worklog.md) | One entry per session + Saturday reviews. |
| [unresolved-questions.md](unresolved-questions.md) | Parking lot and important-problems list. |
| `benchmarks/` | `benchmark.py` harness; `results.csv` auto-records environment + timings. |
| `kernels/`, `tests/` | Code and correctness tests, organized by week/topic. |
| `resources/` | Reading list and checklists. |

## Daily loop

```
srun -p dev --gres=gpu:1 --cpus-per-task=8 --time=02:00:00 --pty bash   # get your OWN GPU via SLURM
cd /home/mohd.nauman/kernel-engineering-lab && source env.sh               # note: ~ is /fsx2 on GPU nodes
nvidia-smi                                                                 # confirm the GPU is idle before benchmarking
open NEXT.md → do the one task → run/measure → ./log.sh
```

`./log.sh` asks five short questions, appends the worklog entry, writes tomorrow's
`NEXT.md`, updates the session count, commits, and pushes.
Use `./log.sh min` for a 15-minute minimum session.

**Hardware:** NVIDIA H100 80GB HBM3 (sm_90), driver 580.95.05, CUDA 13.0, torch 2.13 (cu130), Triton 3.7.1.
First baseline: torch vector add, 2^24 fp32, cold L2 → 2743 GB/s (~82% of HBM3 peak).

## Benchmarking

```python
from benchmarks.benchmark import bench, check, record
```

Every result in `benchmarks/results.csv` includes GPU, compute capability, driver,
CUDA/torch/triton versions, shape, dtype, timing method, median and spread, and
correctness. See [benchmarks/methodology.md](benchmarks/methodology.md).
