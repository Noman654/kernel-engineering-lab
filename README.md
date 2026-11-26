# Kernel Engineering Lab

A 15-week, 60-minute-a-day practice of GPU kernel engineering: understand a workload,
write a correct kernel, profile it, find the real bottleneck, optimize it, and defend
the measured result.

> *Depth over novelty, truth over ego, execution over appearance, long-term excellence
> over short-term recognition.*

Sessions logged: **0** / 90 (target ≥ 80)

## How this repo works

| File | Purpose |
|---|---|
| [NEXT.md](NEXT.md) | Today's first action. Open it and start. |
| [PLAN.md](PLAN.md) | The locked 15-week roadmap. |
| [RULEBOOK.md](RULEBOOK.md) | Philosophy, habit rules, engineering rules, AI rules. |
| [worklog.md](worklog.md) | One entry per session + Saturday reviews. |
| [unresolved-questions.md](unresolved-questions.md) | Parking lot and important-problems list. |
| `benchmarks/` | `benchmark.py` harness; `results.csv` auto-records environment + timings. |
| `kernels/`, `tests/` | Code and correctness tests, organized by week/topic. |
| `resources/` | Reading list and checklists. |

## Daily loop

```
open NEXT.md → do the one task → run/measure → ./log.sh
```

`./log.sh` asks five short questions, appends the worklog entry, writes tomorrow's
`NEXT.md`, updates the session count, commits, and pushes.
Use `./log.sh min` for a 15-minute minimum session.

## Benchmarking

```python
from benchmarks.benchmark import bench, check, record
```

Every result in `benchmarks/results.csv` includes GPU, compute capability, driver,
CUDA/torch/triton versions, shape, dtype, timing method, median and spread, and
correctness. See [benchmarks/methodology.md](benchmarks/methodology.md).
