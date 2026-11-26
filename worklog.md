# Worklog

Append-only. One entry per session, written by `./log.sh`. Benchmark details live in
`benchmarks/results.csv` (recorded automatically).

**Daily entry format**
- **Did:** what I worked on (one topic)
- **Result:** what happened, including the prediction vs reality
- **Evidence:** numbers, test result, profiler finding, or file path
- **Lesson:** what was in my control; where attention leaked; what failure taught me
- **Next:** tomorrow's first action (also written to `NEXT.md`)

**Saturday weekly review** (add by hand under that week's entries)
- Sessions this week: _/6
- Blank-page test: kernel ___ → pass / fail
- What I can now explain closed-book:
- Biggest misconception corrected:
- Where I practised (or failed to practise) the personal code:
- Philosophy reading: pages ___ — one idea that applied this week:

---

## Session 1 — 2026-11-26 (full)
- **Did:** yes
- **Result:** yes
- **Evidence:** kernels/week01/vector_add.cu; PASS + sanitizer 0 errors; block 32/64/128 = 321/163/85 us (~0.62 ns per block); busy GPU gave impossible 4.5 TB/s
- **Lesson:** check always gpu utilisation it can show the wrong result
- **Next:** run cuobjdump -sass on kernels/week01/vector_add.cu; check if add_lane_div uses a branch (BRA) or predication (@P/SEL) - explains its 1.03x vs add_warp_div
