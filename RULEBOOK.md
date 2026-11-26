# Rulebook

This file is the constitution of the lab. It changes rarely and deliberately. The
roadmap is in [PLAN.md](PLAN.md); today's task is in [NEXT.md](NEXT.md).

---

## 1. Personal code

> I will pursue **depth over novelty, truth over ego, execution over appearance, and
> long-term excellence over short-term recognition.**
>
> I will question what I learn, measure what I build, take responsibility for my
> decisions, and remain a student even when I become good at something.

*(A personal code inspired by the thinkers below, not a historical quotation.)*

---

## 2. Philosophical foundation

Three pillars, each answering one question. Plus four modern engineers/scientists who
turned the same ideas into working practice.

### The three pillars

| Thinker | Question it answers | Core idea | How it shows up here |
|---|---|---|---|
| **Aristotle** — foundation | How do I become excellent? | Excellence is a *habit* formed by repeated action ("we become builders by building", *Nicomachean Ethics* II.1, Ross tr.). *Phronesis* = practical wisdom: knowing what to do in *this* situation. | The daily session. Building beats reading. Judgment comes from many small decisions logged and reviewed. |
| **Socrates** — method of thinking | How do I think independently? | Examine claims, expose contradictions, admit what you don't know. | The *Investigate* slot. Never accept "it's faster" without asking why, and what would prove it wrong. |
| **Epictetus** — handling difficulty | How do I stay disciplined when it's hard? | Separate what is in my control (my effort, judgments, actions) from what is not (outcomes, others' progress, hardware access). (Paraphrase of *Enchiridion* 1.) | The *Reflect* slot. Failed experiments are data, not verdicts. Compare myself only with my past log. |

### Four modern reinforcements

| Source | Lesson | Rule it becomes |
|---|---|---|
| **Richard Feynman**, *Cargo Cult Science* (1974) | "The first principle is that you must not fool yourself — and you are the easiest person to fool." | Benchmark honesty. Report what did *not* work. A fast wrong kernel is a failure. |
| **Richard Hamming**, *You and Your Research* (1986) | "If you do not work on an important problem, it's unlikely you'll do important work." Keep a list of important problems; work so others can build on it; keep the door open. | Maintain an *important problems* list (in `unresolved-questions.md`). Publish reproducible work. Share progress publicly. |
| **K. Anders Ericsson**, deliberate practice | Skill grows from practice at the *edge* of ability, with a *specific goal* and *fast feedback*. (Note: researchers debate how precisely the theory is defined — use it as a practical guide, not dogma.) | Every session has one specific goal. Feedback comes from tests, profilers, and the blank-page test — not from feeling productive. |
| **Cal Newport**, *So Good They Can't Ignore You* | Craftsman mindset: ask what value you produce, not what the work gives you. Build *rare and valuable* skills (career capital) first; passion follows competence. | Kernel engineering *is* the rare skill. The output is evidence of value: kernels, worklogs, merged PRs. |

### Reading (Saturdays only, 10–15 pages, replaces part of review)

1. Aristotle — *Nicomachean Ethics*, Books I, II, VI
2. Plato — *Apology* (then *Euthyphro*)
3. Epictetus — *Enchiridion* (then selected *Discourses*)
4. Feynman — *Cargo Cult Science* (one sitting)
5. Hamming — *You and Your Research* (one sitting)
6. Newport — *So Good They Can't Ignore You*

Studying a philosophy is not practising it. **The test is whether my decisions and
behaviour change.** Each Saturday, write one line in the weekly review: *where did I
practise (or fail to practise) the code this week?*

---

## 3. Habit rules (the anti-juggling system)

1. **Fixed slot.** Same time, same place, anchored to an existing habit.
2. **Zero-decision start.** Open `NEXT.md` and begin. Tomorrow's first action is
   written at the end of today.
3. **One topic open at a time.** New ideas go to `unresolved-questions.md`, never into
   the session.
4. **Minimum viable session = 15 minutes.** 2 min open `NEXT.md` → 10 min smallest
   experiment → 3 min log + next action. It counts.
5. **One tool, one log.** This repo is the single source of truth. No second tracker,
   no parallel course, no new roadmap.
6. **Rest is planned, not missed.** Measure *weekly consistency*, not streaks.
7. **Fall behind → shift right.** Never compress or cram.
8. **The destination is locked for 15 weeks.** Only the next task adapts.

### The 60-minute session

**Order is fixed: Learn → Practice → Questions.** Never questions before the topic is taught.
If I already know the topic, say so — it is skipped and we move to the next one.

| Time | Slot | Pillar | What to do |
|---|---|---|---|
| 20 min | **Learn** | Aristotle | AI teaches the topic directly and clearly: concept, numbers, diagrams, examples. No quizzing yet. |
| 30 min | **Practice** | Aristotle | Code, run, measure. Predict the result *before* running. |
| 10 min | **Questions + log** | Socrates / Epictetus | AI asks a few questions on what was learned and practised. Then `./log.sh`. |

### Weekly rhythm

| Day | Purpose |
|---|---|
| Mon–Fri | Core sessions from `PLAN.md` |
| Sat | Buffer → catch up first. Then: blank-page test, closed-book quiz, philosophy reading, weekly review |
| Sun | Rest |

---

## 4. Engineering rules

1. **Correctness before speed.** Compare against a trusted reference; test awkward
   sizes, tails, dtypes, strides, boundaries.
2. **Sanitize from day one.** Run `compute-sanitizer` on every new CUDA kernel.
3. **Measure before changing.** Keep a baseline; one targeted change per experiment.
4. **Timeline before kernel.** Nsight Systems to find where time goes, then Nsight
   Compute to explain the limiting kernel.
5. **No vanity metrics.** Occupancy and "GPU utilization" are not goals.
6. **Benchmark honestly.** Warm up, synchronize, repeat, report median + spread, fix
   shapes/dtypes/layouts, compare equivalent precision. `benchmarks/benchmark.py`
   records the environment automatically.
7. **Predict first.** Before every run, write the expected bottleneck and number.
8. **Keep or revert.** Every change: hypothesis → expected effect → measurement →
   profiler evidence → decision.
9. **Approved compute only.** Use GPUs I am authorized to use.

---

## 5. AI rules — speed without getting weaker

**Core rule: I generate first, AI critiques second.** The struggle of producing the
first attempt is where learning happens.

| Phase | AI role | Allowed | Not allowed |
|---|---|---|---|
| Weeks 1–5 (basics) | **Teacher** | Teach the topic directly first, then set the practice, then quiz at the end; write harness/plot boilerplate | Writing kernel code; quizzing before teaching |
| Weeks 6–10 | **Reviewer** | Review my code for bugs, generate adversarial tests, help interpret profiler output (I verify) | Rewriting my code |
| Weeks 11–15 | **Accelerator** | Generate variants, scaffold experiments, draft docs | Having the final word — the harness decides |

### Standard prompts

1. **Teacher:** "Teach me X clearly and directly. Then give me practice. Questions at the end."
2. **Predict:** "Here is my kernel and GPU. Quiz me on the expected bottleneck before I run it."
3. **Reviewer:** "Find correctness bugs, races, and untested edge cases. Don't rewrite. List issues only."
4. **Examiner (Sat):** "Quiz me closed-book on this week's concepts. Grade harshly."

### The blank-page test (every Saturday)

Rewrite one kernel from this week from an empty file — no notes, no AI. If I can't, I
don't own it yet; it goes back into `NEXT.md`.

---

## 6. What separates me from the crowd

Everyone can get AI to write a kernel. Few can:

1. **Predict** performance before measuring (roofline, coalescing, occupancy).
2. **Verify** — catch the kernel that is fast because it is wrong.
3. **Explain with evidence** — profiler data and an honest worklog.
4. **Show public proof of work** — published worklogs and, after ~Week 10, one merged
   PR in a real project (vLLM, SGLang, FlashInfer, Triton, PyTorch).

Fundamentals give judgment. AI gives speed. Public evidence gives proof.

---

## 7. Success after 15 weeks

- ≥ **80 of 90** study sessions completed
- One reproducible capstone with tests, benchmarks, profiler evidence
- A documented benchmark history in `benchmarks/results.csv`
- Can explain and defend every optimization decision
