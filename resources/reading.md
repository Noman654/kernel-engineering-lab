# Reading list

Read only what the current week in `PLAN.md` calls for.

## Philosophy & craft (Saturdays, 10–15 pages)
- Aristotle — *Nicomachean Ethics*, Books I, II, VI
- Plato — *Apology*, *Euthyphro*
- Epictetus — *Enchiridion*, selected *Discourses*
- Richard Feynman — *Cargo Cult Science* (1974): https://calteches.library.caltech.edu/51/2/CargoCult.htm
- Richard Hamming — *You and Your Research* (1986): https://www.cs.utexas.edu/~dahlin/bookshelf/hamming.html
- Cal Newport — *So Good They Can't Ignore You*
- Background on deliberate practice (with critique): https://pmc.ncbi.nlm.nih.gov/articles/PMC7461852

## Foundations & CUDA
- NVIDIA CUDA C++ Programming Guide; CUDA C++ Best Practices Guide
- Williams, Waterman, Patterson — *Roofline: An Insightful Visual Performance Model*
- Sasha Rush — GPU Puzzles
- Simon Boehm — *How to Optimize a CUDA Matmul Kernel* (2022; learn the method, not the numbers): https://siboehm.com/articles/22/CUDA-MMM

## Profiling & correctness
- Nsight Systems / Nsight Compute documentation; Compute Sanitizer documentation

## Triton, CUTLASS, CuTe
- Official Triton tutorials; Triton Puzzles; Gluon tutorials
- NVIDIA CUTLASS / CuTe documentation

## Attention & inference
- FlashAttention (2022), FlashAttention-2 (2023), FlashAttention-3 (2024), FlashAttention-4
- FlashInfer (2025)

## Frontier & AI-generated kernels
- DeepGEMM, FlashMLA repositories
- KernelBench paper + repository

## Practice platform (optional warm-up)
- TensorTonic — https://www.tensortonic.com/ (free tier is enough). Rule: max **1 problem, 10 min**, only on the **current week's topic**, or on Saturdays. It checks correctness, not speed — real measurement and profiling stay in this repo.
  - Week 1: "GPU Occupancy Calculator". Weeks 9–11: attention / normalization problems.
