# ECE 60827 CUDA Part 2 - Report

**Name:**
**Email:**

---

## Part 1: Understanding the Codebase

**How does a Python call to `cuda_gemm.gemm(A, B)` end up executing your CUDA kernel? Trace the path through `test_gemm.py`, `setup.py`, and `cuda_gemm.cu`.**

<!-- Your answer here -->

---

## Part 2: Performance Analysis

### Question 1

**Run `make test` and report the timing results. How does the shared memory version compare to the naive version? What speedup do you observe?**

<!-- Paste your make test output and analysis here -->


### Question 2

**For the naive kernel, how many global memory loads does each thread perform to compute one element of C? For an M×K by K×N multiplication, what is the total number of global memory loads across all threads?**

<!-- Your answer here -->

### Question 3

**For the shared memory kernel with tile size T (BLOCK_SIZE), how many global memory loads does each thread perform? How does tiling reduce the total number of global memory accesses?**

<!-- Your answer here -->

### Question 4

**Both your implementations are likely slower than PyTorch's `torch.mm`. Why do you think this happens?**

<!-- Your answer here -->

---

## Part 3: Loop Unrolling Analysis

### Question 5

**What is loop unrolling and how does `#pragma unroll` work at the compiler level? What trade-offs does it introduce (e.g., instruction cache pressure, register usage)?**

<!-- Your answer here -->

### Question 6

**Compare the performance of your unrolled kernel against the shared memory version. Report the timing results and speedup. Does unrolling help equally for all matrix sizes? Why or why not?**

<!-- Your answer here -->

### Question 7

**Describe one additional optimization technique (beyond shared memory tiling and loop unrolling) that could further improve GEMM performance. Explain the underlying principle and expected benefit.**

<!-- Your answer here -->
