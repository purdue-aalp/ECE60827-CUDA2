# ECE 60827 CUDA Part 2 - Report

**Please write all your answers directly in this `report.md` file. Do not create a separate PDF or other document.**

---

## Part 1: Understanding the Codebase

**How does a Python call to `cuda_gemm.gemm(A, B)` end up executing your CUDA kernel? Trace the path through `test_gemm.py`, `setup.py`, and `cuda_gemm.cu`.**

<!-- Your answer here -->

---

## Part 2: Performance Analysis

### Question 1

**Run `make test` and report the timing results. How does the shared memory version compare to the naive version? What speedup do you observe?**

<!-- Your answer here -->


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

**What is loop unrolling and how does it help?  What trade-offs does it introduce (e.g., instruction cache pressure, register usage)?**

<!-- Your answer here -->

### Question 6

**Compare the performance of your unrolled kernel against the shared memory version. Do you see any speedup? Why or Why not?**

<!-- Your answer here -->

### Question 7

**Describe one additional optimization technique (beyond shared memory tiling and loop unrolling) that could further improve GEMM performance. Explain the underlying principle and expected benefit.**

<!-- Your answer here -->

### Question 8

**Write a separate Python script that runs your three GEMM implementations and PyTorch's `torch.mm` using FP16 (`torch.float16`) tensors instead of FP32. Report the timing results for all four. What happens to PyTorch's performance compared to FP32? What happens to your kernels' performance? Explain why PyTorch sees a significant speedup with FP16 while your custom kernels do not. (hint: Some special hardware in Volta can only do FP16, but not FP32.)**

<!-- Paste your FP16 timing results and explanation here -->
