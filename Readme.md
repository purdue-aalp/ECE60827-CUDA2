# ECE 60827 CUDA Programming Part 2

## Professor: Timothy Rogers <br> TA: Junrui Pan

## Introduction

Large Language Models (LLMs) like GPT and LLaMA are built on the Transformer architecture, which relies heavily on matrix multiplications. The core operations in Transformers—attention mechanisms (Q×K^T, softmax×V) and feed-forward layers—are essentially sequences of GEMMs. In fact, matrix multiplications account for the vast majority of compute in modern LLMs, making GEMM optimization critical for efficient inference and training.

The purpose of this lab is to deepen your understanding of CUDA programming by implementing General Matrix Multiplication (GEMM). You will implement two versions—a naive approach and an optimized shared memory version—and compare their performance against PyTorch's highly-optimized implementation.

The official [CUDA Documentation](https://docs.nvidia.com/cuda/) is the best resource for implementation details and API specifics.

-----------------------------------------------------------
<br>

## GEMM (General Matrix Multiplication)

GEMM computes the matrix product:
```
C = A × B
```

Where:
- A is an M × K matrix
- B is a K × N matrix
- C is an M × N matrix (output)

Each element of C is computed as:
```
C[i][j] = Σ(k=0 to K-1) A[i][k] * B[k][j]
```

### Matrix Storage

All matrices are stored in **row-major order**:
- `A[i][j]` is stored at `A[i * K + j]`
- `B[i][j]` is stored at `B[i * N + j]`
- `C[i][j]` is stored at `C[i * N + j]`

-----------------------------------------------------------
<br>

## Part A: Naive GEMM

Implement `gemm_kernel` in `cuda_gemm.cu`.

Each thread computes **one element** of the output matrix C. The thread reads an entire row of A and an entire column of B from global memory to compute the dot product.

-----------------------------------------------------------
<br>

## Part B: Shared Memory GEMM

Implement `gemm_shared_kernel` in `cuda_gemm.cu`.

Use **shared memory** to reduce global memory accesses. Threads within a block should cooperatively load data into shared memory before computing.

-----------------------------------------------------------
<br>

## Building and Testing

### Setup Environment

```bash
module load gcc/11.4.1 cuda ngc pytorch
```

### Build

```bash
make build
```

**Note:** The first build may take a few minutes (~3 min) as it compiles the PyTorch CUDA extension.

### Test

```bash
make test
```

This will run correctness tests and benchmarks comparing your implementations against PyTorch's optimized GEMM.

### Clean

```bash
make clean
```

-----------------------------------------------------------
<br>

## Expected Output

When your implementation is correct, `make test` should show:

```
==================================================
Benchmarking...
==================================================

Matrix size: (1024x2048) x (2048x512)
Naive max error:  X.XXXXXXe-XX
Shared max error: X.XXXXXXe-XX

PyTorch mm:    X.XXX ms
Naive GEMM:    X.XXX ms
Shared GEMM:   X.XXX ms

Speedup (shared vs naive): X.XXx
Relative to PyTorch (shared): X.XXx
```

-----------------------------------------------------------
<br>

## Submission and Autograding

When you submit your assignment through GitHub Classroom, an autograder will automatically test your implementation.

### What Gets Tested

- **Part A - Naive GEMM:** Correctness against PyTorch reference
- **Part B - Shared Memory GEMM:** Correctness against PyTorch reference

### Running the Grader Locally

```bash
make test
```

-----------------------------------------------------------
<br>

## Report

### Report Questions

Answer the following questions in your report. Read through all provided files (`cuda_gemm.cu`, `setup.py`, `test_gemm.py`) carefully before answering.

#### Part 1: Understanding the Codebase

How does a Python call to `cuda_gemm.gemm(A, B)` end up executing your CUDA kernel? Trace the path through `test_gemm.py`, `setup.py`, and `cuda_gemm.cu`. See [Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) for additional context.

#### Part 2: Performance Analysis

1. Run `make test` and report the timing results. How does the shared memory version compare to the naive version? What speedup do you observe?

2. For the naive kernel, how many global memory loads does each thread perform to compute one element of C? For an M×K by K×N multiplication, what is the total number of global memory loads across all threads?

3. For the shared memory kernel with tile size T (BLOCK_SIZE), how many global memory loads does each thread perform? How does tiling reduce the total number of global memory accesses?

4. Both your implementations are likely slower than PyTorch's `torch.mm`. Why do you think this happens?

#### Part 3: Further Optimizations

5. Describe **two** additional optimization techniques (beyond shared memory tiling) that could improve GEMM performance. For each technique, explain the underlying principle and expected benefit.

### Grading Rubric

| Component | Points |
|-----------|--------|
| Part A - Naive GEMM | 40 |
| Part B - Shared Memory GEMM | 40 |
| Report | 20 |
| **Total** | **100** |

### Submission Requirements

Include your written report named **`report.md`** in the root folder of your repository. See [Markdown Guide](https://www.markdownguide.org/basic-syntax/) for formatting help.

### Academic Integrity

**The use of AI tools (ChatGPT, GitHub Copilot, Claude, etc.) is prohibited for this assignment.** All code must be written by you. Violations will be treated as academic dishonesty.

## References

[1] [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

[2] [PyTorch C++/CUDA Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)

[3] [Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
