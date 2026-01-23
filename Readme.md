# ECE 60827 CUDA Programming Part 2

## Professor: Timothy Rogers <br> TA: Junrui Pan

## Introduction

Large Language Models (LLMs) like GPT and LLaMA are built on the Transformer architecture, which relies heavily on matrix multiplications. The core operations in Transformers—attention mechanisms (Q×K^T, softmax×V) and feed-forward layers—are essentially sequences of GEMMs. In fact, matrix multiplications account for the vast majority of compute in modern LLMs, making GEMM optimization critical for efficient inference and training. See [Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) for more details.

The purpose of this lab is to deepen your understanding of CUDA programming by implementing General Matrix Multiplication (GEMM). You will implement two versions—a naive approach and an optimized shared memory version—and compare their performance against PyTorch's highly-optimized implementation. Beyond learning CUDA itself, this lab helps you understand how widely-used frameworks like PyTorch interact with the underlying GPU hardware. By building your own GEMM kernels and comparing them against PyTorch's implementation, you'll gain insight into the entire software stack—from high-level Python APIs down to low-level GPU execution.

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

## Repository Structure

This repository demonstrates how Python frameworks like PyTorch integrate with custom CUDA code:

- **`test_gemm.py`** — The Python test script that imports `cuda_gemm` as a module and calls functions like `cuda_gemm.gemm(A, B)` on PyTorch GPU tensors.
- **`setup.py`** — Defines how the `cuda_gemm` module gets built as a PyTorch C++/CUDA extension.
- **`cuda_gemm.cu`** — Contains the CUDA kernels you will implement, along with wrapper functions that bridge Python calls to GPU execution.

Understanding this flow—from high-level Python API to low-level CUDA kernel—is a key learning objective of this lab.

### test_gemm.py

| Function | Description |
|----------|-------------|
| `test_naive()` | Tests Part A correctness by comparing your naive GEMM output against `torch.mm()` across multiple matrix sizes. |
| `test_shared()` | Tests Part B correctness by comparing your shared memory GEMM output against `torch.mm()` across multiple matrix sizes. |
| `benchmark()` | Measures and compares execution time of your implementations against PyTorch's optimized GEMM. |

### cuda_gemm.cu

| Function | Description |
|----------|-------------|
| `gemm_kernel` | **[TODO]** The naive CUDA kernel where each thread computes one element of the output matrix. |
| `gemm_shared_kernel` | **[TODO]** The shared memory CUDA kernel that uses tiling to reduce global memory accesses. |
| `gemm_cuda()` | C++ wrapper that validates input tensors, configures grid/block dimensions, and launches `gemm_kernel`. |
| `gemm_shared_cuda()` | C++ wrapper that validates input tensors, configures grid/block dimensions, and launches `gemm_shared_kernel`. |

**Note:** The grid and block dimensions in the wrapper functions can be adjusted as needed for your kernel implementation.
| `PYBIND11_MODULE` | Registers the C++ functions as Python-callable methods (`cuda_gemm.gemm` and `cuda_gemm.gemm_shared`). See [pybind11 documentation](https://pybind11.readthedocs.io/). |

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

To test each part separately (as the autograder does):

```bash
# Test Part A only
python3 test_gemm.py --part-a

# Test Part B only
python3 test_gemm.py --part-b
```

Running `make test` will execute the full benchmark, which tests both parts and reports timing comparisons.

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
