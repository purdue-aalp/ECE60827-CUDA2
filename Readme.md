# ECE 60827 CUDA Programming Part 2

## Professor: Timothy Rogers <br> TA: Junrui Pan

## Introduction

The purpose of this lab is to deepen your understanding of CUDA programming by implementing General Matrix Multiplication (GEMM), a fundamental operation in linear algebra and deep learning. You will implement two versions of GEMM and compare their performance.

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
module load gcc cuda
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
Testing correctness...
==================================================
Size (64x64) x (64x64):
  Naive max error:  X.XXXXXXe-XX
  Shared max error: X.XXXXXXe-XX
...

All correctness tests passed!

==================================================
Benchmarking...
==================================================

Matrix size: (2048x2048) x (2048x2048)
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

**DO NOT modify files in the `autograder/` directory** - any changes to these files will be flagged and result in an instant 0.

-----------------------------------------------------------
<br>

## Report

### Profiling

Use Nsight Systems to profile your implementations:

```bash
nsys profile -o gemm_report python test_gemm.py
nsys stats gemm_report.nsys-rep
```

### Report Questions

Answer the following questions in your report:

1. **Performance comparison**
   - How does the shared memory version compare to the naive version?
   - What speedup do you observe?

2. **Memory access patterns**
   - How many global memory accesses does each thread make in the naive version?
   - How does shared memory reduce this?

3. **Profiling analysis**
   - What does the profiler show about memory bandwidth utilization?
   - Where are the bottlenecks in each implementation?

### Grading Rubric

| Component | Points |
|-----------|--------|
| Part A - Naive GEMM | 40 |
| Part B - Shared Memory GEMM | 40 |
| Report | 20 |
| **Total** | **100** |

### Submission Requirements

Include your written report named **`report.pdf`** in the root folder of your repository.

### Academic Integrity

**The use of AI tools (ChatGPT, GitHub Copilot, Claude, etc.) is prohibited for this assignment.** All code must be written by you. Violations will be treated as academic dishonesty.

## References

[1] [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

[2] [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
