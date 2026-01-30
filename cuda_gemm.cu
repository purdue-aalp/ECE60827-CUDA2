#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// =============================================================================
// ECE60827 CUDA Homework: GEMM Implementation
// =============================================================================
//
// Your task: Implement three GEMM (General Matrix Multiply) kernels
//   C = A * B
//   where A is (M x K), B is (K x N), C is (M x N)
//
// Matrices are stored in row-major order:
//   A[i][j] = A[i * K + j]
//   B[i][j] = B[i * N + j]
//   C[i][j] = C[i * N + j]
//
// =============================================================================

// TODO 1: Naive GEMM - each thread computes one element of C
template <typename scalar_t>
__global__ void gemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N) {

    // Your code here

}

// TODO 2: GEMM using shared memory
#define BLOCK_SIZE 16

template <typename scalar_t>
__global__ void gemm_shared_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N) {

    // Your code here

}

// TODO 3: GEMM using shared memory + loop unrolling
template <typename scalar_t>
__global__ void gemm_unrolled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N) {

    // Your code here
    // Build on your shared memory implementation and apply #pragma unroll
    // to the inner computation loop(s).

}

// =============================================================================
// C++ Wrapper Functions
// =============================================================================

torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A columns must match B rows");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Adjust block/grid dimensions as needed for your kernel implementation
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "gemm_cuda", ([&] {
        gemm_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}

torch::Tensor gemm_shared_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A columns must match B rows");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Adjust block/grid dimensions as needed for your kernel implementation
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "gemm_shared_cuda", ([&] {
        gemm_shared_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}

torch::Tensor gemm_unrolled_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A columns must match B rows");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Adjust block/grid dimensions as needed for your kernel implementation
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "gemm_unrolled_cuda", ([&] {
        gemm_unrolled_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_cuda, "GEMM (CUDA) - naive implementation");
    m.def("gemm_shared", &gemm_shared_cuda, "GEMM (CUDA) - shared memory implementation");
    m.def("gemm_unrolled", &gemm_unrolled_cuda, "GEMM (CUDA) - shared memory + loop unrolling");
}
