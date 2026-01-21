import torch
import cuda_gemm
import time

def test_correctness():
    """Test that our GEMM produces correct results compared to PyTorch."""
    print("=" * 50)
    print("Testing correctness...")
    print("=" * 50)

    # Test various sizes
    sizes = [(64, 64, 64), (128, 256, 512), (1000, 500, 750), (1024, 1024, 1024)]

    for M, K, N in sizes:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)

        # PyTorch reference
        C_ref = torch.mm(A, B)

        # Our naive implementation
        C_naive = cuda_gemm.gemm(A, B)

        # Our shared memory implementation
        C_shared = cuda_gemm.gemm_shared(A, B)

        # Check correctness
        naive_error = torch.max(torch.abs(C_ref - C_naive)).item()
        shared_error = torch.max(torch.abs(C_ref - C_shared)).item()

        print(f"Size ({M}x{K}) x ({K}x{N}):")
        print(f"  Naive max error:  {naive_error:.6e}")
        print(f"  Shared max error: {shared_error:.6e}")

        # Assert correctness (allowing for floating point tolerance)
        assert naive_error < 1e-3, f"Naive GEMM error too large: {naive_error}"
        assert shared_error < 1e-3, f"Shared GEMM error too large: {shared_error}"

    print("\nAll correctness tests passed!")

def benchmark():
    """Benchmark our GEMM against PyTorch."""
    print("\n" + "=" * 50)
    print("Benchmarking...")
    print("=" * 50)

    M, K, N = 2048, 2048, 2048
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = torch.mm(A, B)
        _ = cuda_gemm.gemm(A, B)
        _ = cuda_gemm.gemm_shared(A, B)

    torch.cuda.synchronize()

    n_runs = 100

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = torch.mm(A, B)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_runs * 1000

    # Benchmark naive GEMM
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = cuda_gemm.gemm(A, B)
    torch.cuda.synchronize()
    naive_time = (time.perf_counter() - start) / n_runs * 1000

    # Benchmark shared memory GEMM
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = cuda_gemm.gemm_shared(A, B)
    torch.cuda.synchronize()
    shared_time = (time.perf_counter() - start) / n_runs * 1000

    print(f"\nMatrix size: ({M}x{K}) x ({K}x{N})")
    print(f"PyTorch mm:    {pytorch_time:.3f} ms")
    print(f"Naive GEMM:    {naive_time:.3f} ms")
    print(f"Shared GEMM:   {shared_time:.3f} ms")
    print(f"\nSpeedup (shared vs naive): {naive_time/shared_time:.2f}x")
    print(f"Relative to PyTorch (shared): {pytorch_time/shared_time:.2f}x")

if __name__ == "__main__":
    test_correctness()
    benchmark()
