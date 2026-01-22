import torch
import cuda_gemm
import time
import sys
import argparse

def test_naive():
    """Test Part A: Naive GEMM correctness."""
    print("=" * 50)
    print("Part A: Testing Naive GEMM correctness...")
    print("=" * 50)

    sizes = [(64, 64, 64), (128, 256, 512), (1000, 500, 750), (1024, 1024, 1024)]

    for M, K, N in sizes:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)

        C_ref = torch.mm(A, B)
        C_naive = cuda_gemm.gemm(A, B)

        naive_error = torch.max(torch.abs(C_ref - C_naive)).item()

        print(f"Size ({M}x{K}) x ({K}x{N}): max error = {naive_error:.6e}")

        if naive_error >= 1e-3:
            print(f"FAILED: Naive GEMM error too large: {naive_error}")
            sys.exit(1)

    print("\nPart A: All naive GEMM tests passed!")

def test_shared():
    """Test Part B: Shared memory GEMM correctness."""
    print("=" * 50)
    print("Part B: Testing Shared Memory GEMM correctness...")
    print("=" * 50)

    sizes = [(64, 64, 64), (128, 256, 512), (1000, 500, 750), (1024, 1024, 1024)]

    for M, K, N in sizes:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)

        C_ref = torch.mm(A, B)
        C_shared = cuda_gemm.gemm_shared(A, B)

        shared_error = torch.max(torch.abs(C_ref - C_shared)).item()

        print(f"Size ({M}x{K}) x ({K}x{N}): max error = {shared_error:.6e}")

        if shared_error >= 1e-3:
            print(f"FAILED: Shared GEMM error too large: {shared_error}")
            sys.exit(1)

    print("\nPart B: All shared memory GEMM tests passed!")

def benchmark():
    """Benchmark our GEMM against PyTorch."""
    print("\n" + "=" * 50)
    print("Benchmarking...")
    print("=" * 50)

    M, K, N = 1024, 2048, 512
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)

    # Check correctness first
    C_ref = torch.mm(A, B)
    C_naive = cuda_gemm.gemm(A, B)
    C_shared = cuda_gemm.gemm_shared(A, B)

    naive_error = torch.max(torch.abs(C_ref - C_naive)).item()
    shared_error = torch.max(torch.abs(C_ref - C_shared)).item()

    print(f"\nMatrix size: ({M}x{K}) x ({K}x{N})")
    print(f"Naive max error:  {naive_error:.6e}")
    print(f"Shared max error: {shared_error:.6e}")

    if naive_error >= 1e-3:
        print(f"FAILED: Naive GEMM error too large: {naive_error}")
        sys.exit(1)
    if shared_error >= 1e-3:
        print(f"FAILED: Shared GEMM error too large: {shared_error}")
        sys.exit(1)

    # Warmup
    for _ in range(10):
        _ = torch.mm(A, B)
        _ = cuda_gemm.gemm(A, B)
        _ = cuda_gemm.gemm_shared(A, B)

    torch.cuda.synchronize()

    n_runs = 10

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

    print(f"\nPyTorch mm:    {pytorch_time:.3f} ms")
    print(f"Naive GEMM:    {naive_time:.3f} ms")
    print(f"Shared GEMM:   {shared_time:.3f} ms")
    print(f"\nSpeedup (shared vs naive): {naive_time/shared_time:.2f}x")
    print(f"Relative to PyTorch (shared): {pytorch_time/shared_time:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CUDA GEMM implementations')
    parser.add_argument('--part-a', action='store_true', help='Test Part A: Naive GEMM only')
    parser.add_argument('--part-b', action='store_true', help='Test Part B: Shared memory GEMM only')
    args = parser.parse_args()

    if args.part_a:
        test_naive()
    elif args.part_b:
        test_shared()
    else:
        # No argument provided: run benchmark
        benchmark()
