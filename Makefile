SHELL := /bin/bash
PYTHON = python3
MODULES = module load gcc/11.4.1 cuda ngc pytorch
SRUN = srun -c4 -A gpu --gres=gpu:1

.PHONY: all build clean test help

all: build

# Build the CUDA extension
build:
	$(SRUN) bash -c '$(MODULES) && $(PYTHON) setup.py build_ext --inplace'

# Run correctness and benchmark tests
test: build
	$(SRUN) bash -c '$(MODULES) && $(PYTHON) test_gemm.py'

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf *.egg-info/
	rm -f *.so
	rm -rf __pycache__/
	rm -rf *.pyc

# Show help
help:
	@echo "CUDA GEMM Homework"
	@echo ""
	@echo "Usage:"
	@echo "  make build   - Build the CUDA extension"
	@echo "  make test    - Run tests and benchmarks"
	@echo "  make clean   - Remove build artifacts"
	@echo ""
	@echo "Workflow:"
	@echo "  1. Implement kernels in cuda_gemm.cu"
	@echo "  2. make build"
	@echo "  3. make test"
