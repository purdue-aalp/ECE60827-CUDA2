from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_gemm',
    ext_modules=[
        CUDAExtension(
            'cuda_gemm',
            ['cuda_gemm.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
