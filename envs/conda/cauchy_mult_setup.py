from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        "cauchy_mult",
        [
            "cauchy.cpp",
            "cauchy_cuda.cu",
        ],
        extra_compile_args={
            "cxx": ["-g", "-march=native", "-funroll-loops"],
            "nvcc": ["-O2", "-lineinfo", "--use_fast_math"],
        },
    )
]

setup(
    name="cauchy_mult", ext_modules=ext_modules, cmdclass={"build_ext": BuildExtension}
)
