"""
setup.py — builds qwen_megakernel_C PyTorch extension for RTX 5090 (sm_120)

Usage (on the RTX 5090 server):
    pip install ninja
    python setup.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

TORCH_LIB = os.path.join(
    os.path.dirname(__import__("torch").__file__), "lib"
)

setup(
    name="qwen_megakernel_C",
    ext_modules=[
        CUDAExtension(
            name="qwen_megakernel_C",
            sources=[
                "/workspace/Qwen3-0.6B-megakernel-RTX-5090/Tools/megakernel/qwen_ops.cpp",
                "/workspace/Qwen3-0.6B-megakernel-RTX-5090/Tools/megakernel/megakernel_5090.cu",
            ],
            include_dirs=[
                "/workspace/Qwen3-0.6B-megakernel-RTX-5090/Tools/rmsnorm",
                "/workspace/Qwen3-0.6B-megakernel-RTX-5090/Tools/swiglu",
                ".",
            ],
            extra_compile_args={
                "cxx":  ["-O3"],
                "nvcc": [
                    "-arch=sm_120",
                    "-O3",
                    "--use_fast_math",
                    "-diag-suppress=550",
                    "--expt-relaxed-constexpr",
                ],
            },
            library_dirs=[TORCH_LIB],
            runtime_library_dirs=[TORCH_LIB],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)