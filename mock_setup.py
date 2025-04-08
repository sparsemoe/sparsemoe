import os
import setuptools
from setuptools import setup, Extension
from torch.utils import cpp_extension

# Define the extension module
ext_modules = [
    cpp_extension.CppExtension(
        name='sparsemoe_bindings',
        sources=['core/mock_bindings.cpp'],
        extra_compile_args=['-std=c++14'],
    )
]

setup(
    name='sparsemoe_bindings',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)