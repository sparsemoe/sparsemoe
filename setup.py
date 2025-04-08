import os
import sys
import subprocess
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Get long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Check if we're in mock mode
use_mock = '--mock' in sys.argv
if use_mock:
    sys.argv.remove('--mock')
    print("Using mock C++ bindings for testing...")

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # If using mock bindings, don't attempt to build C++ extensions
        if use_mock:
            self.install_mock_bindings()
            return
            
        # Check if CMake is installed
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            print("CMake not found. Using mock C++ bindings instead.")
            self.install_mock_bindings()
            return

        # Try to build extensions, fall back to mock if it fails
        try:
            for ext in self.extensions:
                self.build_extension(ext)
        except Exception as e:
            print(f"Failed to build C++ extensions: {e}")
            print("Falling back to mock C++ bindings...")
            self.install_mock_bindings()

    def build_extension(self, ext):
        # Set directories
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release'
        ]

        # Create build directory
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)
        
        # Run CMake and build
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'], cwd=build_temp)
        
        print(f"Successfully built C++ extension at {extdir}")
        
    def install_mock_bindings(self):
        """Install mock C++ bindings instead of building real ones."""
        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            os.makedirs(extdir, exist_ok=True)
            
            # Create a __init__.py that imports the mock implementation
            init_file = os.path.join(extdir, "__init__.py")
            with open(init_file, 'w') as f:
                f.write("# Mock C++ bindings\n")
                f.write("from sparsemoe.mock_sparsemoe_bindings import *\n")
                
            print(f"Installed mock C++ bindings at {extdir}")

setup(
    name="sparsemoe",
    version="0.1.0",
    author="SparseMoE Team",
    author_email="support@sparsemoe.ai",
    description="High-performance language model with hybrid C++/Python implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sparsemoe/sparsemoe",
    packages=find_packages(where="py"),
    package_dir={"": "py"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "transformers>=4.30.0",
        "safetensors>=0.3.0",
    ],
    ext_modules=[CMakeExtension("sparsemoe.sparsemoe_bindings", sourcedir="core")],
    cmdclass={"build_ext": CMakeBuild},
    scripts=[
        "py/scripts/sparsemoe-train",
        "py/scripts/sparsemoe-generate",
    ],
)