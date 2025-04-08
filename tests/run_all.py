#!/usr/bin/env python3

"""
Main test runner for SparseMoE.

This script runs all tests in the project, both Python and C++ tests.
"""

import os
import sys
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run all SparseMoE tests")
    parser.add_argument("--python-only", action="store_true", help="Run only Python tests")
    parser.add_argument("--cpp-only", action="store_true", help="Run only C++ tests")
    return parser.parse_args()

def run_python_tests():
    """Run Python tests."""
    print("=" * 80)
    print("Running Python Tests")
    print("=" * 80)
    
    python_test_dir = os.path.join(os.path.dirname(__file__), "py")
    python_test_script = os.path.join(python_test_dir, "run_tests.py")
    
    result = subprocess.run([sys.executable, python_test_script])
    return result.returncode == 0

def run_cpp_tests():
    """Build and run C++ tests."""
    print("=" * 80)
    print("Building and Running C++ Tests")
    print("=" * 80)
    
    # Get test directory
    cpp_test_dir = os.path.join(os.path.dirname(__file__), "core")
    
    # Create build directory
    build_dir = os.path.join(cpp_test_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    # Configure with CMake
    print("Configuring CMake...")
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    
    # Build tests
    print("Building tests...")
    subprocess.run(["cmake", "--build", "."], cwd=build_dir, check=True)
    
    # Run tests
    print("Running tests...")
    test_result_dnps = subprocess.run(["./test_dnps"], cwd=build_dir)
    test_result_attention = subprocess.run(["./test_attention"], cwd=build_dir)
    
    return test_result_dnps.returncode == 0 and test_result_attention.returncode == 0

if __name__ == "__main__":
    args = parse_args()
    
    success = True
    
    # Run tests based on command-line arguments
    if args.python_only:
        success = run_python_tests()
    elif args.cpp_only:
        success = run_cpp_tests()
    else:
        # Run both Python and C++ tests
        python_success = run_python_tests()
        cpp_success = run_cpp_tests()
        success = python_success and cpp_success
    
    # Print summary
    print("=" * 80)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed. See above output for details.")
        sys.exit(1)