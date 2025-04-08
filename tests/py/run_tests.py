#!/usr/bin/env python3

"""
Test runner for SparseMoE Python tests.

This script discovers and runs all test cases in the current directory.
"""

import unittest
import sys
import os

# Add the SparseMoE package to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

if __name__ == '__main__':
    # Discover and run all tests in current directory
    test_loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    test_suite = test_loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful())