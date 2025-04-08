# SparseMoE Testing

This directory contains tests for the SparseMoE project, organized into Python tests and C++ tests.

## Test Organization

- `py/`: Python unit tests
  - `test_model.py`: Tests for the SparseMoEModel class
  - `test_dnps.py`: Tests for the Dynamic Neural Pathway Selection
  - `test_attention.py`: Tests for attention mechanisms
  - `test_config.py`: Tests for configuration handling
  - `test_integration.py`: Integration tests for the full model
  - `run_tests.py`: Script to run all Python tests

- `core/`: C++ unit tests
  - `test_dnps.cpp`: Tests for C++ DNPS implementation
  - `test_attention.cpp`: Tests for C++ attention implementation
  - `CMakeLists.txt`: Build configuration for C++ tests

- `run_all.py`: Master script to run all tests (both Python and C++)

## Mock C++ Bindings

Since the project uses a hybrid C++/Python approach with CUDA-optimized kernels, running tests on systems without CUDA can be challenging. We've implemented mock C++ bindings to simulate the behavior of the C++ components without requiring actual CUDA compilation.

The mock bindings are located in `py/sparsemoe/mock_sparsemoe_bindings.py` and simulate:
- Dynamic Neural Pathway Selection
- RMS Normalization
- Rotary Position Embeddings

To run tests with mock bindings:

```bash
# Run specific mock tests
python test_mock_cpp.py

# Run model test with mock bindings
python tests/model_mock_test.py

# Run all Python tests
python -m unittest discover -s tests/py
```

## Debugging Tips

- If you get errors related to incorrect shapes or dimensions, check the torch.einsum operations carefully
- For attention mask errors, verify that the mask shape matches the sequence length in all dimensions
- Mock bindings output debug information with [C++ Mock] prefix to help diagnose issues

## Current Test Status

- Python tests: All passing (with CUDA tests skipped on non-CUDA platforms)
- C++ tests: Not implemented yet (requires actual CUDA access)
- Mock integration: Working correctly with full model inference and generation

## Future Testing Improvements

- Add more comprehensive test cases for edge conditions
- Implement test fixtures for common setup code
- Add tests for quantization features
- Improve generation testing with validation of coherent outputs
- Create CI pipeline with Docker + CUDA for comprehensive testing