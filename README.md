# SparseMoE: Sparse Mixture-of-Experts Language Model

SparseMoE is a high-performance language model implementation that uses a Sparse Mixture-of-Experts (SMoE) architecture with a hybrid C++/Python approach for optimal performance.

## Current Status and Limitations

**What Works:**
- Core model architecture with Dynamic Neural Pathway Selection
- Mock C++ bindings for testing without CUDA
- Model inference with various configurations
- Basic text generation capabilities
- Comprehensive Python test suite
- Pure Python fallback for all operations

**What Doesn't Work Yet:**
- Actual C++ extensions and CUDA kernels (currently using mock implementations)
- Quantization features (INT8/BF16 support planned but not implemented)
- Training capabilities (inference-only at present)
- Full-scale model loading (parameters architecture works but no pre-trained weights)
- Distributed inference across multiple GPUs

**Minimum Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CPU-only operation works with mock bindings
- For full performance: CUDA-capable GPU (when C++ is implemented)

## Overview

SparseMoE features Dynamic Neural Pathway Selection (DNPS), which selectively routes inputs to appropriate experts, activating only a small fraction of parameters per token. This architecture allows us to build larger models with better efficiency compared to dense models.

Key features:
- Hybrid C++/Python implementation design pattern (with mock C++ currently)
- Dynamic Neural Pathway Selection for adaptive expert routing
- Flexible attention mechanism with rotary position embeddings
- Configurable model sizes from 16B to 1024B parameters (architecture only)
- Planned: INT8/BF16 quantization, CUDA optimization (not yet implemented)

## Architecture

![Architecture Diagram](docs/architecture.md)

SparseMoE consists of several key components:

1. **Core C++ Implementation**
   - High-performance CUDA kernels for speed-critical operations
   - Optimized attention and DNPS implementations
   - PyBind11 bindings for integration with Python

2. **Python API**
   - High-level API for model configuration and usage
   - Integration with PyTorch for training and inference
   - Fallback pure Python implementations for development without CUDA

3. **Model Components**
   - Transformer blocks with attention and feed-forward layers
   - Dynamic Neural Pathway Selection (DNPS) for expert routing
   - Efficient attention with rotary position embeddings and KV caching

## Getting Started

### Installation

```bash
# Install with mock C++ bindings (for testing without CUDA)
pip install -e . --config-settings="--mock"

# Install with C++ extensions (requires CUDA - NOT WORKING YET)
# pip install -e .
```

### Basic Usage

```python
import torch
from sparsemoe import create_model

# Create a small model for testing (architecture only, not trained)
model = create_model(model_size="16B", use_cpp_backend=False)

# Or create a minimal config for faster testing
from sparsemoe import SparseMoEConfig, SparseMoEModel
config = SparseMoEConfig(
    vocab_size=1000,
    dim=64,
    n_layers=2,
    n_heads=4,
    n_routed_experts=8,
    use_cpp_backend=True  # Will use mock backend
)
model = SparseMoEModel(config)

# Run inference (with random outputs since model isn't trained)
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Example input
logits = model(input_ids)

# Generate text (also random outputs)
generated = model.generate(input_ids, max_new_tokens=10)
```

### Current Capabilities and Limitations

**Model Architecture:**
- ✅ Transformer-based architecture
- ✅ Dynamic Neural Pathway Selection
- ✅ Configurable model sizes and parameters
- ✅ Attention with rotary position embeddings
- ✅ KV caching for efficient inference
- ✅ Expert-based feed-forward networks
- ❌ Pre-trained weights (not available)
- ❌ Actual CUDA optimization (mock implementation only)

**Performance:**
- ✅ Works on CPU (slow, for testing only)
- ✅ Basic inference functionality
- ✅ Text generation capability
- ❌ GPU acceleration (planned but not implemented)
- ❌ Quantization for efficiency (planned but not implemented)
- ❌ Optimized for production (development version only)

**Integration:**
- ✅ Pure Python interface
- ✅ Compatible with PyTorch ecosystem
- ✅ Configurable through JSON or parameters
- ❌ Hugging Face transformers integration (not implemented)
- ❌ ONNX export capability (not implemented)
- ❌ Server deployment options (not implemented)

## Testing with Mock C++ Bindings

For development and testing without requiring CUDA, we provide mock C++ bindings that simulate the behavior of the actual C++ implementation:

```bash
# Run test script for mock C++ bindings
python test_mock_cpp.py

# Run Python unit tests
python -m unittest discover -s tests/py
```

The mock implementation allows developing and testing the model architecture on systems without CUDA support.

## Configuration

Model sizes and parameters can be configured either through JSON config files or programmatically:

```python
from sparsemoe import SparseMoEConfig, SparseMoEModel

# Create custom configuration
config = SparseMoEConfig(
    dim=4096,                # Hidden dimension
    n_layers=32,             # Number of transformer layers
    n_heads=32,              # Number of attention heads
    n_routed_experts=128,    # Number of routed experts
    n_activated_experts=8,   # Number of experts activated per token
    vocab_size=32000,        # Vocabulary size
    use_cpp_backend=True     # Whether to use C++ backend
)

# Create model with custom config
model = SparseMoEModel(config)
```

## Development

### Implemented Features

The current implementation focuses on core architecture components:

1. **SparseMoEModel**: Base model with transformer layers
2. **DNPSLayer**: Dynamic Neural Pathway Selection implementation
3. **DynamicAttention**: Attention mechanism with rotary embeddings
4. **Mock C++ Integration**: For testing the overall architecture

### Missing Features

Several planned features are still pending implementation:

1. **Actual C++ Backend**: Real CUDA kernels and optimizations
2. **Training Pipeline**: Currently inference-only
3. **Tokenizer Integration**: No tokenizer implementation yet
4. **Model Weights**: No pre-trained weights available
5. **Larger Scale Testing**: Not tested at full scale (1T+ parameters)

### Testing

```bash
# Run all tests
python tests/run_all.py

# Run only Python tests
python tests/py/run_tests.py

# Test the mock C++ bindings specifically
python test_mock_cpp.py
```

### Development Roadmap

1. **Phase 1 (Current)**: Architecture implementation with mock C++
2. **Phase 2**: Implement CUDA kernels for critical operations
3. **Phase 3**: Training infrastructure and weight initialization
4. **Phase 4**: Performance optimization and scaling tests
5. **Phase 5**: Tokenizer integration and full model training

## Known Issues

- Mock implementation returns zeros for DNPS outputs, so generation is random
- KV cache implementation may have memory issues for very long sequences
- Attention mask handling has edge cases with variable sequence lengths
- CPU-only operation is very slow due to lack of optimizations
- Model does not correctly calculate model sizes based on parameters

## License

[MIT License](LICENSE)