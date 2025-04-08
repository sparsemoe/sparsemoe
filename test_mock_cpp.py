"""
Test script to verify that the mock C++ integration is working.
"""

import torch
import sparsemoe
from sparsemoe import mock_sparsemoe_bindings

# Test if our mock C++ bindings are accessible
print(f"SparseMoE version: {sparsemoe.__version__}")
print(f"C++ backend available: {sparsemoe.HAS_CPP_BACKEND}")
print(f"Mock bindings has version attribute: {hasattr(mock_sparsemoe_bindings, '__version__')}")
if hasattr(mock_sparsemoe_bindings, '__version__'):
    print(f"Mock bindings version: {mock_sparsemoe_bindings.__version__}")

# Test the RMS norm from C++ bindings
print("\nTesting mock C++ RMS norm:")
x = torch.randn(2, 4, 16)
w = torch.ones(16)
result = mock_sparsemoe_bindings.rms_norm(x, w, 1e-6)
print(f"RMS norm result shape: {result.shape}")

# Test the Dynamic Neural Pathway from C++ bindings
print("\nTesting mock C++ Dynamic Neural Pathway:")
dnps = mock_sparsemoe_bindings.DynamicNeuralPathway(16, 8, 2, 1, "sigmoid", 1.0)
input_tensor = torch.randn(2, 4, 16)
expert_weights = [torch.randn(16, 64) for _ in range(8)]
result = dnps.forward(input_tensor, expert_weights)
print(f"DNPS forward result shape: {result.shape}")

# Test using the DNPSLayer with C++ backend
print("\nTesting DNPSLayer with mock C++ backend:")
dnps_layer = sparsemoe.DNPSLayer(
    dim=16,
    num_experts=8,
    num_selected=2,
    use_cpp_backend=True
)
result = dnps_layer(input_tensor)
print(f"DNPSLayer forward result shape: {result.shape}")

print("\nAll tests completed successfully!")