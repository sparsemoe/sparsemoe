"""
Mock implementation of sparsemoe_bindings to simulate C++ integration.
This allows testing the hybrid architecture without requiring CUDA or C++ compilation.
"""

import torch
import numpy as np

# Flag to indicate this is a mock implementation
__is_mock__ = True
__version__ = "0.1.0-mock"
HAS_CUDA = False

class DynamicNeuralPathway:
    """Mock implementation of DynamicNeuralPathway."""
    
    def __init__(self, dim, num_experts, num_selected, group_size=1, 
                 score_func="sigmoid", route_scale=1.0):
        self.dim = dim
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.group_size = group_size
        self.score_func = score_func
        self.route_scale = route_scale
        
        print(f"[C++ Mock] Created DynamicNeuralPathway with {num_experts} experts")
        
    def forward(self, input_tensor, expert_weights, routing_logits=None):
        """Mock implementation of forward pass."""
        batch_size, seq_len, _ = input_tensor.shape
        
        print(f"[C++ Mock] DNPS forward: input shape {input_tensor.shape}")
        print(f"[C++ Mock] Using {len(expert_weights)} experts for forward pass")
        
        # Simulate C++ processing - just return zeros of the same shape as input
        return torch.zeros_like(input_tensor)
        
def rms_norm(input_tensor, weight, eps=1e-6):
    """Mock implementation of RMS normalization."""
    print(f"[C++ Mock] Running RMS norm on tensor of shape {input_tensor.shape}")
    
    # Implement simple RMS norm to simulate C++ implementation
    var = input_tensor.pow(2).mean(-1, keepdim=True)
    x = input_tensor * torch.rsqrt(var + eps)
    return x * weight

def apply_rotary_pos_emb(x, freqs_cis):
    """Mock implementation of rotary position embeddings."""
    print(f"[C++ Mock] Applying rotary embeddings to tensor of shape {x.shape}")
    # Just return the input unchanged for mock implementation
    return x