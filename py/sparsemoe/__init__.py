"""
SparseMoE - High-performance language model with hybrid C++/Python implementation
"""

__version__ = "0.1.0"

# Import mock_sparsemoe_bindings for testing the hybrid approach
from . import mock_sparsemoe_bindings

# Core components
from .model import SparseMoEModel, SparseMoEConfig
from .dnps import DNPSLayer

# Set HAS_CPP_BACKEND to True since we're using mock bindings
HAS_CPP_BACKEND = True
print("Using mock C++ bindings for testing hybrid architecture")

# Convenient function to create models
def create_model(config_path=None, model_size="16B", use_cpp_backend=True):
    """
    Create a SparseMoE model with the specified configuration.
    
    Args:
        config_path: Path to a JSON configuration file
        model_size: Model size if config_path is not provided
        use_cpp_backend: Whether to use C++ backend when available
        
    Returns:
        SparseMoEModel instance
    """
    if config_path:
        config = SparseMoEConfig.from_json(config_path)
    else:
        config = SparseMoEConfig()
        
        # Set parameters based on model size
        if model_size == "1024B":
            config.dim = 8192
            config.inter_dim = 24576
            config.moe_inter_dim = 3072
            config.n_layers = 80
            config.n_dense_layers = 4
            config.n_heads = 192
            config.n_routed_experts = 384
            config.n_shared_experts = 2
            config.n_activated_experts = 12
            config.n_expert_groups = 12
            config.n_limited_groups = 6
            config.q_lora_rank = 2048
            config.kv_lora_rank = 768
        elif model_size == "671B":
            config.dim = 7168
            config.inter_dim = 18432
            config.moe_inter_dim = 2048
            config.n_layers = 61
            config.n_dense_layers = 3
            config.n_heads = 128
            config.n_routed_experts = 256
            config.n_shared_experts = 1
            config.n_activated_experts = 8
            config.n_expert_groups = 8
            config.n_limited_groups = 4
            config.q_lora_rank = 1536
            config.kv_lora_rank = 512
        elif model_size == "236B":
            config.dim = 4096
            config.inter_dim = 14336
            config.moe_inter_dim = 1536
            config.n_layers = 46
            config.n_dense_layers = 2
            config.n_heads = 64
            config.n_routed_experts = 128
            config.n_shared_experts = 1
            config.n_activated_experts = 8
            config.n_expert_groups = 4
            config.n_limited_groups = 2
            config.q_lora_rank = 1024
            config.kv_lora_rank = 384
        # Default is 16B
            
    config.use_cpp_backend = use_cpp_backend and HAS_CPP_BACKEND
    
    return SparseMoEModel(config)