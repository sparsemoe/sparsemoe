"""
Test script to validate the SparseMoE model with mock C++ bindings.
"""

import torch
import sys
import os
import json
import tempfile
import time

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from sparsemoe import create_model, SparseMoEConfig

def main():
    print("Testing SparseMoE model with mock C++ bindings")
    
    # Create a minimal config for testing
    config_data = {
        "vocab_size": 1000,
        "dim": 64,
        "inter_dim": 256,
        "moe_inter_dim": 128,
        "n_layers": 2,
        "n_dense_layers": 1,
        "n_heads": 4,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
        "max_seq_len": 128,
        "max_batch_size": 2,
        "use_cpp_backend": True,  # Use mock C++ backend
        "route_scale": 1.0,
        "score_func": "sigmoid",
        "q_lora_rank": 16,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "eos_token_id": 2
    }
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        json.dump(config_data, tmp)
        config_path = tmp.name
    
    try:
        # Create model with mock C++ backend
        print("\nCreating model with mock C++ backend...")
        model = create_model(config_path=config_path, use_cpp_backend=True)
        print(f"Model created with use_cpp_backend={model.config.use_cpp_backend}")
        
        # Generate random input for testing
        print("\nGenerating random input...")
        input_ids = torch.randint(0, config_data["vocab_size"], (1, 8))
        print(f"Input shape: {input_ids.shape}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        start_time = time.time()
        logits = model(input_ids)
        print(f"Forward pass completed in {time.time() - start_time:.4f}s")
        print(f"Output logits shape: {logits.shape}")
        
        # Test generation
        print("\nTesting text generation...")
        start_time = time.time()
        max_new_tokens = 4
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9
        )
        print(f"Generation completed in {time.time() - start_time:.4f}s")
        print(f"Generated output shape: {generated.shape}")
        print(f"Generated tokens: {generated[0].tolist()}")
        
        print("\nAll tests passed successfully!")
        
    finally:
        # Clean up the temporary file
        os.unlink(config_path)

if __name__ == "__main__":
    main()