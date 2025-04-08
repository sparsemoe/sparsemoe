#!/usr/bin/env python3
"""
Demonstration of SparseMoE with Adaptive Experts

This script demonstrates the unique adaptive experts feature of SparseMoE,
which dynamically adjusts expert capacity based on usage patterns.
"""

import torch
import sys
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from sparsemoe import SparseMoEConfig, SparseMoEModel


def create_adaptive_model(hidden_dim=64, num_experts=8, num_layers=4):
    """Create a small model with adaptive experts for demonstration."""
    config = SparseMoEConfig(
        vocab_size=1000,
        dim=hidden_dim,
        n_layers=num_layers,
        n_dense_layers=1,  # First layer is dense, rest use adaptive experts
        n_heads=4,
        n_routed_experts=num_experts,
        n_activated_experts=2,
        expert_type="adaptive",  # Use adaptive experts
        adaptive_initial_dim=hidden_dim * 2,  # Initial hidden dimension
        adaptive_min_dim=hidden_dim,          # Minimum hidden dimension
        adaptive_max_dim=hidden_dim * 4,      # Maximum hidden dimension
        adaptive_scaling_interval=50,         # Scale more frequently for demo
        use_cpp_backend=False  # Use pure Python implementation
    )
    
    return SparseMoEModel(config)


def create_fixed_model(hidden_dim=64, num_experts=8, num_layers=4):
    """Create a model with fixed experts for comparison."""
    config = SparseMoEConfig(
        vocab_size=1000,
        dim=hidden_dim,
        n_layers=num_layers,
        n_dense_layers=1,
        n_heads=4,
        n_routed_experts=num_experts,
        n_activated_experts=2,
        expert_type="fixed",  # Use fixed experts
        moe_inter_dim=hidden_dim * 2,
        use_cpp_backend=False
    )
    
    return SparseMoEModel(config)


def simulate_biased_inputs(model, num_steps=200, batch_size=4, seq_len=16):
    """
    Simulate inputs that intentionally bias toward certain experts
    to demonstrate adaptive scaling.
    """
    print(f"Running simulation with {num_steps} steps...")
    device = next(model.parameters()).device
    
    for step in range(num_steps):
        # Create input that will bias toward specific experts
        # We'll use different token patterns in each quarter of the simulation
        if step < num_steps // 4:
            # Bias toward first few experts
            input_ids = torch.randint(0, 200, (batch_size, seq_len), device=device)
        elif step < num_steps // 2:
            # Bias toward middle experts
            input_ids = torch.randint(200, 400, (batch_size, seq_len), device=device)
        elif step < 3 * num_steps // 4:
            # Bias toward last experts
            input_ids = torch.randint(400, 600, (batch_size, seq_len), device=device)
        else:
            # Mix of all experts
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            
        # Forward pass and backward pass
        with torch.no_grad():
            model(input_ids)
        
        # Print progress
        if step % 20 == 0:
            print(f"Step {step}/{num_steps}")


def plot_expert_sizes(model):
    """Plot expert sizes and importance to visualize adaptation."""
    stats = model.get_expert_stats()
    
    plt.figure(figsize=(15, 10))
    
    layers = list(stats.keys())
    for i, layer_name in enumerate(layers):
        layer_stats = stats[layer_name]
        
        if "expert_sizes" not in layer_stats:
            continue
            
        plt.subplot(len(layers), 1, i+1)
        
        sizes = layer_stats["expert_sizes"]
        importance = layer_stats["importance_scores"]
        
        x = np.arange(len(sizes))
        plt.bar(x, sizes, alpha=0.7, label='Hidden Dimension')
        
        # Add importance scores
        plt.plot(x, importance, 'ro-', label='Importance')
        
        plt.title(f"{layer_name} Expert Sizes and Importance")
        plt.xlabel("Expert Index")
        plt.ylabel("Size / Importance")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("expert_adaptation.png")
    plt.close()
    
    print("Expert size visualization saved to 'expert_adaptation.png'")


def main():
    parser = argparse.ArgumentParser(description="Demonstrate adaptive experts in SparseMoE")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--compare", action="store_true", help="Compare with fixed experts")
    args = parser.parse_args()
    
    # Create model with adaptive experts
    print("Creating model with adaptive experts...")
    adaptive_model = create_adaptive_model(
        hidden_dim=args.dim,
        num_experts=args.experts,
        num_layers=args.layers
    )
    
    # Run simulation with biased inputs
    print("\nRunning simulation with adaptive experts...")
    start_time = time.time()
    simulate_biased_inputs(adaptive_model, num_steps=args.steps)
    adaptive_time = time.time() - start_time
    
    # Visualize expert adaptation
    print("\nVisualizing expert adaptation...")
    adaptive_model.visualize_expert_scaling()
    plot_expert_sizes(adaptive_model)
    
    # Print stats
    stats = adaptive_model.get_expert_stats()
    for layer_name, layer_stats in stats.items():
        if "expert_sizes" in layer_stats:
            print(f"\n{layer_name} expert sizes:")
            for i, size in enumerate(layer_stats["expert_sizes"]):
                print(f"  Expert {i}: {size} (importance: {layer_stats['importance_scores'][i]:.3f})")
                
    # If comparison requested, run with fixed experts too
    if args.compare:
        print("\nCreating model with fixed experts for comparison...")
        fixed_model = create_fixed_model(
            hidden_dim=args.dim,
            num_experts=args.experts,
            num_layers=args.layers
        )
        
        print("\nRunning simulation with fixed experts...")
        start_time = time.time()
        simulate_biased_inputs(fixed_model, num_steps=args.steps)
        fixed_time = time.time() - start_time
        
        # Compare performance
        print("\nPerformance comparison:")
        print(f"  Adaptive experts: {adaptive_time:.2f} seconds")
        print(f"  Fixed experts: {fixed_time:.2f} seconds")
        
        # Compare parameter counts
        adaptive_params = sum(p.numel() for p in adaptive_model.parameters())
        fixed_params = sum(p.numel() for p in fixed_model.parameters())
        
        print("\nParameter count comparison:")
        print(f"  Adaptive experts: {adaptive_params:,} parameters")
        print(f"  Fixed experts: {fixed_params:,} parameters")
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()