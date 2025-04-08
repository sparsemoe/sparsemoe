import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dnps import DNPSLayer
from .adaptive_experts import AdaptiveExpertLayer

@dataclass
class SparseMoEConfig:
    """Configuration for SparseMoE model."""
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5
    score_func: str = "sigmoid"
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    max_seq_len: int = 4096 * 32
    max_batch_size: int = 8
    use_cpp_backend: bool = True
    eos_token_id: int = 2
    
    # Adaptive expert configuration
    use_adaptive_experts: bool = False  # Whether to use the adaptive expert mechanism
    adaptive_initial_dim: int = 2048    # Initial hidden dimension for adaptive experts
    adaptive_min_dim: int = 512         # Minimum hidden dimension for adaptive experts
    adaptive_max_dim: int = 4096        # Maximum hidden dimension for adaptive experts
    adaptive_scaling_interval: int = 1000  # How often to check for expert rescaling
    adaptive_router_capacity: float = 1.5  # Router capacity factor
    expert_type: Literal["fixed", "adaptive"] = "fixed"  # Type of expert to use
    
    @classmethod
    def from_json(cls, path: str) -> "SparseMoEConfig":
        """Load config from a JSON file."""
        with open(path, 'r') as f:
            return cls(**json.load(f))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        # Use mock C++ bindings for RMS norm
        from sparsemoe.mock_sparsemoe_bindings import rms_norm
        return rms_norm(x, self.weight, self.eps)
        
        # Fallback to Python implementation
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class DynamicAttention(nn.Module):
    """Dynamic Attention layer with rotary embeddings."""
    
    def __init__(self, config: SparseMoEConfig):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.use_cpp_backend = config.use_cpp_backend
        
        # Initialize query projection
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        
        # Initialize key-value projection
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        
        # Initialize output projection
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        
        # Initialize attention scaling factor
        self.softmax_scale = self.qk_head_dim ** -0.5
        
        # Initialize C++ backend if available
        try:
            import sparsemoe.sparsemoe_bindings as cpp
            if hasattr(cpp, "DynamicAttention") and self.use_cpp_backend:
                self.cpp_impl = cpp.DynamicAttention(
                    self.dim,
                    self.n_heads,
                    self.qk_head_dim,
                    config.max_seq_len,
                    config.max_batch_size,
                    self.softmax_scale
                )
            else:
                self.cpp_impl = None
        except (ImportError, NotImplementedError):
            self.cpp_impl = None
            
        # Initialize KV cache for Python implementation if needed
        if self.cpp_impl is None:
            self.register_buffer("k_cache", torch.zeros(
                config.max_batch_size, config.max_seq_len, self.n_heads, self.qk_head_dim
            ), persistent=False)
            self.register_buffer("v_cache", torch.zeros(
                config.max_batch_size, config.max_seq_len, self.n_heads, self.v_head_dim
            ), persistent=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the attention layer."""
        batch_size, seq_len, _ = x.shape
        
        # Project query
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        # Reshape query and split into non-positional and rotary parts
        q = q.view(batch_size, seq_len, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Apply rotary embeddings using mock C++ bindings
        from sparsemoe.mock_sparsemoe_bindings import apply_rotary_pos_emb
        if self.use_cpp_backend:
            q_pe = apply_rotary_pos_emb(q_pe, freqs_cis)
        else:
            q_pe = self._apply_rotary_emb(q_pe, freqs_cis)
        
        # Project key-value
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Apply rotary embeddings to key positional part using mock C++ bindings
        from sparsemoe.mock_sparsemoe_bindings import apply_rotary_pos_emb
        if self.use_cpp_backend:
            k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), freqs_cis)
        else:
            k_pe = self._apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # Use C++ backend for attention if available
        if self.cpp_impl is not None:
            try:
                # Process key-value
                kv = self.kv_norm(kv)
                kv = self.wkv_b(kv)
                kv = kv.view(batch_size, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
                
                # Update KV cache
                self.cpp_impl.update_kv_cache(k, v, start_pos)
                
                # Combine query parts
                q = torch.cat([q_nope, q_pe], dim=-1)
                
                # Compute attention and return output
                end_pos = start_pos + seq_len
                return self.cpp_impl.forward_decoder(q, start_pos, mask)
            except Exception as e:
                print(f"C++ backend failed with error: {e}. Falling back to Python implementation.")
        
        # Fallback to Python implementation
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.kv_norm(kv)
        kv = self.wkv_b(kv)
        kv = kv.view(batch_size, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
        
        # Update KV cache
        end_pos = start_pos + seq_len
        self.k_cache[:batch_size, start_pos:end_pos] = k
        self.v_cache[:batch_size, start_pos:end_pos] = v
        
        # Compute attention scores
        scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:batch_size, :end_pos]) * self.softmax_scale
        
        # Apply mask if provided
        if mask is not None:
            # Make sure mask matches the dimension of the scores
            # scores shape is [batch_size, seq_len, n_heads, kv_seq_len]
            # kv_seq_len is the sequence length of the key-value cache
            kv_seq_len = scores.size(-1)
            if mask.size(-1) != kv_seq_len:
                # Adjust the mask to match the KV cache length
                seq_mask = torch.zeros((mask.size(0), kv_seq_len), device=mask.device).fill_(float("-inf"))
                seq_mask[:, -mask.size(1):] = mask
                mask = seq_mask
            
            # Add mask to scores
            scores += mask.unsqueeze(1)
        
        # Compute attention weights
        scores = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
        
        # Compute weighted sum of values
        output = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:batch_size, :end_pos])
        
        # Project to output dimension
        return self.wo(output.reshape(batch_size, seq_len, -1))
    
    def _apply_rotary_emb(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embeddings."""
        # Convert to complex representation
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # Reshape freqs_cis for broadcasting
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
        
        # Apply complex multiplication
        x_rotated = x_complex * freqs_cis
        
        # Convert back to real representation
        return torch.view_as_real(x_rotated).flatten(3).to(x.dtype)


class TransformerBlock(nn.Module):
    """Transformer block with attention and expert layers."""
    
    def __init__(self, layer_id: int, config: SparseMoEConfig):
        super().__init__()
        self.attn = DynamicAttention(config)
        
        # For first few layers, use MLP instead of expert layers
        if layer_id < config.n_dense_layers:
            self.ffn = nn.Sequential(
                nn.Linear(config.dim, config.inter_dim, bias=False),
                nn.SiLU(),
                nn.Linear(config.inter_dim, config.dim, bias=False)
            )
        else:
            # Choose expert type based on configuration
            if config.expert_type == "adaptive":
                # Use adaptive experts with dynamic scaling
                self.ffn = AdaptiveExpertLayer(
                    dim=config.dim,
                    num_experts=config.n_routed_experts,
                    num_selected=config.n_activated_experts,
                    initial_expert_dim=config.adaptive_initial_dim,
                    min_expert_dim=config.adaptive_min_dim,
                    max_expert_dim=config.adaptive_max_dim,
                    scaling_interval=config.adaptive_scaling_interval,
                    router_capacity_factor=config.adaptive_router_capacity
                )
            else:
                # Use standard fixed-size experts
                self.ffn = DNPSLayer(
                    dim=config.dim,
                    num_experts=config.n_routed_experts,
                    num_selected=config.n_activated_experts,
                    group_size=config.n_expert_groups,
                    score_func=config.score_func,
                    route_scale=config.route_scale,
                    moe_dim=config.moe_inter_dim,
                    num_shared_experts=config.n_shared_experts,
                    use_cpp_backend=config.use_cpp_backend
                )
        
        # Layer normalization
        self.attn_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the transformer block."""
        # Attention with residual connection
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.ffn_norm(x))
        
        return x


class SparseMoEModel(nn.Module):
    """SparseMoE large language model."""
    
    def __init__(self, config: SparseMoEConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        
        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(i, config) for i in range(config.n_layers)
        ])
        
        # Final layer normalization
        self.norm = RMSNorm(config.dim)
        
        # Output projection
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Precompute rotary embeddings
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(
            config.qk_rope_head_dim,
            config.max_seq_len
        ))
    
    def _precompute_freqs_cis(self, dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
        """Precompute complex exponential values for rotary embeddings."""
        # Compute frequency
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # Compute phases
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        
        # Complex exponential
        return torch.polar(torch.ones_like(freqs), freqs)
    
    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Forward pass for training or inference."""
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.max_seq_len, f"Sequence length exceeds model limit: {seq_len} > {self.max_seq_len}"
        
        # Token embeddings
        h = self.embed(tokens)
        
        # Get relevant slice of rotary embeddings
        freqs_cis = self.freqs_cis[start_pos:start_pos+seq_len]
        
        # Create attention mask for causal attention
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
        
        # Forward through transformer blocks
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        
        # Final layer norm
        h = self.norm(h)
        
        # Return logits for last position only during inference
        if not self.training:
            h = h[:, -1]
            
        return self.head(h)
    
    @torch.no_grad()
    def generate(
        self, 
        tokens: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_p: float = 0.95
    ) -> torch.Tensor:
        """Generate text using the model."""
        batch_size, seq_len = tokens.shape
        max_len = min(self.max_seq_len, seq_len + max_new_tokens)
        generated = tokens.clone()
        
        # Track which sequences have completed
        completed = torch.zeros(batch_size, dtype=torch.bool, device=tokens.device)
        
        for pos in range(seq_len, max_len):
            # Get input slice
            input_slice = generated[:, max(0, pos-seq_len):pos]
            
            # Forward pass
            logits = self.forward(input_slice, max(0, pos-seq_len))
            
            # Simple sampling or greedy decoding
            if temperature > 0:
                logits = logits / temperature
                
                # Make sure we're working with the right dimensions for multinomial
                # Eval mode should return [batch_size, vocab_size]
                if len(logits.shape) == 2:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    # If we're in training mode with [batch_size, seq_len, vocab_size]
                    # Just use argmax for simplicity during testing
                    next_token = logits[:, -1].argmax(dim=-1)
            else:
                # Greedy decoding
                if len(logits.shape) == 2:
                    next_token = logits.argmax(dim=-1)
                else:
                    next_token = logits[:, -1].argmax(dim=-1)
            
            # Append next token
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            
            # Check for completed sequences
            completed = completed | (next_token == self.config.eos_token_id)
            if completed.all():
                break
        
        return generated
    
    def load_weights(self, path: str) -> None:
        """Load model weights from a file."""
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)
        
    def save_weights(self, path: str) -> None:
        """Save model weights to a file."""
        torch.save(self.state_dict(), path)
        
    def get_expert_stats(self) -> Dict:
        """
        Get statistics about experts in the model, including adaptive expert metrics.
        
        Returns:
            Dictionary containing expert statistics for each layer
        """
        stats = {}
        
        for i, layer in enumerate(self.layers):
            # Skip dense layers
            if i < self.config.n_dense_layers:
                continue
                
            # Check if this layer uses adaptive experts
            if hasattr(layer.ffn, 'get_expert_stats'):
                stats[f'layer_{i}'] = layer.ffn.get_expert_stats()
            elif hasattr(layer.ffn, 'experts'):
                # Basic stats for regular DNPSLayer
                stats[f'layer_{i}'] = {
                    "num_experts": len(layer.ffn.experts),
                    "expert_type": "fixed"
                }
                
        return stats
    
    def visualize_expert_scaling(self, layer_idx: int = None):
        """
        Visualize how experts have been scaled over time.
        Requires matplotlib to be installed.
        
        Args:
            layer_idx: Optional index of specific layer to visualize. If None, visualizes all adaptive layers.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib is required for visualization. Please install with 'pip install matplotlib'")
            return
            
        stats = self.get_expert_stats()
        
        if not stats:
            print("No adaptive expert layers found in the model.")
            return
            
        if layer_idx is not None:
            layer_key = f'layer_{layer_idx}'
            if layer_key not in stats:
                print(f"Layer {layer_idx} does not exist or does not use adaptive experts.")
                return
                
            layers_to_plot = {layer_key: stats[layer_key]}
        else:
            layers_to_plot = stats
            
        for layer_name, layer_stats in layers_to_plot.items():
            if "scaling_history" not in layer_stats:
                print(f"{layer_name} does not use adaptive experts or has no scaling history.")
                continue
                
            scaling_history = layer_stats["scaling_history"]
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot expert sizes over time
            for i, expert_history in enumerate(scaling_history):
                if not expert_history:
                    continue
                    
                steps = [h['step'] for h in expert_history]
                sizes = [h['new_size'] for h in expert_history]
                importance = [h['importance'] for h in expert_history]
                
                ax1.plot(steps, sizes, 'o-', label=f'Expert {i}')
                
                # Add annotations for importance
                for j, (x, y, imp) in enumerate(zip(steps, sizes, importance)):
                    ax1.annotate(f"{imp:.2f}", (x, y), 
                               xytext=(0, 5), textcoords='offset points',
                               ha='center', va='bottom', fontsize=8)
            
            ax1.set_title(f"{layer_name} Expert Size Evolution")
            ax1.set_xlabel("Training Step")
            ax1.set_ylabel("Hidden Dimension")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot current expert sizes and importance
            expert_sizes = layer_stats["expert_sizes"]
            importance = layer_stats["importance_scores"]
            
            x = np.arange(len(expert_sizes))
            ax2.bar(x, expert_sizes, alpha=0.7, label='Hidden Dimension')
            
            # Add importance scores as line
            ax2_twin = ax2.twinx()
            ax2_twin.plot(x, importance, 'ro-', label='Importance')
            
            ax2.set_title(f"{layer_name} Current Expert Sizes and Importance")
            ax2.set_xlabel("Expert Index")
            ax2.set_ylabel("Hidden Dimension")
            ax2_twin.set_ylabel("Importance Score")
            
            # Add legends for both axes
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            ax2.set_xticks(x)
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()