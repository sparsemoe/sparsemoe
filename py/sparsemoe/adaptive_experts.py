"""
Adaptive Expert Scaling mechanism for SparseMoE.

This module implements a novel approach to expert scaling where expert capacity
(parameter count) is dynamically adjusted based on usage patterns. Frequently used
experts gain more capacity while underutilized ones are pruned or merged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import math


class ExpertUsageTracker:
    """
    Tracks expert usage statistics to inform expert scaling decisions.
    """
    
    def __init__(self, num_experts: int, window_size: int = 1000, alpha: float = 0.9):
        """
        Initialize expert usage tracker.
        
        Args:
            num_experts: Number of experts in the system
            window_size: Number of forward passes to consider for scaling decisions
            alpha: Exponential moving average decay factor
        """
        self.num_experts = num_experts
        self.window_size = window_size
        self.alpha = alpha
        
        # Activation counts for each expert
        self.activation_counts = torch.zeros(num_experts)
        
        # Exponential moving average of importance scores
        self.importance_ema = torch.ones(num_experts) / num_experts
        
        # Total forward passes tracked
        self.total_forwards = 0
        
        # History of scaling operations for each expert
        self.scaling_history = [[] for _ in range(num_experts)]
    
    def update(self, expert_indices: torch.Tensor, importance_scores: torch.Tensor):
        """
        Update usage statistics based on a forward pass.
        
        Args:
            expert_indices: Tensor of indices of experts that were activated
            importance_scores: Tensor of routing weights/importance for each activated expert
        """
        # Update activation counts
        expert_counts = torch.bincount(expert_indices.flatten(), 
                                     minlength=self.num_experts)
        self.activation_counts += expert_counts
        
        # Update importance EMA
        current_importance = torch.zeros(self.num_experts, 
                                       device=importance_scores.device)
        
        # For each unique expert index, sum its importance scores
        for idx, score in zip(expert_indices.flatten(), importance_scores.flatten()):
            current_importance[idx] += score.item()
            
        # Normalize importance
        if current_importance.sum() > 0:
            current_importance = current_importance / current_importance.sum()
            
        # Update EMA
        self.importance_ema = self.alpha * self.importance_ema + (1 - self.alpha) * current_importance.cpu()
        
        # Increment forward count
        self.total_forwards += 1
    
    def should_scale(self) -> bool:
        """
        Determine if it's time to rescale experts.
        
        Returns:
            True if experts should be rescaled, False otherwise
        """
        return self.total_forwards % self.window_size == 0 and self.total_forwards > 0
    
    def get_scaling_factors(self) -> torch.Tensor:
        """
        Calculate scaling factors for each expert based on usage.
        
        Returns:
            Tensor of scaling factors for each expert
        """
        # Calculate scaling factors based on importance EMA
        # Experts with higher importance get higher scaling factors
        normalized_importance = self.importance_ema / self.importance_ema.mean()
        
        # Apply sigmoid-like function to smooth extremes
        scaling_factors = torch.tanh(normalized_importance - 1.0) * 0.5 + 1.0
        
        # Cap scaling factors to reasonable range
        scaling_factors = torch.clamp(scaling_factors, 0.5, 2.0)
        
        return scaling_factors
    
    def reset_window(self):
        """Reset usage statistics for a new window."""
        self.activation_counts.zero_()
    
    def log_scaling(self, expert_idx: int, old_size: int, new_size: int):
        """
        Log a scaling operation for an expert.
        
        Args:
            expert_idx: Index of the expert that was scaled
            old_size: Previous parameter count/dimension
            new_size: New parameter count/dimension
        """
        self.scaling_history[expert_idx].append({
            'step': self.total_forwards,
            'old_size': old_size,
            'new_size': new_size,
            'importance': self.importance_ema[expert_idx].item()
        })


class AdaptiveExpert(nn.Module):
    """
    An expert module that can dynamically adjust its capacity.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 initial_dim: int,
                 output_dim: int,
                 min_dim: int,
                 max_dim: int,
                 growth_factor: float = 0.25,
                 shrink_factor: float = 0.25):
        """
        Initialize an adaptive expert.
        
        Args:
            input_dim: Input dimension
            initial_dim: Initial hidden dimension
            output_dim: Output dimension
            min_dim: Minimum allowed hidden dimension
            max_dim: Maximum allowed hidden dimension
            growth_factor: Maximum fraction to grow by in one step
            shrink_factor: Maximum fraction to shrink by in one step
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = initial_dim
        self.output_dim = output_dim
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        
        # Expert layers
        self.up_proj = nn.Linear(input_dim, initial_dim, bias=False)
        self.activation = nn.SiLU()
        self.down_proj = nn.Linear(initial_dim, output_dim, bias=False)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights with appropriate scaling."""
        # Initialize with scaled normal distribution
        with torch.no_grad():
            std = 1 / math.sqrt(self.input_dim)
            nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
            
            std = 1 / math.sqrt(self.hidden_dim)
            nn.init.normal_(self.down_proj.weight, mean=0.0, std=std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert."""
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x
    
    def resize(self, new_dim: int) -> bool:
        """
        Resize the expert's hidden dimension.
        
        Args:
            new_dim: New hidden dimension
            
        Returns:
            True if resizing was successful, False otherwise
        """
        # Ensure new dimension is within allowed range
        new_dim = max(self.min_dim, min(self.max_dim, new_dim))
        
        # No need to resize if dimensions match
        if new_dim == self.hidden_dim:
            return False
        
        # Store old weights
        old_up_weight = self.up_proj.weight.data
        old_down_weight = self.down_proj.weight.data
        old_dim = self.hidden_dim
        
        # Create new layers
        new_up_proj = nn.Linear(self.input_dim, new_dim, bias=False)
        new_down_proj = nn.Linear(new_dim, self.output_dim, bias=False)
        
        # Initialize new weights
        with torch.no_grad():
            std = 1 / math.sqrt(self.input_dim)
            nn.init.normal_(new_up_proj.weight, mean=0.0, std=std)
            
            std = 1 / math.sqrt(new_dim)
            nn.init.normal_(new_down_proj.weight, mean=0.0, std=std)
        
        # If we're growing, copy old weights and initialize new ones
        if new_dim > old_dim:
            # Copy existing weights
            new_up_proj.weight.data[:, :self.input_dim][:old_dim] = old_up_weight
            new_down_proj.weight.data[:, :old_dim] = old_down_weight
        else:
            # If we're shrinking, keep the most important connections
            # For simplicity, we'll just keep the first `new_dim` neurons
            # A more sophisticated approach would use importance metrics
            new_up_proj.weight.data = old_up_weight[:new_dim]
            new_down_proj.weight.data = old_down_weight[:, :new_dim]
        
        # Replace layers
        self.up_proj = new_up_proj
        self.down_proj = new_down_proj
        self.hidden_dim = new_dim
        
        return True


class AdaptiveExpertLayer(nn.Module):
    """
    A layer of adaptive experts with dynamic scaling capabilities.
    """
    
    def __init__(
        self,
        dim: int,
        num_experts: int,
        num_selected: int,
        initial_expert_dim: int,
        min_expert_dim: int,
        max_expert_dim: int,
        scaling_interval: int = 1000,
        router_capacity_factor: float = 1.5,
    ):
        """
        Initialize an adaptive expert layer.
        
        Args:
            dim: Input/output dimension
            num_experts: Number of experts in the layer
            num_selected: Number of experts to select per token
            initial_expert_dim: Initial hidden dimension for each expert
            min_expert_dim: Minimum allowed hidden dimension
            max_expert_dim: Maximum allowed hidden dimension
            scaling_interval: How often to check for expert rescaling
            router_capacity_factor: Scaling factor for router capacity (>1 means tokens can be routed to experts beyond their capacity)
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.router_capacity_factor = router_capacity_factor
        
        # Router for selecting experts
        self.router = nn.Linear(dim, num_experts, bias=True)
        
        # Initialize adaptive experts
        self.experts = nn.ModuleList([
            AdaptiveExpert(
                input_dim=dim,
                initial_dim=initial_expert_dim,
                output_dim=dim,
                min_dim=min_expert_dim,
                max_dim=max_expert_dim
            )
            for _ in range(num_experts)
        ])
        
        # Expert usage tracker
        self.tracker = ExpertUsageTracker(num_experts, window_size=scaling_interval)
        
        # Shared expert (applied to all tokens)
        self.shared_expert = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False)
        )
        
        # Total parameter count (for tracking efficiency)
        self.total_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic expert selection and scaling.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute routing logits
        routing_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # Apply sigmoid activation
        routing_weights = torch.sigmoid(routing_logits)
        
        # Get top-k experts
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.num_selected, dim=-1
        )  # [batch_size, seq_len, num_selected]
        
        # Normalize weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Update expert usage tracker
        self.tracker.update(top_k_indices, top_k_weights)
        
        # Initialize output
        moe_output = torch.zeros_like(x)
        
        # Apply selected experts to each token
        # For simplicity, we'll use a loop-based implementation here
        # A more efficient implementation would use scatter operations
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                token_input = x[batch_idx, seq_idx].unsqueeze(0)  # [1, dim]
                
                for k in range(self.num_selected):
                    expert_idx = top_k_indices[batch_idx, seq_idx, k].item()
                    weight = top_k_weights[batch_idx, seq_idx, k].item()
                    
                    # Apply expert and weight the output
                    expert_output = self.experts[expert_idx](token_input)
                    moe_output[batch_idx, seq_idx] += expert_output.squeeze(0) * weight
        
        # Add shared expert output
        shared_output = self.shared_expert(x)
        
        # Check if we should scale experts based on usage patterns
        if self.tracker.should_scale():
            self._scale_experts()
            self.tracker.reset_window()
        
        return moe_output + shared_output
    
    def _scale_experts(self):
        """
        Scale experts based on usage patterns.
        This is the key innovation of the adaptive expert approach.
        """
        # Get scaling factors for each expert
        scaling_factors = self.tracker.get_scaling_factors()
        
        # Log current sizes for all experts
        current_sizes = [expert.hidden_dim for expert in self.experts]
        
        # Calculate new sizes ensuring total parameter count roughly stays the same
        current_total = sum(current_sizes)
        new_sizes = [max(int(size * factor.item()), self.experts[0].min_dim) 
                   for size, factor in zip(current_sizes, scaling_factors)]
        
        # Normalize to keep total parameters approximately the same
        new_total = sum(new_sizes)
        if new_total > 0:  # Safeguard against division by zero
            scale = current_total / new_total
            new_sizes = [max(int(size * scale), self.experts[0].min_dim) for size in new_sizes]
        
        # Apply sizing changes
        for i, new_size in enumerate(new_sizes):
            # Attempt to resize expert
            resized = self.experts[i].resize(new_size)
            
            if resized:
                # Log the scaling operation
                self.tracker.log_scaling(i, current_sizes[i], new_size)
                
        # Update total parameter count
        self.total_params = sum(p.numel() for p in self.parameters())
        
    def get_expert_sizes(self) -> List[int]:
        """
        Get current sizes of all experts.
        
        Returns:
            List of hidden dimensions for each expert
        """
        return [expert.hidden_dim for expert in self.experts]
    
    def get_expert_stats(self) -> Dict:
        """
        Get detailed statistics about expert usage and scaling.
        
        Returns:
            Dictionary containing expert statistics
        """
        return {
            "expert_sizes": self.get_expert_sizes(),
            "importance_scores": self.tracker.importance_ema.tolist(),
            "activation_counts": self.tracker.activation_counts.tolist(),
            "scaling_history": self.tracker.scaling_history,
            "total_params": self.total_params
        }