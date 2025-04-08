import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# Import our mock C++ bindings
from sparsemoe.mock_sparsemoe_bindings import DynamicNeuralPathway as CPPDynamicNeuralPathway
HAS_CPP_BACKEND = True  # Using mock C++ backend

class DNPSLayer(nn.Module):
    """
    Dynamic Neural Pathway Selection layer that implements a sparse mixture of experts.
    
    This layer routes each input token to a selected subset of experts and computes
    a weighted combination of their outputs.
    """
    
    def __init__(
        self,
        dim: int,
        num_experts: int,
        num_selected: int,
        group_size: int = 1,
        score_func: str = "sigmoid",
        route_scale: float = 1.0,
        moe_dim: Optional[int] = None,
        num_shared_experts: int = 1,
        use_cpp_backend: bool = True
    ):
        """
        Initialize the DNPSLayer.
        
        Args:
            dim: Input dimension
            num_experts: Total number of experts
            num_selected: Number of experts to activate per token
            group_size: Number of experts in each group (for hierarchical routing)
            score_func: Scoring function ("sigmoid" or "softmax")
            route_scale: Scaling factor for routing scores
            moe_dim: Expert inner dimension (defaults to dim * 4)
            num_shared_experts: Number of experts applied to all tokens
            use_cpp_backend: Whether to use C++ backend implementation when available
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.group_size = group_size
        self.score_func = score_func
        self.route_scale = route_scale
        self.moe_dim = moe_dim if moe_dim is not None else dim * 4
        self.num_shared_experts = num_shared_experts
        self.use_cpp_backend = use_cpp_backend and HAS_CPP_BACKEND
        
        # Initialize router
        self.router = nn.Linear(dim, num_experts, bias=True)
        
        # Initialize experts (each expert is a MLP with SiLU activation)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.moe_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.moe_dim, dim, bias=False)
            )
            for _ in range(num_experts)
        ])
        
        # Initialize shared expert (applied to all tokens)
        self.shared_expert = nn.Sequential(
            nn.Linear(dim, self.moe_dim * num_shared_experts, bias=False),
            nn.SiLU(),
            nn.Linear(self.moe_dim * num_shared_experts, dim, bias=False)
        )
        
        # Initialize C++ backend if available and requested
        if self.use_cpp_backend:
            self.cpp_impl = CPPDynamicNeuralPathway(
                dim, num_experts, num_selected, group_size, score_func, route_scale
            )
        else:
            self.cpp_impl = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DNPS layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # Try C++ implementation if available
        if self.cpp_impl is not None:
            try:
                # Extract expert weights for C++ implementation
                expert_weights = [
                    expert[0].weight.t()  # First Linear layer weights
                    for expert in self.experts
                ]
                
                # Compute routing logits
                routing_logits = self.router(x)
                
                # Call C++ implementation
                moe_output = self.cpp_impl.forward(x, expert_weights, routing_logits)
                
                # Add shared expert output
                shared_output = self.shared_expert(x)
                
                return moe_output + shared_output
                
            except Exception as e:
                print(f"C++ implementation failed with error: {e}. Falling back to Python implementation.")
        
        # Pure Python implementation
        batch_size, seq_len, _ = x.shape
        
        # Compute routing logits
        routing_logits = self.router(x)
        
        # Apply scoring function
        if self.score_func == "softmax":
            routing_weights = F.softmax(routing_logits, dim=-1)
        else:  # sigmoid
            routing_weights = torch.sigmoid(routing_logits)
        
        # Apply hierarchical routing if needed
        if self.group_size > 1:
            routing_weights = self._apply_hierarchical_routing(routing_weights)
        
        # Get top-k experts
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.num_selected, dim=-1
        )
        
        # Normalize weights for sigmoid scoring
        if self.score_func == "sigmoid":
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Apply routing scale
        top_k_weights = top_k_weights * self.route_scale
        
        # Initialize output
        moe_output = torch.zeros_like(x)
        
        # Apply selected experts to each token
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                token_input = x[batch_idx, seq_idx].unsqueeze(0)  # [1, dim]
                
                for k in range(self.num_selected):
                    expert_idx = top_k_indices[batch_idx, seq_idx, k].item()
                    weight = top_k_weights[batch_idx, seq_idx, k].item()
                    
                    # Apply expert and weight the output
                    expert_output = self.experts[expert_idx](token_input)
                    # Squeeze out the batch dimension as we're accessing a specific position in the output
                    moe_output[batch_idx, seq_idx] += expert_output.squeeze(0) * weight
        
        # Add shared expert output
        shared_output = self.shared_expert(x)
        
        return moe_output + shared_output
    
    def _apply_hierarchical_routing(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply hierarchical routing for efficient expert selection.
        
        Args:
            scores: Expert scores of shape [batch_size, seq_len, num_experts]
            
        Returns:
            Modified scores with hierarchical routing applied
        """
        batch_size, seq_len, _ = scores.shape
        num_groups = self.num_experts // self.group_size
        
        # Reshape scores to separate expert groups
        grouped_scores = scores.view(batch_size, seq_len, num_groups, self.group_size)
        
        # Compute group scores (sum of expert scores in each group)
        group_scores = grouped_scores.sum(dim=-1)  # [batch_size, seq_len, num_groups]
        
        # Select top-k groups
        _, top_groups = torch.topk(
            group_scores, min(num_groups, self.num_selected), dim=-1
        )  # [batch_size, seq_len, min(num_groups, num_selected)]
        
        # Create mask for selected groups
        group_mask = torch.zeros_like(group_scores)
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(-1, seq_len, top_groups.size(-1))
        seq_indices = torch.arange(seq_len).view(1, -1, 1).expand(batch_size, -1, top_groups.size(-1))
        group_mask[batch_indices, seq_indices, top_groups] = 1.0
        
        # Apply mask to original scores
        masked_scores = grouped_scores * group_mask.unsqueeze(-1)
        
        # Reshape back to original shape
        return masked_scores.view(batch_size, seq_len, -1)