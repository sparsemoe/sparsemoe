#include "dnps.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

namespace sparsemoe {

// Forward declaration of CUDA kernel launcher
#ifdef USE_CUDA
torch::Tensor compute_routing_cuda(
    const torch::Tensor& input,
    const torch::Tensor& router_weights,
    const torch::Tensor& router_bias,
    int num_experts,
    int num_selected,
    int group_size,
    const std::string& score_func,
    float route_scale
);
#endif

DynamicNeuralPathway::DynamicNeuralPathway(
    int dim, 
    int num_experts, 
    int num_selected, 
    int group_size,
    const std::string& score_func,
    float route_scale
) : dim_(dim),
    num_experts_(num_experts),
    num_selected_(num_selected),
    group_size_(group_size),
    score_func_(score_func),
    route_scale_(route_scale),
    use_cuda_kernels_(true) {
    
    // Initialize router weights and bias
    router_weights_ = torch::empty({num_experts_, dim_}, torch::dtype(torch::kFloat32));
    router_bias_ = torch::empty({num_experts_}, torch::dtype(torch::kFloat32));
    
    // Initialize with small random values
    torch::nn::init::normal_(router_weights_, 0.0, 0.02);
    torch::nn::init::zeros_(router_bias_);
}

torch::Tensor DynamicNeuralPathway::forward(
    const torch::Tensor& input,
    const std::vector<torch::Tensor>& expert_weights,
    const torch::Tensor& routing_logits
) {
    // Get dimensions
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto ffn_dim = expert_weights[0].size(1);
    
    // Compute routing weights if not provided
    torch::Tensor weights;
    torch::Tensor indices;
    
    if (routing_logits.defined()) {
        auto routing_scores = apply_score_function(routing_logits);
        std::tie(weights, indices) = routing_scores.topk(num_selected_, -1);
        
        // Normalize weights
        if (score_func_ == "sigmoid") {
            weights = weights / (weights.sum(-1, true) + 1e-6);
        }
        weights = weights * route_scale_;
    } else {
        std::tie(weights, indices) = compute_routing_weights(input);
    }
    
    // Flatten the input for expert computation
    auto flat_input = input.reshape({-1, dim_});
    auto flat_weights = weights.reshape({-1, num_selected_});
    auto flat_indices = indices.reshape({-1, num_selected_});
    
    // Initialize output
    auto output = torch::zeros({flat_input.size(0), ffn_dim}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    // Apply selected experts
    for (int expert_idx = 0; expert_idx < num_experts_; expert_idx++) {
        // Find tokens that route to this expert
        auto mask = (flat_indices == expert_idx);
        if (!mask.any().item<bool>()) {
            continue;
        }
        
        // Get relevant token indices and weights
        auto token_indices = mask.nonzero().select(1, 0);
        auto expert_input = flat_input.index_select(0, token_indices);
        
        // Extract corresponding weights for this expert
        auto weight_indices = mask.nonzero();
        auto batch_indices = weight_indices.select(1, 0);
        auto selected_indices = weight_indices.select(1, 1);
        auto expert_weights_scalar = flat_weights.index_select(0, batch_indices)
                                               .gather(1, selected_indices.unsqueeze(1))
                                               .squeeze();
        
        // Apply expert transformation
        auto expert_output = torch::matmul(expert_input, expert_weights[expert_idx]);
        
        // Scale by routing weights
        expert_output = expert_output * expert_weights_scalar.unsqueeze(1);
        
        // Add to output
        output.index_add_(0, token_indices, expert_output);
    }
    
    // Reshape output back to original dimensions
    return output.reshape({batch_size, seq_len, ffn_dim});
}

std::tuple<torch::Tensor, torch::Tensor> DynamicNeuralPathway::compute_routing_weights(
    const torch::Tensor& input
) {
    // Use CUDA kernels if available and enabled
#ifdef USE_CUDA
    if (use_cuda_kernels_ && input.is_cuda()) {
        return compute_routing_cuda(
            input, router_weights_, router_bias_,
            num_experts_, num_selected_, group_size_,
            score_func_, route_scale_
        );
    }
#endif

    // Fall back to PyTorch implementation
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    
    // Reshape input for routing computation
    auto flat_input = input.reshape({-1, dim_});
    
    // Compute routing logits
    auto logits = torch::matmul(flat_input, router_weights_.transpose(0, 1));
    logits = logits + router_bias_;
    
    // Apply score function
    auto scores = apply_score_function(logits);
    
    // Apply hierarchical routing if needed
    if (group_size_ > 1) {
        scores = apply_hierarchical_routing(scores);
    }
    
    // Select top-k experts
    auto topk_result = scores.topk(num_selected_, -1);
    auto weights = std::get<0>(topk_result);
    auto indices = std::get<1>(topk_result);
    
    // Normalize weights if using sigmoid scoring
    if (score_func_ == "sigmoid") {
        weights = weights / (weights.sum(-1, true) + 1e-6);
    }
    
    // Apply routing scale
    weights = weights * route_scale_;
    
    // Reshape back to original dimensions
    weights = weights.reshape({batch_size, seq_len, num_selected_});
    indices = indices.reshape({batch_size, seq_len, num_selected_});
    
    return std::make_tuple(weights, indices);
}

torch::Tensor DynamicNeuralPathway::apply_score_function(const torch::Tensor& logits) {
    if (score_func_ == "softmax") {
        return torch::softmax(logits, -1);
    } else if (score_func_ == "sigmoid") {
        return torch::sigmoid(logits);
    } else {
        throw std::runtime_error("Unsupported score function: " + score_func_);
    }
}

torch::Tensor DynamicNeuralPathway::apply_hierarchical_routing(const torch::Tensor& scores) {
    auto batch_size = scores.size(0);
    auto num_groups = num_experts_ / group_size_;
    
    // Reshape scores to separate groups
    auto grouped_scores = scores.reshape({batch_size, num_groups, group_size_});
    
    // Compute group scores
    auto group_scores = grouped_scores.sum(-1);
    
    // Select top groups
    auto top_groups = group_scores.topk(std::min(num_groups, num_selected_), -1);
    auto top_group_indices = std::get<1>(top_groups);
    
    // Create mask for selected groups
    auto group_mask = torch::zeros_like(group_scores);
    group_mask.scatter_(-1, top_group_indices, 1.0);
    
    // Apply mask to original scores
    auto masked_scores = scores.reshape({batch_size, num_groups, group_size_})
                               .clone();
    
    for (int i = 0; i < num_groups; i++) {
        masked_scores.select(1, i).mul_(group_mask.select(1, i).unsqueeze(1));
    }
    
    return masked_scores.reshape({batch_size, num_experts_});
}

void DynamicNeuralPathway::set_use_cuda_kernels(bool use_cuda) {
    use_cuda_kernels_ = use_cuda;
}

} // namespace sparsemoe