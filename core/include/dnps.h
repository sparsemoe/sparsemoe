#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

namespace sparsemoe {

/**
 * DynamicNeuralPathway implements the Dynamic Neural Pathway Selection (DNPS)
 * algorithm that selectively activates a subset of experts based on input.
 */
class DynamicNeuralPathway {
public:
    /**
     * Constructs a DynamicNeuralPathway module.
     * 
     * @param dim Input dimension
     * @param num_experts Total number of experts
     * @param num_selected Number of experts to activate per token
     * @param group_size Number of experts in each group (for hierarchical routing)
     * @param score_func Scoring function ("sigmoid" or "softmax")
     * @param route_scale Scaling factor for routing scores
     */
    DynamicNeuralPathway(
        int dim, 
        int num_experts, 
        int num_selected, 
        int group_size = 1,
        const std::string& score_func = "sigmoid",
        float route_scale = 1.0
    );

    /**
     * Forward pass of the DNPS algorithm.
     * 
     * @param input Input tensor [batch_size, seq_len, dim]
     * @param expert_weights Expert weights [num_experts, dim, ffn_dim]
     * @param routing_logits Optional pre-computed routing logits
     * @return Output tensor [batch_size, seq_len, dim]
     */
    torch::Tensor forward(
        const torch::Tensor& input,
        const std::vector<torch::Tensor>& expert_weights,
        const torch::Tensor& routing_logits = {}
    );

    /**
     * Computes routing weights and expert selection.
     * 
     * @param input Input tensor [batch_size, seq_len, dim]
     * @return Tuple of (routing_weights, selected_experts)
     */
    std::tuple<torch::Tensor, torch::Tensor> compute_routing_weights(
        const torch::Tensor& input
    );

    /**
     * Sets whether to use CUDA kernels for routing computation.
     * 
     * @param use_cuda Whether to use custom CUDA kernels
     */
    void set_use_cuda_kernels(bool use_cuda);

private:
    // Parameters
    int dim_;
    int num_experts_;
    int num_selected_;
    int group_size_;
    std::string score_func_;
    float route_scale_;
    bool use_cuda_kernels_;

    // Router parameters
    torch::Tensor router_weights_;
    torch::Tensor router_bias_;

    // Internal methods
    torch::Tensor apply_score_function(const torch::Tensor& logits);
    torch::Tensor apply_hierarchical_routing(const torch::Tensor& scores);
};

} // namespace sparsemoe