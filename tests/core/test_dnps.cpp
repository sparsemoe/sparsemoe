#include <torch/torch.h>
#include <gtest/gtest.h>
#include "dnps.h"

// Test fixture for DNPS tests
class DNPSTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize common parameters
        dim = 16;
        num_experts = 8;
        num_selected = 2;
        batch_size = 2;
        seq_len = 4;
        
        // Create a DNPS instance
        dnps = std::make_unique<sparsemoe::DynamicNeuralPathway>(
            dim, num_experts, num_selected, 1, "sigmoid", 1.0
        );
        
        // Create sample input tensor
        input = torch::randn({batch_size, seq_len, dim});
        
        // Create sample expert weights
        expert_weights.resize(num_experts);
        for (int i = 0; i < num_experts; ++i) {
            expert_weights[i] = torch::randn({dim, dim * 4});
        }
    }

    // Test parameters
    int dim;
    int num_experts;
    int num_selected;
    int batch_size;
    int seq_len;
    
    // Test objects
    std::unique_ptr<sparsemoe::DynamicNeuralPathway> dnps;
    torch::Tensor input;
    std::vector<torch::Tensor> expert_weights;
};

// Test DNPS initialization
TEST_F(DNPSTest, Initialization) {
    // Create DNPS with different parameters
    auto dnps_custom = sparsemoe::DynamicNeuralPathway(
        32, 16, 4, 2, "softmax", 2.0
    );
    
    // Nothing to assert directly as we can't access private members
    // Just test that creation doesn't throw an exception
}

// Test forward pass
TEST_F(DNPSTest, ForwardShape) {
    // Compute routing logits
    torch::Tensor routing_logits = torch::randn({batch_size, seq_len, num_experts});
    
    // Run forward pass
    torch::Tensor output = dnps->forward(input, expert_weights, routing_logits);
    
    // Check output shape
    ASSERT_EQ(output.sizes(), input.sizes());
}

// Test with hierarchical routing
TEST_F(DNPSTest, HierarchicalRouting) {
    // Create DNPS with hierarchical routing
    auto dnps_hierarchical = sparsemoe::DynamicNeuralPathway(
        dim, num_experts, num_selected, 2, "sigmoid", 1.0
    );
    
    // Compute routing logits
    torch::Tensor routing_logits = torch::randn({batch_size, seq_len, num_experts});
    
    // Run forward pass
    torch::Tensor output = dnps_hierarchical.forward(input, expert_weights, routing_logits);
    
    // Check output shape
    ASSERT_EQ(output.sizes(), input.sizes());
}

// Test with different scoring functions
TEST_F(DNPSTest, ScoringFunctions) {
    // Create DNPS with softmax scoring
    auto dnps_softmax = sparsemoe::DynamicNeuralPathway(
        dim, num_experts, num_selected, 1, "softmax", 1.0
    );
    
    // Compute routing logits
    torch::Tensor routing_logits = torch::randn({batch_size, seq_len, num_experts});
    
    // Run forward pass with sigmoid scoring
    torch::Tensor output_sigmoid = dnps->forward(input, expert_weights, routing_logits);
    
    // Run forward pass with softmax scoring
    torch::Tensor output_softmax = dnps_softmax.forward(input, expert_weights, routing_logits);
    
    // Both should produce outputs of the correct shape
    ASSERT_EQ(output_sigmoid.sizes(), input.sizes());
    ASSERT_EQ(output_softmax.sizes(), input.sizes());
}

// Test CUDA support if available
TEST_F(DNPSTest, CudaSupport) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    // Move input and expert weights to CUDA
    torch::Tensor input_cuda = input.to(torch::kCUDA);
    std::vector<torch::Tensor> expert_weights_cuda;
    expert_weights_cuda.resize(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        expert_weights_cuda[i] = expert_weights[i].to(torch::kCUDA);
    }
    
    // Compute routing logits on CUDA
    torch::Tensor routing_logits_cuda = torch::randn(
        {batch_size, seq_len, num_experts}, 
        torch::TensorOptions().device(torch::kCUDA)
    );
    
    // Enable CUDA kernels
    dnps->set_use_cuda_kernels(true);
    
    // Run forward pass on CUDA
    torch::Tensor output_cuda = dnps->forward(input_cuda, expert_weights_cuda, routing_logits_cuda);
    
    // Check output is on CUDA and has correct shape
    ASSERT_TRUE(output_cuda.device().is_cuda());
    ASSERT_EQ(output_cuda.sizes(), input_cuda.sizes());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}