#include <torch/torch.h>
#include <gtest/gtest.h>
#include "attention.h"

// Test fixture for attention tests
class AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize common parameters
        dim = 64;
        num_heads = 4;
        head_dim = 16;
        max_seq_len = 128;
        max_batch_size = 2;
        softmax_scale = 1.0 / std::sqrt(head_dim);
        
        // Create a DynamicAttention instance
        attention = std::make_unique<sparsemoe::DynamicAttention>(
            dim, num_heads, head_dim, max_seq_len, max_batch_size, softmax_scale
        );
        
        // Create sample input tensors
        batch_size = 2;
        seq_len = 8;
        
        query = torch::randn({batch_size, seq_len, num_heads, head_dim});
        key = torch::randn({batch_size, seq_len, num_heads, head_dim});
        value = torch::randn({batch_size, seq_len, num_heads, head_dim});
    }

    // Test parameters
    int dim;
    int num_heads;
    int head_dim;
    int max_seq_len;
    int max_batch_size;
    float softmax_scale;
    int batch_size;
    int seq_len;
    
    // Test objects
    std::unique_ptr<sparsemoe::DynamicAttention> attention;
    torch::Tensor query;
    torch::Tensor key;
    torch::Tensor value;
};

// Test initialization
TEST_F(AttentionTest, Initialization) {
    // Create with different parameters
    auto attention_custom = sparsemoe::DynamicAttention(
        128, 8, 32, 256, 4, 0.1
    );
    
    // Nothing to assert directly as we can't access private members
    // Just test that creation doesn't throw an exception
}

// Test KV cache update
TEST_F(AttentionTest, KVCacheUpdate) {
    int start_pos = 0;
    
    // Update KV cache
    attention->update_kv_cache(key, value, start_pos);
    
    // Cannot directly verify cache contents with current interface
    // Will verify indirectly through forward pass
}

// Test forward pass
TEST_F(AttentionTest, ForwardDecoder) {
    int start_pos = 0;
    
    // Update KV cache
    attention->update_kv_cache(key, value, start_pos);
    
    // Create attention mask (causal mask)
    torch::Tensor mask = torch::zeros({seq_len, seq_len});
    mask = torch::triu(mask, /*diagonal=*/1);
    mask = mask * -1e9;
    
    // Run forward pass
    torch::Tensor output = attention->forward_decoder(query, start_pos, mask);
    
    // Check output shape
    ASSERT_EQ(output.sizes()[0], batch_size);
    ASSERT_EQ(output.sizes()[1], seq_len);
    ASSERT_EQ(output.sizes()[2], num_heads * head_dim);
}

// Test incremental decoding
TEST_F(AttentionTest, IncrementalDecoding) {
    // First update KV cache with 4 tokens
    int start_pos_1 = 0;
    int seq_len_1 = 4;
    torch::Tensor key_1 = key.slice(1, 0, seq_len_1);
    torch::Tensor value_1 = value.slice(1, 0, seq_len_1);
    torch::Tensor query_1 = query.slice(1, 0, seq_len_1);
    
    attention->update_kv_cache(key_1, value_1, start_pos_1);
    
    // No attention mask needed for single-token processing
    torch::Tensor mask_1 = torch::zeros({seq_len_1, seq_len_1});
    mask_1 = torch::triu(mask_1, /*diagonal=*/1);
    mask_1 = mask_1 * -1e9;
    
    // Process first 4 tokens
    torch::Tensor output_1 = attention->forward_decoder(query_1, start_pos_1, mask_1);
    
    // Then process next 4 tokens
    int start_pos_2 = 4;
    int seq_len_2 = 4;
    torch::Tensor key_2 = key.slice(1, seq_len_1, seq_len_1 + seq_len_2);
    torch::Tensor value_2 = value.slice(1, seq_len_1, seq_len_1 + seq_len_2);
    torch::Tensor query_2 = query.slice(1, seq_len_1, seq_len_1 + seq_len_2);
    
    attention->update_kv_cache(key_2, value_2, start_pos_2);
    
    // Process next 4 tokens
    torch::Tensor output_2 = attention->forward_decoder(query_2, start_pos_2, torch::Tensor());
    
    // Check output shapes
    ASSERT_EQ(output_1.sizes()[0], batch_size);
    ASSERT_EQ(output_1.sizes()[1], seq_len_1);
    ASSERT_EQ(output_1.sizes()[2], num_heads * head_dim);
    
    ASSERT_EQ(output_2.sizes()[0], batch_size);
    ASSERT_EQ(output_2.sizes()[1], seq_len_2);
    ASSERT_EQ(output_2.sizes()[2], num_heads * head_dim);
}

// Test CUDA support if available
TEST_F(AttentionTest, CudaSupport) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    // Create a CUDA-based attention layer
    auto cuda_attention = std::make_unique<sparsemoe::DynamicAttention>(
        dim, num_heads, head_dim, max_seq_len, max_batch_size, softmax_scale
    );
    
    // Move tensors to CUDA
    torch::Tensor query_cuda = query.to(torch::kCUDA);
    torch::Tensor key_cuda = key.to(torch::kCUDA);
    torch::Tensor value_cuda = value.to(torch::kCUDA);
    
    // Create attention mask on CUDA
    torch::Tensor mask_cuda = torch::zeros({seq_len, seq_len}, 
                                         torch::TensorOptions().device(torch::kCUDA));
    mask_cuda = torch::triu(mask_cuda, /*diagonal=*/1);
    mask_cuda = mask_cuda * -1e9;
    
    // Update KV cache
    cuda_attention->update_kv_cache(key_cuda, value_cuda, 0);
    
    // Run forward pass
    torch::Tensor output_cuda = cuda_attention->forward_decoder(query_cuda, 0, mask_cuda);
    
    // Check output is on CUDA and has correct shape
    ASSERT_TRUE(output_cuda.device().is_cuda());
    ASSERT_EQ(output_cuda.sizes()[0], batch_size);
    ASSERT_EQ(output_cuda.sizes()[1], seq_len);
    ASSERT_EQ(output_cuda.sizes()[2], num_heads * head_dim);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}