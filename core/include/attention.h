#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

namespace sparsemoe {

/**
 * DynamicAttention implements multi-head attention with optimization
 * for decoder-only transformer models.
 */
class DynamicAttention {
public:
    /**
     * Constructs a DynamicAttention module.
     * 
     * @param dim Model dimension
     * @param num_heads Number of attention heads
     * @param head_dim Dimension of each attention head
     * @param max_seq_len Maximum sequence length
     * @param max_batch_size Maximum batch size
     * @param scale Attention scale factor (default: 1/sqrt(head_dim))
     */
    DynamicAttention(
        int dim,
        int num_heads,
        int head_dim,
        int max_seq_len,
        int max_batch_size,
        float scale = 0.0f
    );

    /**
     * Forward pass for encoder-decoder attention.
     * 
     * @param query Query tensor [batch_size, seq_len, dim]
     * @param key Key tensor [batch_size, seq_len, dim]
     * @param value Value tensor [batch_size, seq_len, dim]
     * @param mask Optional attention mask
     * @param start_pos Starting position for caching
     * @return Output tensor [batch_size, seq_len, dim]
     */
    torch::Tensor forward(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        const torch::Tensor& mask = {},
        int start_pos = 0
    );

    /**
     * Forward pass for decoder self-attention with cached keys and values.
     * 
     * @param query Query tensor [batch_size, seq_len, dim]
     * @param start_pos Starting position for caching
     * @param freqs_cis Complex exponentials for rotary position embeddings
     * @return Output tensor [batch_size, seq_len, dim]
     */
    torch::Tensor forward_decoder(
        const torch::Tensor& query,
        int start_pos = 0,
        const torch::Tensor& freqs_cis = {}
    );

    /**
     * Updates the key-value cache.
     * 
     * @param key Key tensor [batch_size, seq_len, num_heads, head_dim]
     * @param value Value tensor [batch_size, seq_len, num_heads, head_dim]
     * @param start_pos Starting position for caching
     */
    void update_kv_cache(
        const torch::Tensor& key,
        const torch::Tensor& value,
        int start_pos = 0
    );

private:
    // Parameters
    int dim_;
    int num_heads_;
    int head_dim_;
    int max_seq_len_;
    int max_batch_size_;
    float scale_;

    // KV cache tensors
    torch::Tensor k_cache_;
    torch::Tensor v_cache_;

    // Apply rotary position embeddings
    torch::Tensor apply_rotary_emb(
        const torch::Tensor& x,
        const torch::Tensor& freqs_cis
    );
};

} // namespace sparsemoe