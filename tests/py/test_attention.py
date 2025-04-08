import unittest
import os
import sys
import torch

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sparsemoe.model import SparseMoEConfig, DynamicAttention

class TestDynamicAttention(unittest.TestCase):
    def setUp(self):
        # Create a small config for testing
        self.config = SparseMoEConfig(
            dim=64,
            n_heads=4,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=16,
            max_seq_len=128,
            max_batch_size=2,
            use_cpp_backend=False  # Use Python implementation for testing
        )
        
        # Create attention layer
        self.attn = DynamicAttention(self.config)
        
        # Precompute rotary embeddings (normally done in the model)
        dim = self.config.qk_rope_head_dim
        max_seq_len = self.config.max_seq_len
        
        # Compute frequency
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        
        # Compute phases
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        
        # Complex exponential
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        # Create sample input
        self.batch_size = 2
        self.seq_len = 8
        self.x = torch.randn(self.batch_size, self.seq_len, self.config.dim)
        
    def test_initialization(self):
        """Test attention layer initialization."""
        # Check dimensions
        self.assertEqual(self.attn.dim, self.config.dim)
        self.assertEqual(self.attn.n_heads, self.config.n_heads)
        
        # Check projections
        if self.config.q_lora_rank == 0:
            self.assertIsNotNone(self.attn.wq)
        else:
            self.assertIsNotNone(self.attn.wq_a)
            self.assertIsNotNone(self.attn.wq_b)
            self.assertIsNotNone(self.attn.q_norm)
        
        self.assertIsNotNone(self.attn.wkv_a)
        self.assertIsNotNone(self.attn.wkv_b)
        self.assertIsNotNone(self.attn.kv_norm)
        self.assertIsNotNone(self.attn.wo)
        
    def test_apply_rotary_emb(self):
        """Test application of rotary embeddings."""
        # Create tensor to apply rotary embeddings to
        batch_size = 2
        seq_len = 8
        n_heads = 4
        head_dim = self.config.qk_rope_head_dim
        x = torch.randn(batch_size, seq_len, n_heads, head_dim)
        
        # Get relevant slice of rotary embeddings
        freqs_cis_slice = self.freqs_cis[:seq_len]
        
        # Apply rotary embeddings
        rotated = self.attn._apply_rotary_emb(x, freqs_cis_slice)
        
        # Check output shape
        self.assertEqual(rotated.shape, x.shape)
        
    def test_forward(self):
        """Test forward pass of attention mechanism."""
        start_pos = 0
        freqs_cis_slice = self.freqs_cis[:self.seq_len]
        
        # Create causal mask
        mask = torch.full((self.seq_len, self.seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        
        # Forward pass
        output = self.attn(self.x, start_pos, freqs_cis_slice, mask)
        
        # Check output shape
        self.assertEqual(output.shape, self.x.shape)
        
    def test_kv_cache(self):
        """Test KV cache functionality."""
        # Only run this test for the Python implementation
        if self.attn.cpp_impl is not None:
            self.skipTest("KV cache test only applicable for Python implementation")
        
        # First pass - fill cache for positions 0-3
        start_pos_1 = 0
        seq_len_1 = 4
        x_1 = self.x[:, :seq_len_1]
        freqs_cis_1 = self.freqs_cis[:seq_len_1]
        
        # Create causal mask
        mask_1 = torch.full((seq_len_1, seq_len_1), float("-inf"))
        mask_1 = torch.triu(mask_1, diagonal=1)
        
        # First forward pass
        output_1 = self.attn(x_1, start_pos_1, freqs_cis_1, mask_1)
        
        # Check that KV cache was updated
        self.assertTrue(torch.any(self.attn.k_cache[:self.batch_size, :seq_len_1] != 0))
        self.assertTrue(torch.any(self.attn.v_cache[:self.batch_size, :seq_len_1] != 0))
        
        # Second pass - for positions 4-7
        start_pos_2 = 4
        seq_len_2 = 4
        x_2 = self.x[:, seq_len_1:seq_len_1+seq_len_2]
        freqs_cis_2 = self.freqs_cis[seq_len_1:seq_len_1+seq_len_2]
        
        # Second forward pass
        output_2 = self.attn(x_2, start_pos_2, freqs_cis_2, None)
        
        # Check that KV cache was updated for new positions
        self.assertTrue(torch.any(self.attn.k_cache[:self.batch_size, start_pos_2:start_pos_2+seq_len_2] != 0))
        self.assertTrue(torch.any(self.attn.v_cache[:self.batch_size, start_pos_2:start_pos_2+seq_len_2] != 0))
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_support(self):
        """Test that the attention layer works on CUDA."""
        if torch.cuda.is_available():
            attn_cuda = DynamicAttention(self.config).cuda()
            x_cuda = self.x.cuda()
            freqs_cis_cuda = self.freqs_cis.cuda()
            
            # Forward pass on CUDA
            output_cuda = attn_cuda(x_cuda, 0, freqs_cis_cuda[:self.seq_len])
            
            # Check output is on CUDA
            self.assertTrue(output_cuda.is_cuda)
            self.assertEqual(output_cuda.shape, x_cuda.shape)

if __name__ == '__main__':
    unittest.main()