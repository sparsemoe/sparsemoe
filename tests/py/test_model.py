import unittest
import os
import sys
import torch

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from py.sparsemoe.model import SparseMoEConfig, SparseMoEModel, RMSNorm

class TestRMSNorm(unittest.TestCase):
    def test_rms_norm(self):
        """Test RMS normalization layer."""
        # Create a small RMSNorm layer
        dim = 16
        rms_norm = RMSNorm(dim)
        
        # Test with batch size 2, sequence length 4
        x = torch.randn(2, 4, dim)
        y = rms_norm(x)
        
        # Output should have the same shape as input
        self.assertEqual(y.shape, x.shape)
        
        # Manually calculate RMS norm for the first element to verify
        var = x[0, 0].pow(2).mean()
        scale = torch.rsqrt(var + rms_norm.eps)
        expected = x[0, 0] * scale * rms_norm.weight
        
        # Check that values are close
        torch.testing.assert_close(y[0, 0], expected, rtol=1e-4, atol=1e-4)

class TestSparseMoEModel(unittest.TestCase):
    def setUp(self):
        # Create a small config for testing
        self.config = SparseMoEConfig(
            vocab_size=1000,
            dim=64,
            inter_dim=256,
            moe_inter_dim=128,
            n_layers=2,
            n_dense_layers=1,
            n_heads=4,
            n_routed_experts=8,
            n_shared_experts=1,
            n_activated_experts=2,
            max_seq_len=128,
            max_batch_size=2,
            use_cpp_backend=False  # Use Python implementation for testing
        )
        
        # Create a small model for testing
        self.model = SparseMoEModel(self.config)
        
        # Sample input tokens
        self.tokens = torch.randint(0, self.config.vocab_size, (2, 8))
        
    def test_initialization(self):
        """Test model initialization."""
        # Check that all components were created
        self.assertEqual(len(self.model.layers), self.config.n_layers)
        self.assertEqual(self.model.embed.num_embeddings, self.config.vocab_size)
        self.assertEqual(self.model.embed.embedding_dim, self.config.dim)
        
        # Check that first layer is MLP and second layer is DNPS
        self.assertFalse(hasattr(self.model.layers[0].ffn, 'experts'))  # First layer is MLP
        self.assertTrue(hasattr(self.model.layers[1].ffn, 'experts'))   # Second layer is DNPS
        
    def test_precompute_freqs_cis(self):
        """Test precomputation of rotary embeddings."""
        freqs_cis = self.model.freqs_cis
        
        # Check shape of precomputed embeddings
        expected_shape = (self.config.max_seq_len, self.config.qk_rope_head_dim // 2)
        self.assertEqual(freqs_cis.shape, expected_shape)
        
        # Check dtype and device
        self.assertEqual(freqs_cis.dtype, torch.complex64)
        
    def test_forward(self):
        """Test forward pass."""
        # Forward pass in training mode
        self.model.train()
        output_train = self.model(self.tokens)
        
        # Check output shape in training mode
        expected_shape_train = (self.tokens.shape[0], self.tokens.shape[1], self.config.vocab_size)
        self.assertEqual(output_train.shape, expected_shape_train)
        
        # Forward pass in evaluation mode
        self.model.eval()
        output_eval = self.model(self.tokens)
        
        # Check output shape in eval mode (should return logits for last position)
        expected_shape_eval = (self.tokens.shape[0], self.config.vocab_size)
        self.assertEqual(output_eval.shape, expected_shape_eval)
        
    def test_generate(self):
        """Test text generation."""
        self.model.eval()
        
        # Generate text with greedy decoding
        max_new_tokens = 4
        generated = self.model.generate(
            self.tokens, 
            max_new_tokens=max_new_tokens, 
            temperature=0.0  # Greedy decoding
        )
        
        # Check shape of generated text
        expected_shape = (self.tokens.shape[0], self.tokens.shape[1] + max_new_tokens)
        self.assertEqual(generated.shape, expected_shape)
        
        # Check that the original tokens are preserved
        torch.testing.assert_close(generated[:, :self.tokens.shape[1]], self.tokens)
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_support(self):
        """Test that the model works on CUDA."""
        if torch.cuda.is_available():
            model_cuda = SparseMoEModel(self.config).cuda()
            tokens_cuda = self.tokens.cuda()
            
            # Forward pass on CUDA
            output_cuda = model_cuda(tokens_cuda)
            
            # Check output is on CUDA
            self.assertTrue(output_cuda.is_cuda)

if __name__ == '__main__':
    unittest.main()