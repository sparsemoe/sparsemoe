import unittest
import os
import sys
import torch
import json
import tempfile

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sparsemoe import create_model

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Create a minimal config for testing
        self.config_data = {
            "vocab_size": 1000,
            "dim": 64,
            "inter_dim": 256,
            "moe_inter_dim": 128,
            "n_layers": 2,
            "n_dense_layers": 1,
            "n_heads": 4,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "max_seq_len": 128,
            "max_batch_size": 2,
            "use_cpp_backend": False,
            "route_scale": 1.0,
            "score_func": "sigmoid",
            "q_lora_rank": 16,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "eos_token_id": 2
        }
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            json.dump(self.config_data, tmp)
            self.config_path = tmp.name
            
        # Create a sample input
        self.input_ids = torch.randint(0, self.config_data["vocab_size"], (1, 8))
        
    def tearDown(self):
        # Clean up the temporary file
        os.unlink(self.config_path)
        
    def test_create_model_from_config(self):
        """Test creating a model from a config file."""
        model = create_model(config_path=self.config_path)
        
        # Check that model was created successfully
        self.assertIsNotNone(model)
        self.assertEqual(model.config.vocab_size, self.config_data["vocab_size"])
        self.assertEqual(model.config.dim, self.config_data["dim"])
        
    def test_create_model_from_size(self):
        """Test creating a model from a predefined size."""
        # Note: This only tests that the function runs without errors
        # The actual parameters depend on the defaults in the create_model function
        model = create_model(model_size="16B", use_cpp_backend=False)
        
        # Check that model was created successfully
        self.assertIsNotNone(model)
        
    def test_model_inference(self):
        """Test model inference (forward pass and generation)."""
        model = create_model(config_path=self.config_path)
        
        # Forward pass
        logits = model(self.input_ids)
        
        # Check output shape
        self.assertEqual(logits.shape, (1, self.config_data["vocab_size"]))
        
        # Generation
        max_new_tokens = 4
        generated = model.generate(
            self.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9
        )
        
        # Check generated shape
        expected_shape = (self.input_ids.shape[0], self.input_ids.shape[1] + max_new_tokens)
        self.assertEqual(generated.shape, expected_shape)
        
        # Make sure the original input is preserved in the output
        torch.testing.assert_close(generated[:, :self.input_ids.shape[1]], self.input_ids)
        
    def test_model_save_load(self):
        """Test saving and loading model weights."""
        model = create_model(config_path=self.config_path)
        
        # Save model to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            # Save weights
            model.save_weights(model_path)
            
            # Create a new model
            new_model = create_model(config_path=self.config_path)
            
            # Load weights
            new_model.load_weights(model_path)
            
            # Check that weights were loaded correctly by comparing forward pass results
            with torch.no_grad():
                original_output = model(self.input_ids)
                loaded_output = new_model(self.input_ids)
                
                # Outputs should be identical
                torch.testing.assert_close(original_output, loaded_output)
        finally:
            # Clean up the temporary file
            os.unlink(model_path)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_support(self):
        """Test that the model works on CUDA."""
        model = create_model(config_path=self.config_path)
        
        # Move model to CUDA
        model.cuda()
        input_cuda = self.input_ids.cuda()
        
        # Forward pass on CUDA
        output_cuda = model(input_cuda)
        
        # Check that output is on CUDA
        self.assertTrue(output_cuda.is_cuda)
        
        # Generation on CUDA
        generated_cuda = model.generate(input_cuda, max_new_tokens=4)
        
        # Check that generated is on CUDA
        self.assertTrue(generated_cuda.is_cuda)

if __name__ == '__main__':
    unittest.main()