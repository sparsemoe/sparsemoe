import unittest
import os
import sys
import torch

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sparsemoe.dnps import DNPSLayer

class TestDNPSLayer(unittest.TestCase):
    def setUp(self):
        # Set up common test parameters
        self.batch_size = 2
        self.seq_len = 4
        self.dim = 16
        self.num_experts = 8
        self.num_selected = 2
        
        # Create a sample input tensor
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.dim)
        
        # Create DNPSLayer with Python-only implementation
        self.dnps_layer = DNPSLayer(
            dim=self.dim,
            num_experts=self.num_experts,
            num_selected=self.num_selected,
            use_cpp_backend=False  # Force Python implementation
        )
        
    def test_initialization(self):
        """Test that the DNPS layer is initialized correctly."""
        self.assertEqual(self.dnps_layer.dim, self.dim)
        self.assertEqual(self.dnps_layer.num_experts, self.num_experts)
        self.assertEqual(self.dnps_layer.num_selected, self.num_selected)
        
        # Check that all experts were created
        self.assertEqual(len(self.dnps_layer.experts), self.num_experts)
        
        # Check that the router was created
        self.assertIsNotNone(self.dnps_layer.router)
        self.assertEqual(self.dnps_layer.router.in_features, self.dim)
        self.assertEqual(self.dnps_layer.router.out_features, self.num_experts)
        
    def test_forward_shape(self):
        """Test that the forward pass produces output with the correct shape."""
        output = self.dnps_layer(self.input_tensor)
        
        # Output should have the same shape as input
        self.assertEqual(output.shape, self.input_tensor.shape)
        
    def test_forward_different_batch_sizes(self):
        """Test that the forward pass works with different batch sizes."""
        # Test with batch size 1
        input_tensor_1 = torch.randn(1, self.seq_len, self.dim)
        output_1 = self.dnps_layer(input_tensor_1)
        self.assertEqual(output_1.shape, input_tensor_1.shape)
        
        # Test with batch size 4
        input_tensor_4 = torch.randn(4, self.seq_len, self.dim)
        output_4 = self.dnps_layer(input_tensor_4)
        self.assertEqual(output_4.shape, input_tensor_4.shape)
        
    def test_expert_selection(self):
        """Test that routing selects the correct number of experts."""
        # Set up custom simple router to control expert selection behavior
        with torch.no_grad():
            # Set router weights to route to specific experts predictably
            self.dnps_layer.router.weight.zero_()
            self.dnps_layer.router.bias.zero_()
            
            # Make first token always select experts 0,1 and second token select experts 2,3
            self.dnps_layer.router.bias[0] = 10.0
            self.dnps_layer.router.bias[1] = 9.0
            self.dnps_layer.router.bias[2] = 8.0
            self.dnps_layer.router.bias[3] = 7.0
        
        # Compute routing logits manually
        routing_logits = self.dnps_layer.router(self.input_tensor)
        
        # Check that top logits correspond to the expected experts
        top_logits, top_indices = torch.topk(routing_logits, self.num_selected, dim=-1)
        
        # Each token should select the same experts due to our bias settings
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                selected_experts = top_indices[i, j].tolist()
                # Should select experts 0 and 1 based on our bias settings
                self.assertIn(0, selected_experts)
                self.assertIn(1, selected_experts)
    
    def test_hierarchical_routing(self):
        """Test hierarchical routing with expert groups."""
        # Create DNPSLayer with hierarchical routing
        dnps_layer_hierarchical = DNPSLayer(
            dim=self.dim,
            num_experts=8,
            num_selected=2,
            group_size=2,  # 4 groups of 2 experts each
            use_cpp_backend=False  # Force Python implementation
        )
        
        # Forward pass should still work
        output = dnps_layer_hierarchical(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_support(self):
        """Test that the DNPS layer works on CUDA."""
        # Move model and inputs to CUDA
        dnps_layer_cuda = DNPSLayer(
            dim=self.dim,
            num_experts=self.num_experts,
            num_selected=self.num_selected,
            use_cpp_backend=False
        ).cuda()
        
        input_cuda = self.input_tensor.cuda()
        
        # Forward pass on CUDA
        output_cuda = dnps_layer_cuda(input_cuda)
        
        # Check output is on CUDA and has correct shape
        self.assertTrue(output_cuda.is_cuda)
        self.assertEqual(output_cuda.shape, input_cuda.shape)

if __name__ == '__main__':
    unittest.main()