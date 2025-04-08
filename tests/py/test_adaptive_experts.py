import unittest
import os
import sys
import torch

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sparsemoe.adaptive_experts import ExpertUsageTracker, AdaptiveExpert, AdaptiveExpertLayer


class TestExpertUsageTracker(unittest.TestCase):
    def setUp(self):
        # Set up common test parameters
        self.num_experts = 4
        self.window_size = 10
        self.tracker = ExpertUsageTracker(
            num_experts=self.num_experts,
            window_size=self.window_size
        )
        
    def test_initialization(self):
        """Test that the tracker is initialized correctly."""
        self.assertEqual(self.tracker.num_experts, self.num_experts)
        self.assertEqual(self.tracker.window_size, self.window_size)
        self.assertEqual(self.tracker.total_forwards, 0)
        self.assertEqual(self.tracker.activation_counts.shape, torch.Size([self.num_experts]))
        
    def test_update(self):
        """Test updating the tracker with expert activations."""
        # Simulate a forward pass where experts 0 and 1 are activated
        expert_indices = torch.tensor([[0, 1]])
        importance_scores = torch.tensor([[0.7, 0.3]])
        
        self.tracker.update(expert_indices, importance_scores)
        
        # Check that the activation counts were updated
        self.assertEqual(self.tracker.activation_counts[0].item(), 1)
        self.assertEqual(self.tracker.activation_counts[1].item(), 1)
        self.assertEqual(self.tracker.activation_counts[2].item(), 0)
        
        # Check that the importance EMA was updated
        self.assertGreater(self.tracker.importance_ema[0].item(), self.tracker.importance_ema[2].item())
        
        # Check that the total forwards was incremented
        self.assertEqual(self.tracker.total_forwards, 1)
        
    def test_should_scale(self):
        """Test the scaling decision logic."""
        # Should not scale initially
        self.assertFalse(self.tracker.should_scale())
        
        # Update multiple times but still below window size
        for _ in range(self.window_size - 1):
            expert_indices = torch.tensor([[0, 1]])
            importance_scores = torch.tensor([[0.5, 0.5]])
            self.tracker.update(expert_indices, importance_scores)
            
        # Still should not scale
        self.assertFalse(self.tracker.should_scale())
        
        # One more update should trigger scaling
        self.tracker.update(expert_indices, importance_scores)
        self.assertTrue(self.tracker.should_scale())
        
    def test_get_scaling_factors(self):
        """Test the calculation of scaling factors."""
        # Simulate biased usage toward expert 0
        for _ in range(self.window_size):
            expert_indices = torch.tensor([[0, 1]])
            importance_scores = torch.tensor([[0.9, 0.1]])
            self.tracker.update(expert_indices, importance_scores)
            
        # Get scaling factors
        scaling_factors = self.tracker.get_scaling_factors()
        
        # Expert 0 should have a larger scaling factor
        self.assertGreater(scaling_factors[0].item(), scaling_factors[1].item())
        
        # Scaling factors should be bounded
        for factor in scaling_factors:
            self.assertGreaterEqual(factor.item(), 0.5)
            self.assertLessEqual(factor.item(), 2.0)


class TestAdaptiveExpert(unittest.TestCase):
    def setUp(self):
        # Set up common test parameters
        self.input_dim = 32
        self.initial_dim = 64
        self.output_dim = 32
        self.min_dim = 16
        self.max_dim = 128
        
        self.expert = AdaptiveExpert(
            input_dim=self.input_dim,
            initial_dim=self.initial_dim,
            output_dim=self.output_dim,
            min_dim=self.min_dim,
            max_dim=self.max_dim
        )
        
    def test_initialization(self):
        """Test that the adaptive expert is initialized correctly."""
        self.assertEqual(self.expert.input_dim, self.input_dim)
        self.assertEqual(self.expert.hidden_dim, self.initial_dim)
        self.assertEqual(self.expert.output_dim, self.output_dim)
        
        # Check layer dimensions
        self.assertEqual(self.expert.up_proj.in_features, self.input_dim)
        self.assertEqual(self.expert.up_proj.out_features, self.initial_dim)
        self.assertEqual(self.expert.down_proj.in_features, self.initial_dim)
        self.assertEqual(self.expert.down_proj.out_features, self.output_dim)
        
    def test_forward(self):
        """Test the forward pass."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.input_dim)
        
        output = self.expert(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.output_dim))
        
    def test_resize_grow(self):
        """Test resizing the expert to a larger hidden dimension."""
        # Original dimension
        old_dim = self.expert.hidden_dim
        
        # Grow the expert
        new_dim = 96  # Between initial_dim and max_dim
        success = self.expert.resize(new_dim)
        
        # Resizing should succeed
        self.assertTrue(success)
        
        # Check new dimensions
        self.assertEqual(self.expert.hidden_dim, new_dim)
        self.assertEqual(self.expert.up_proj.out_features, new_dim)
        self.assertEqual(self.expert.down_proj.in_features, new_dim)
        
        # Test forward pass with new dimensions
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.input_dim)
        output = self.expert(input_tensor)
        self.assertEqual(output.shape, (batch_size, self.output_dim))
        
    def test_resize_shrink(self):
        """Test resizing the expert to a smaller hidden dimension."""
        # Original dimension
        old_dim = self.expert.hidden_dim
        
        # Shrink the expert
        new_dim = 32  # Between min_dim and initial_dim
        success = self.expert.resize(new_dim)
        
        # Resizing should succeed
        self.assertTrue(success)
        
        # Check new dimensions
        self.assertEqual(self.expert.hidden_dim, new_dim)
        self.assertEqual(self.expert.up_proj.out_features, new_dim)
        self.assertEqual(self.expert.down_proj.in_features, new_dim)
        
        # Test forward pass with new dimensions
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.input_dim)
        output = self.expert(input_tensor)
        self.assertEqual(output.shape, (batch_size, self.output_dim))
        
    def test_resize_bound_check(self):
        """Test that resizing respects min and max bounds."""
        # Try to resize below minimum
        below_min = self.min_dim - 8
        success = self.expert.resize(below_min)
        
        # Resizing should succeed, but dimension should be clamped to min_dim
        self.assertTrue(success)
        self.assertEqual(self.expert.hidden_dim, self.min_dim)
        
        # Try to resize above maximum
        above_max = self.max_dim + 8
        success = self.expert.resize(above_max)
        
        # Resizing should succeed, but dimension should be clamped to max_dim
        self.assertTrue(success)
        self.assertEqual(self.expert.hidden_dim, self.max_dim)


class TestAdaptiveExpertLayer(unittest.TestCase):
    def setUp(self):
        # Set up common test parameters
        self.dim = 32
        self.num_experts = 4
        self.num_selected = 2
        self.initial_expert_dim = 64
        self.min_expert_dim = 16
        self.max_expert_dim = 128
        self.scaling_interval = 5
        
        self.layer = AdaptiveExpertLayer(
            dim=self.dim,
            num_experts=self.num_experts,
            num_selected=self.num_selected,
            initial_expert_dim=self.initial_expert_dim,
            min_expert_dim=self.min_expert_dim,
            max_expert_dim=self.max_expert_dim,
            scaling_interval=self.scaling_interval
        )
        
    def test_initialization(self):
        """Test that the adaptive expert layer is initialized correctly."""
        self.assertEqual(self.layer.dim, self.dim)
        self.assertEqual(self.layer.num_experts, self.num_experts)
        self.assertEqual(self.layer.num_selected, self.num_selected)
        
        # Check that experts were created
        self.assertEqual(len(self.layer.experts), self.num_experts)
        
        # Check that all experts have the initial dimension
        for expert in self.layer.experts:
            self.assertEqual(expert.hidden_dim, self.initial_expert_dim)
            
        # Check that the router was created
        self.assertEqual(self.layer.router.in_features, self.dim)
        self.assertEqual(self.layer.router.out_features, self.num_experts)
        
    def test_forward_shape(self):
        """Test that the forward pass produces output with the correct shape."""
        batch_size = 2
        seq_len = 4
        input_tensor = torch.randn(batch_size, seq_len, self.dim)
        
        output = self.layer(input_tensor)
        
        # Output should have the same shape as input
        self.assertEqual(output.shape, input_tensor.shape)
        
    def test_expert_scaling(self):
        """Test that experts are scaled based on usage patterns."""
        batch_size = 2
        seq_len = 4
        
        # First pass - experts should not be scaled yet
        input_tensor = torch.randn(batch_size, seq_len, self.dim)
        self.layer(input_tensor)
        
        # Get initial expert sizes
        initial_sizes = self.layer.get_expert_sizes()
        
        # Initialize router weights to bias toward specific experts
        with torch.no_grad():
            # Bias heavily toward first expert, then second, etc.
            self.layer.router.bias.data = torch.tensor(
                [5.0, 4.0, 3.0, 2.0], dtype=torch.float32
            )
        
        # Run remaining passes until scaling interval is reached
        for _ in range(self.scaling_interval - 1):
            input_tensor = torch.randn(batch_size, seq_len, self.dim)
            self.layer(input_tensor)
            
        # Get expert sizes after scaling
        scaled_sizes = self.layer.get_expert_sizes()
        
        # First expert should be larger than average, last should be smaller
        # Not guaranteed due to randomness, but likely
        self.assertNotEqual(initial_sizes, scaled_sizes)
        
    def test_get_expert_stats(self):
        """Test the expert statistics reporting."""
        # Run a few forward passes
        batch_size = 2
        seq_len = 4
        for _ in range(3):
            input_tensor = torch.randn(batch_size, seq_len, self.dim)
            self.layer(input_tensor)
            
        # Get stats
        stats = self.layer.get_expert_stats()
        
        # Check that stats contains required fields
        self.assertIn("expert_sizes", stats)
        self.assertIn("importance_scores", stats)
        self.assertIn("activation_counts", stats)
        self.assertIn("scaling_history", stats)
        self.assertIn("total_params", stats)
        
        # Check that expert sizes match what's in the model
        self.assertEqual(stats["expert_sizes"], self.layer.get_expert_sizes())


if __name__ == "__main__":
    unittest.main()