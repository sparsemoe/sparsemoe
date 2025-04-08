import unittest
import os
import tempfile
import json
import sys
import torch

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sparsemoe.model import SparseMoEConfig

class TestConfig(unittest.TestCase):
    def test_default_config(self):
        """Test default configuration initialization."""
        config = SparseMoEConfig()
        
        # Check default values
        self.assertEqual(config.vocab_size, 129280)
        self.assertEqual(config.dim, 7168)
        self.assertEqual(config.n_layers, 61)
        self.assertEqual(config.n_heads, 128)
        self.assertEqual(config.max_seq_len, 131072)  # 4096 * 32
        
    def test_from_json(self):
        """Test loading configuration from JSON file."""
        # Create a temporary config file
        config_data = {
            "vocab_size": 50000,
            "dim": 2048,
            "inter_dim": 8192,
            "n_layers": 24,
            "n_heads": 32
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            json.dump(config_data, tmp)
            tmp_path = tmp.name
        
        try:
            # Load config from the temporary file
            config = SparseMoEConfig.from_json(tmp_path)
            
            # Check that values were loaded correctly
            self.assertEqual(config.vocab_size, 50000)
            self.assertEqual(config.dim, 2048)
            self.assertEqual(config.inter_dim, 8192)
            self.assertEqual(config.n_layers, 24)
            self.assertEqual(config.n_heads, 32)
            
            # Check that default values are used for unspecified parameters
            self.assertEqual(config.max_seq_len, 131072)  # Default value
        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)

if __name__ == '__main__':
    unittest.main()