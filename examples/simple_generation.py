#!/usr/bin/env python3

"""
Simple example showing how to generate text with SparseMoE.

Example usage:
  python examples/simple_generation.py --model-size 16B --ckpt-path /path/to/model/weights
"""

import argparse
import torch
from transformers import AutoTokenizer
from sparsemoe import create_model

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with SparseMoE")
    parser.add_argument("--model-size", type=str, default="16B", 
                       choices=["16B", "236B", "671B", "1024B"],
                       help="Model size to use")
    parser.add_argument("--config-path", type=str, default=None,
                       help="Optional path to config file")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer-path", type=str, default="gpt2",
                       help="Path to tokenizer (defaults to gpt2)")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Text prompt to generate from")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--use-cpp", action="store_true",
                       help="Use C++ backend when available")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create model
    print(f"Creating SparseMoE-{args.model_size} model...")
    model = create_model(
        config_path=args.config_path,
        model_size=args.model_size,
        use_cpp_backend=args.use_cpp
    )
    
    # Load weights
    print(f"Loading weights from {args.ckpt_path}...")
    model.load_weights(args.ckpt_path)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Tokenize prompt
    print(f"Generating from prompt: {args.prompt}")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()