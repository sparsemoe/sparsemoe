#!/usr/bin/env python3

"""
Benchmark script comparing Python vs C++ backend performance.

Example usage:
  python examples/performance_benchmark.py --model-size 16B --ckpt-path /path/to/weights
"""

import argparse
import time
import torch
from transformers import AutoTokenizer
from sparsemoe import create_model

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SparseMoE performance")
    parser.add_argument("--model-size", type=str, default="16B", 
                        choices=["16B", "236B", "671B", "1024B"],
                        help="Model size to benchmark")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Optional path to config file")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer-path", type=str, default="gpt2",
                        help="Path to tokenizer (defaults to gpt2)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for benchmark")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for benchmark")
    parser.add_argument("--num-tokens", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of iterations for each test")
    return parser.parse_args()

def benchmark_model(model, input_ids, num_tokens, iterations):
    """Run benchmark and return average time per token."""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = torch.tensor(input_ids, device=device).repeat(1, 1)
    
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model.generate(input_tensor, max_new_tokens=10)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.time()
            _ = model.generate(input_tensor, max_new_tokens=num_tokens)
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Calculate metrics
    avg_time = sum(times) / len(times)
    tokens_per_second = num_tokens / avg_time
    
    return {
        "avg_time": avg_time,
        "tokens_per_second": tokens_per_second,
        "times": times
    }

def main():
    args = parse_args()
    
    # Check for CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU will be very slow.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Prepare input
    input_text = "This is a test of the SparseMoE language model performance benchmark. " * 4
    input_ids = tokenizer.encode(input_text)
    if len(input_ids) > args.seq_len:
        input_ids = input_ids[:args.seq_len]
    
    # Test C++ backend
    print("\nTesting with C++ backend...")
    cpp_model = create_model(
        config_path=args.config_path,
        model_size=args.model_size,
        use_cpp_backend=True
    )
    cpp_model.load_weights(args.ckpt_path)
    cpp_model = cpp_model.to(device)
    cpp_results = benchmark_model(cpp_model, input_ids, args.num_tokens, args.iterations)
    
    # Test Python-only implementation
    print("\nTesting with Python-only implementation...")
    py_model = create_model(
        config_path=args.config_path,
        model_size=args.model_size,
        use_cpp_backend=False
    )
    py_model.load_weights(args.ckpt_path)
    py_model = py_model.to(device)
    py_results = benchmark_model(py_model, input_ids, args.num_tokens, args.iterations)
    
    # Calculate speedup
    speedup = py_results["avg_time"] / cpp_results["avg_time"]
    
    # Print results
    print("\n" + "=" * 60)
    print(f"PERFORMANCE BENCHMARK: SparseMoE-{args.model_size}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Tokens generated: {args.num_tokens}")
    print(f"Iterations: {args.iterations}")
    print("\nRESULTS:")
    print("-" * 60)
    print(f"C++ Backend:")
    print(f"  - Average time: {cpp_results['avg_time']:.4f} seconds")
    print(f"  - Tokens/second: {cpp_results['tokens_per_second']:.2f}")
    print(f"Python-only Implementation:")
    print(f"  - Average time: {py_results['avg_time']:.4f} seconds")
    print(f"  - Tokens/second: {py_results['tokens_per_second']:.2f}")
    print("-" * 60)
    print(f"Speedup with C++ backend: {speedup:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    main()