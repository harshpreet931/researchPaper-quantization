#!/usr/bin/env python3
"""Benchmark MLX quantized models."""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results_gptq_awq")
RESULTS_DIR.mkdir(exist_ok=True)

def run_benchmark(model_path: str, model_name: str, quant_method: str, enable_thinking: bool = False):
    """Run benchmark on a model.
    
    Args:
        model_path: Path to the model (HuggingFace ID or local path)
        model_name: Display name for the model
        quant_method: Quantization method (fp16, 4bit, 8bit)
        enable_thinking: Enable thinking mode for Qwen3.5 models
    """
    from mlx_lm import load, generate
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {model_name} ({quant_method}){' [thinking]' if enable_thinking else ''}")
    print(f"{'='*60}")
    
    result = {
        "model": model_name,
        "quant_method": quant_method,
        "model_path": model_path,
        "enable_thinking": enable_thinking,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        load_start = time.time()
        model, tokenizer = load(model_path)
        load_time = time.time() - load_start
        result["load_time_seconds"] = load_time
        
        # Count parameters and size
        from mlx.utils import tree_flatten
        params = tree_flatten(model.parameters())
        total_params = sum(p.size for _, p in params)
        total_size = sum(p.size * p.itemsize for _, p in params)
        
        result["parameters"] = total_params
        result["model_size_gb"] = total_size / (1024**3)
        result["bits_per_weight"] = total_size * 8 / total_params
        
        print(f"Parameters: {total_params:,}")
        print(f"Model size: {result['model_size_gb']:.2f} GB")
        print(f"Bits/weight: {result['bits_per_weight']:.2f}")
        
        # Benchmark latency
        prompt = "Explain the concept of machine learning in simple terms."
        max_tokens = 32
        num_runs = 3
        
        # Warmup
        warmup_kwargs = {"max_tokens": max_tokens, "verbose": False}
        if enable_thinking:
            warmup_kwargs["enable_thinking"] = True
        _ = generate(model, tokenizer, prompt, **warmup_kwargs)
        
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            gen_kwargs = {"max_tokens": max_tokens, "verbose": False}
            if enable_thinking:
                gen_kwargs["enable_thinking"] = True
            _ = generate(model, tokenizer, prompt, **gen_kwargs)
            latencies.append(time.perf_counter() - start)
        
        avg_latency = sum(latencies) / len(latencies)
        result["total_latency_seconds"] = avg_latency
        result["ms_per_token"] = (avg_latency / max_tokens) * 1000
        result["tokens_per_second"] = max_tokens / avg_latency
        
        print(f"Latency: {result['ms_per_token']:.2f} ms/token")
        print(f"Throughput: {result['tokens_per_second']:.2f} tok/s")
        
        # MMLU sample
        sample_questions = [
            {"q": "What is the time complexity of binary search?", "choices": "A) O(n) B) O(log n) C) O(n^2) D) O(1)", "a": "B"},
            {"q": "What is the SI unit of electric current?", "choices": "A) Volt B) Ohm C) Ampere D) Watt", "a": "C"},
            {"q": "What is the derivative of x^2?", "choices": "A) x B) 2x C) 2 D) x^2", "a": "B"},
            {"q": "What organelle is responsible for photosynthesis?", "choices": "A) Mitochondria B) Ribosome C) Chloroplast D) Nucleus", "a": "C"},
            {"q": "What is the chemical symbol for gold?", "choices": "A) Go B) Au C) Ag D) Gd", "a": "B"},
        ]
        
        correct = 0
        for sq in sample_questions:
            p = f"Question: {sq['q']}\nChoices: {sq['choices']}\nAnswer with just the letter:\nAnswer:"
            try:
                resp = generate(model, tokenizer, p, max_tokens=5, verbose=False)
                if sq["a"] in resp.upper():
                    correct += 1
            except:
                pass
        
        result["accuracy"] = (correct / len(sample_questions)) * 100
        result["correct"] = correct
        result["total"] = len(sample_questions)
        print(f"MMLU Sample Accuracy: {result['accuracy']:.1f}%")
        
        result["status"] = "success"
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)
    
    return result


def main():
    # Updated model list with Qwen3.5 and Phi-3 Mini
    # Note: Speculative decoding is disabled due to known bugs
    models = [
        # Qwen3.5 FP16 models
        ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "fp16", False),
        ("Qwen/Qwen3.5-2B", "Qwen3.5-2B", "fp16", False),
        ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", "fp16", False),
        # Qwen3.5 with thinking mode enabled
        ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "fp16", True),
        ("Qwen/Qwen3.5-2B", "Qwen3.5-2B", "fp16", True),
        ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", "fp16", True),
        # Qwen3.5 INT4 (pre-quantized MLX versions)
        ("mlx-community/Qwen3.5-0.8B-4bit", "Qwen3.5-0.8B", "4bit", False),
        ("mlx-community/Qwen3.5-2B-4bit", "Qwen3.5-2B", "4bit", False),
        ("mlx-community/Qwen3.5-4B-4bit", "Qwen3.5-4B", "4bit", False),
        # Qwen3.5 INT8
        ("mlx-community/Qwen3.5-0.8B-8bit", "Qwen3.5-0.8B", "8bit", False),
        ("mlx-community/Qwen3.5-2B-8bit", "Qwen3.5-2B", "8bit", False),
        ("mlx-community/Qwen3.5-4B-8bit", "Qwen3.5-4B", "8bit", False),
        # Phi-3 Mini
        ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-Mini", "fp16", False),
        ("mlx-community/Phi-3-mini-4k-instruct-4bit", "Phi-3-Mini", "4bit", False),
        ("mlx-community/Phi-3-mini-4k-instruct-8bit", "Phi-3-Mini", "8bit", False),
    ]
    
    all_results = []
    
    for model_path, name, quant, thinking in models:
        result = run_benchmark(model_path, name, quant, enable_thinking=thinking)
        all_results.append(result)
        
        # Create filename with thinking mode suffix if applicable
        thinking_suffix = "_thinking" if thinking else ""
        fname = f"{name.replace('-', '_')}_{quant}{thinking_suffix}.json"
        with open(RESULTS_DIR / fname, "w") as f:
            json.dump(result, f, indent=2)
    
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'Quant':<8} {'Size(GB)':<10} {'BPW':<8} {'ms/tok':<10} {'MMLU%':<8}")
    print("-" * 60)
    for r in all_results:
        if r.get("status") == "success":
            print(f"{r['model']:<15} {r['quant_method']:<8} {r['model_size_gb']:<10.2f} {r['bits_per_weight']:<8.2f} {r['ms_per_token']:<10.2f} {r['accuracy']:<8.1f}")
    
    print(f"\nResults saved to: {RESULTS_DIR.absolute()}")


if __name__ == "__main__":
    main()
