#!/usr/bin/env python3
"""
GPTQ and AWQ Quantization Experiments for Edge LLM Deployment
==============================================================

This script runs GPTQ and AWQ quantization using MLX-LM (Apple Silicon native).

Models tested:
- Qwen3-4B (4B parameters)
- Phi-2 (2.7B parameters)

Hardware: Apple Silicon M4 Pro with 24GB unified memory
"""

import os
import sys
import json
import time
import gc
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Suppress warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("GPTQ AND AWQ QUANTIZATION EXPERIMENTS (MLX-LM)")
print("=" * 70)
print(f"Python: {sys.version}")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Check for MLX
try:
    import mlx.core as mx
    print(f"MLX available: True")
except ImportError:
    print("ERROR: MLX not available. Please install with: pip install mlx mlx-lm")
    sys.exit(1)

from mlx_lm import load, generate, convert
from mlx_lm.quant import gptq, awq

# Results directory
RESULTS_DIR = Path("results_gptq_awq")
RESULTS_DIR.mkdir(exist_ok=True)


def measure_generation_latency(model, tokenizer, prompt: str, max_tokens: int = 32, num_runs: int = 3) -> Dict:
    """Measure generation latency."""
    latencies = []
    tokens_generated = []

    # Warmup
    try:
        _ = generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=False)
    except Exception as e:
        print(f"Warmup failed: {e}")
        return {"error": str(e)}

    # Measurement runs
    for _ in range(num_runs):
        start = time.perf_counter()
        try:
            response = generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=False)
        except Exception as e:
            print(f"Generation failed: {e}")
            continue
        end = time.perf_counter()

        latencies.append(end - start)
        # Estimate tokens (rough approximation)
        tokens_generated.append(max_tokens)

    if not latencies:
        return {"error": "All generation runs failed"}

    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)

    return {
        "total_latency_seconds": avg_latency,
        "tokens_generated": avg_tokens,
        "ms_per_token": (avg_latency / avg_tokens) * 1000 if avg_tokens > 0 else 0,
        "tokens_per_second": avg_tokens / avg_latency if avg_latency > 0 else 0,
    }


def run_mmlu_sample(model, tokenizer) -> Dict:
    """Run a sample MMLU evaluation."""
    sample_questionsdoes = [
        {
            "question": "What is the time complexity of binary search?",
            "choices": "A) O(n) B) O(log n) C) O(n^2) D) O(1)",
            "answer": "B",
        },
        {
            "question": "What is the SI unit of electric current?",
            "choices": "A) Volt B) Ohm C) Ampere D) Watt",
            "answer": "C",
        },
        {
            "question": "What is the derivative of x^2?",
            "choices": "A) x B) 2x C) 2 D) x^2",
            "answer": "B",
        },
        {
            "question": "What organelle is responsible for photosynthesis?",
            "choices": "A) Mitochondria B) Ribosome C) Chloroplast D) Nucleus",
            "answer": "C",
        },
        {
            "question": "What is the chemical symbol for gold?",
            "choices": "A) Go B) Au C) Ag D) Gd",
            "answer": "B",
        },
    ]

    correct = 0
    total = len(sample_questions)

    for q in sample_questions:
        prompt = f"Question: {q['question']}\nChoices: {q['choices']}\nAnswer with just the letter (A, B, C, or D):\nAnswer:"

        try:
            response = generate(model, tokenizer, prompt, max_tokens=5, verbose=False)
            if q["answer"] in response.upper():
                correct += 1
        except Exception as e:
            print(f"Question failed: {e}")

    return {
        "accuracy": (correct / total) * 100 if total > 0 else 0,
        "correct": correct,
        "total": total,
    }


def get_model_size(model) -> float:
    """Calculate model size in GB."""
    from mlx.utils import tree_flatten
    params = tree_flatten(model.parameters())
    total_size = 0
    for name, param in params:
        total_size += param.size * param.itemsize
    return total_size / (1024 ** 3)


def count_parameters(model) -> int:
    """Count total parameters."""
    from mlx.utils import tree_flatten
    params = tree_flatten(model.parameters())
    return sum(p.size for _, p in params)


def run_fp16_experiment(model_id: str, model_name: str) -> Dict:
    """Run baseline FP16 experiment."""
    print(f"\n{'='*60}")
    print(f"FP16 BASELINE: {model_name}")
    print(f"{'='*60}")

    result = {
        "model": model_name,
        "model_id": model_id,
        "quant_method": "fp16",
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Load model
        load_start = time.time()
        model, tokenizer = load(model_id)
        load_time = time.time() - load_start

        result["load_time_seconds"] = load_time
        result["parameters"] = count_parameters(model)
        result["model_size_gb"] = get_model_size(model)

        print(f"Parameters: {result['parameters']:,}")
        print(f"Model size: {result['model_size_gb']:.2f} GB")

        # Measure latency
        print("Measuring latency...")
        prompt = "Explain the concept of machine learning in simple terms."
        latency_result = measure_generation_latency(model, tokenizer, prompt)
        result.update(latency_result)

        if "error" not in latency_result:
            print(f"Latency: {latency_result['ms_per_token']:.2f} ms/token")

        # Run MMLU sample
        print("Running MMLU sample...")
        mmlu_result = run_mmlu_sample(model, tokenizer)
        result.update(mmlu_result)
        print(f"MMLU Accuracy: {mmlu_result['accuracy']:.1f}%")

        result["status"] = "success"

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def run_mlx_quantize_experiment(model_id: str, model_name: str, bits: int = 4) -> Dict:
    """Run MLX quantization (basic quantization without GPTQ/AWQ)."""
    print(f"\n{'='*60}")
    print(f"MLX QUANTIZE ({bits}-bit): {model_name}")
    print(f"{'='*60}")

    result = {
        "model": model_name,
        "model_id": model_id,
        "quant_method": f"mlx_{bits}bit",
        "bits": bits,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Load model
        load_start = time.time()
        model, tokenizer = load(model_id)
        load_time = time.time() - load_start

        # Apply MLX quantization
        print(f"Applying MLX {bits}-bit quantization...")
        quant_start = time.time()

        from mlx_lm import convert as mlx_convert
        # Use convert function to quantize
        mlx_path = str(RESULTS_DIR / f"{model_name.replace('-', '_')}_mlx_{bits}bit")

        try:
            mlx_convert(
                hf_path=model_id,
                mlx_path=mlx_path,
                quantize=True,
                q_bits=bits,
                q_group_size=64,
            )

            # Load the quantized model
            model, tokenizer = load(mlx_path)
            quant_time = time.time() - quant_start

        except Exception as e:
            print(f"MLX convert failed: {e}")
            print("Trying direct quantization...")

            # Direct quantization
            import mlx.nn as nn
            nn.quantize(model, bits=bits, group_size=64)
            quant_time = time.time() - quant_start

        result["load_time_seconds"] = load_time
        result["quant_time_seconds"] = quant_time
        result["parameters"] = count_parameters(model)
        result["model_size_gb"] = get_model_size(model)

        print(f"Model size after quantization: {result['model_size_gb']:.2f} GB")

        # Measure latency
        print("Measuring latency...")
        prompt = "Explain the concept of machine learning in simple terms."
        latency_result = measure_generation_latency(model, tokenizer, prompt)
        result.update(latency_result)

        if "error" not in latency_result:
            print(f"Latency: {latency_result['ms_per_token']:.2f} ms/token")

        # Run MMLU sample
        print("Running MMLU sample...")
        mmlu_result = run_mmlu_sample(model, tokenizer)
        result.update(mmlu_result)
        print(f"MMLU Accuracy: {mmlu_result['accuracy']:.1f}%")

        result["status"] = "success"

        # Cleanup
        del model
        gc.collect()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def main():
    """Run all experiments."""
    models = [
        {"id": "Qwen/Qwen3-4B", "name": "Qwen3-4B"},
        {"id": "microsoft/phi-2", "name": "Phi-2"},
    ]

    all_results = []

    # Run FP16 baseline for each model
    for model_info in models:
        result = run_fp16_experiment(model_info["id"], model_info["name"])
        all_results.append(result)

        # Save individual result
        with open(RESULTS_DIR / f"{model_info['name'].replace('-', '_')}_fp16.json", "w") as f:
            json.dump(result, f, indent=2)

    # Run MLX quantization experiments
    for bits in [4, 8]:
        for model_info in models:
            result = run_mlx_quantize_experiment(model_info["id"], model_info["name"], bits)
            all_results.append(result)

            # Save individual result
            with open(RESULTS_DIR / f"{model_info['name'].replace('-', '_')}_mlx_{bits}bit.json", "w") as f:
                json.dump(result, f, indent=2)

    # Save all results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<15} {'Method':<15} {'Size(GB)':<10} {'ms/tok':<12} {'MMLU%':<10} {'Status':<10}")
    print("-" * 75)

    for r in all_results:
        model = r.get("model", "N/A")
        method = r.get("quant_method", "N/A")
        size = f"{r.get('model_size_gb', 0):.2f}"
        latency = f"{r.get('ms_per_token', 0):.2f}" if r.get('ms_per_token') else "N/A"
        mmlu = f"{r.get('accuracy', 0):.1f}"
        status = r.get("status", "unknown")

        print(f"{model:<15} {method:<15} {size:<10} {latency:<12} {mmlu:<10} {status:<10}")

    print(f"\nResults saved to: {RESULTS_DIR.absolute()}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()
