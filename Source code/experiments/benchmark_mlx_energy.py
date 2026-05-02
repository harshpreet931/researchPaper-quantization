#!/usr/bin/env python3
"""Benchmark MLX models with energy tracking using CodeCarbon."""

import json
import time
from datetime import datetime
from pathlib import Path
from codecarbon import EmissionsTracker

RESULTS_DIR = Path("results_energy")
RESULTS_DIR.mkdir(exist_ok=True)

def run_benchmark_with_energy(model_path: str, model_name: str, quant_method: str, num_tokens: int = 100):
    """Run benchmark with energy tracking."""
    from mlx_lm import load, generate
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {model_name} ({quant_method})")
    print(f"{'='*60}")
    
    result = {
        "model": model_name,
        "quant_method": quant_method,
        "model_path": model_path,
        "num_tokens": num_tokens,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        # Load model
        load_start = time.time()
        model, tokenizer = load(model_path)
        load_time = time.time() - load_start
        result["load_time_seconds"] = load_time
        
        # Count parameters
        from mlx.utils import tree_flatten
        params = tree_flatten(model.parameters())
        total_params = sum(p.size for _, p in params)
        total_size = sum(p.size * p.itemsize for _, p in params)
        
        result["parameters"] = total_params
        result["model_size_gb"] = total_size / (1024**3)
        
        print(f"Model size: {result['model_size_gb']:.2f} GB")
        
        # Warmup
        prompt = "Explain the concept of machine learning in simple terms."
        _ = generate(model, tokenizer, prompt, max_tokens=32, verbose=False)
        
        # Benchmark with energy tracking
        tracker = EmissionsTracker(measure_power_secs=1)
        tracker.start()
        
        start_time = time.perf_counter()
        _ = generate(model, tokenizer, prompt, max_tokens=num_tokens, verbose=False)
        end_time = time.perf_counter()
        
        emissions = tracker.stop()
        
        # Calculate metrics
        total_time = end_time - start_time
        result["total_time_seconds"] = total_time
        result["ms_per_token"] = (total_time / num_tokens) * 1000
        result["tokens_per_second"] = num_tokens / total_time
        result["emissions_kg"] = emissions
        
        # Get energy from tracker
        try:
            if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data:
                result["energy_kwh"] = tracker.final_emissions_data.energy_consumed
            else:
                # Fallback: estimate from time and power (42.5W CPU + 6W RAM = 48.5W)
                result["energy_kwh"] = total_time * 0.0485 / 3600  # W * s = J, J/3600000 = kWh
        except:
            # Fallback: estimate from time and power
            result["energy_kwh"] = total_time * 0.0485 / 3600
        
        # Calculate efficiency
        if result["energy_kwh"] > 0:
            result["tokens_per_joule"] = num_tokens / (result["energy_kwh"] * 3600000)
        else:
            result["tokens_per_joule"] = 0
        
        print(f"Latency: {result['ms_per_token']:.2f} ms/token")
        print(f"Throughput: {result['tokens_per_second']:.2f} tok/s")
        print(f"Energy: {result['energy_kwh']:.6f} kWh")
        print(f"Efficiency: {result['tokens_per_joule']:.1f} tok/J")
        
        result["status"] = "success"
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)
    
    return result

def main():
    # Models to benchmark - focused subset for energy comparison
    models = [
        # Qwen3.5 FP16
        ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "fp16"),
        ("Qwen/Qwen3.5-2B", "Qwen3.5-2B", "fp16"),
        ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", "fp16"),
        # Qwen3.5 INT4
        ("mlx-community/Qwen3.5-0.8B-4bit", "Qwen3.5-0.8B", "int4"),
        ("mlx-community/Qwen3.5-2B-4bit", "Qwen3.5-2B", "int4"),
        ("mlx-community/Qwen3.5-4B-4bit", "Qwen3.5-4B", "int4"),
        # Qwen3.5 INT8
        ("mlx-community/Qwen3.5-0.8B-8bit", "Qwen3.5-0.8B", "int8"),
        ("mlx-community/Qwen3.5-2B-8bit", "Qwen3.5-2B", "int8"),
        ("mlx-community/Qwen3.5-4B-8bit", "Qwen3.5-4B", "int8"),
        # Phi-3 Mini
        ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-Mini", "fp16"),
        ("mlx-community/Phi-3-mini-4k-instruct-4bit", "Phi-3-Mini", "int4"),
        ("mlx-community/Phi-3-mini-4k-instruct-8bit", "Phi-3-Mini", "int8"),
    ]
    
    all_results = []
    
    for model_path, name, quant in models:
        result = run_benchmark_with_energy(model_path, name, quant)
        all_results.append(result)
        
        fname = f"{name.replace('-', '_')}_{quant}_energy.json"
        with open(RESULTS_DIR / fname, "w") as f:
            json.dump(result, f, indent=2)
    
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Model':<15} {'Quant':<8} {'tok/s':<10} {'ms/tok':<10} {'kWh':<12} {'tok/J':<10}")
    print("-" * 70)
    for r in all_results:
        if r.get("status") == "success":
            print(f"{r['model']:<15} {r['quant_method']:<8} {r['tokens_per_second']:<10.2f} {r['ms_per_token']:<10.2f} {r['energy_kwh']:<12.6f} {r['tokens_per_joule']:<10.1f}")
    
    print(f"\nResults saved to: {RESULTS_DIR.absolute()}")

if __name__ == "__main__":
    main()
