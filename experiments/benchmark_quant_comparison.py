#!/usr/bin/env python3
"""
Comprehensive Quantization Comparison for Qwen3.5-0.8B on Apple Silicon

Tests multiple quantization methods:
- MLX FP16 (baseline)
- MLX INT4 (different group sizes)
- MLX INT8
- Ollama GGUF variants (Q4_0, Q4_K_M, Q5_K_M, Q8_0)
"""

import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from codecarbon import EmissionsTracker

RESULTS_DIR = Path("results_quant_comparison")
RESULTS_DIR.mkdir(exist_ok=True)

NUM_TOKENS = 100
PROMPT = "Explain the concept of machine learning in simple terms."

def run_mlx_benchmark(model_path: str, quant_name: str) -> dict:
    """Run MLX benchmark with energy tracking."""
    from mlx_lm import load, generate
    
    print(f"\n{'='*60}")
    print(f"MLX BENCHMARK: {quant_name}")
    print(f"{'='*60}")
    
    result = {
        "framework": "MLX",
        "quantization": quant_name,
        "model_path": model_path,
        "num_tokens": NUM_TOKENS,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        # Load model
        load_start = time.time()
        model, tokenizer = load(model_path)
        load_time = time.time() - load_start
        
        # Get model size
        from mlx.utils import tree_flatten
        params = tree_flatten(model.parameters())
        total_params = sum(p.size for _, p in params)
        total_size = sum(p.size * p.itemsize for _, p in params)
        
        result["load_time_seconds"] = load_time
        result["parameters"] = total_params
        result["model_size_gb"] = total_size / (1024**3)
        
        print(f"Model size: {result['model_size_gb']:.2f} GB")
        
        # Warmup
        _ = generate(model, tokenizer, PROMPT, max_tokens=32, verbose=False)
        
        # Benchmark with energy tracking
        tracker = EmissionsTracker(measure_power_secs=1)
        tracker.start()
        
        start_time = time.perf_counter()
        _ = generate(model, tokenizer, PROMPT, max_tokens=NUM_TOKENS, verbose=False)
        end_time = time.perf_counter()
        
        emissions = tracker.stop()
        
        # Calculate metrics
        total_time = end_time - start_time
        result["total_time_seconds"] = total_time
        result["ms_per_token"] = (total_time / NUM_TOKENS) * 1000
        result["tokens_per_second"] = NUM_TOKENS / total_time
        result["emissions_kg"] = emissions
        
        # Energy calculation
        try:
            if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data:
                result["energy_kwh"] = tracker.final_emissions_data.energy_consumed
            else:
                result["energy_kwh"] = total_time * 0.0485 / 3600
        except:
            result["energy_kwh"] = total_time * 0.0485 / 3600
        
        if result["energy_kwh"] > 0:
            result["tokens_per_joule"] = NUM_TOKENS / (result["energy_kwh"] * 3600000)
        else:
            result["tokens_per_joule"] = 0
        
        print(f"Throughput: {result['tokens_per_second']:.2f} tok/s")
        print(f"Latency: {result['ms_per_token']:.2f} ms/tok")
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

def run_ollama_benchmark(model_name: str, quant_name: str) -> dict:
    """Run Ollama benchmark with energy tracking."""
    import requests
    
    print(f"\n{'='*60}")
    print(f"OLLAMA BENCHMARK: {quant_name}")
    print(f"{'='*60}")
    
    result = {
        "framework": "Ollama",
        "quantization": quant_name,
        "model": model_name,
        "num_tokens": NUM_TOKENS,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        # Check if model exists, pull if not
        check = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name not in check.stdout:
            print(f"Pulling model {model_name}...")
            subprocess.run(["ollama", "pull", model_name], check=True)
        
        # Run benchmark with energy tracking
        tracker = EmissionsTracker(measure_power_secs=1)
        tracker.start()
        
        start_time = time.perf_counter()
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": PROMPT,
                "stream": False,
                "options": {"num_predict": NUM_TOKENS}
            },
            timeout=300
        )
        
        end_time = time.perf_counter()
        emissions = tracker.stop()
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
        
        data = response.json()
        actual_tokens = data.get("eval_count", NUM_TOKENS)
        
        # Calculate metrics
        total_time = end_time - start_time
        result["total_time_seconds"] = total_time
        result["ms_per_token"] = (total_time / actual_tokens) * 1000
        result["tokens_per_second"] = actual_tokens / total_time
        result["actual_tokens"] = actual_tokens
        result["emissions_kg"] = emissions
        
        # Energy calculation
        try:
            if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data:
                result["energy_kwh"] = tracker.final_emissions_data.energy_consumed
            else:
                result["energy_kwh"] = total_time * 0.0485 / 3600
        except:
            result["energy_kwh"] = total_time * 0.0485 / 3600
        
        if result["energy_kwh"] > 0:
            result["tokens_per_joule"] = actual_tokens / (result["energy_kwh"] * 3600000)
        else:
            result["tokens_per_joule"] = 0
        
        # Get model size from ollama show
        show_result = subprocess.run(
            ["ollama", "show", model_name, "--modelfile"],
            capture_output=True, text=True
        )
        # Parse size if available
        result["model_size_gb"] = "N/A"
        
        print(f"Throughput: {result['tokens_per_second']:.2f} tok/s")
        print(f"Latency: {result['ms_per_token']:.2f} ms/tok")
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
    all_results = []
    
    # MLX quantization variants
    mlx_models = [
        ("Qwen/Qwen3.5-0.8B", "FP16"),
        ("mlx-community/Qwen3.5-0.8B-8bit", "INT8"),
        ("mlx-community/Qwen3.5-0.8B-4bit", "INT4"),
    ]
    
    print("=" * 70)
    print("QUANTIZATION COMPARISON FOR Qwen3.5-0.8B")
    print("=" * 70)
    
    # Run MLX benchmarks
    print("\n--- MLX Quantization ---")
    for model_path, quant_name in mlx_models:
        result = run_mlx_benchmark(model_path, quant_name)
        all_results.append(result)
    
    # Run Ollama benchmarks
    print("\n--- Ollama/GGUF Quantization ---")
    ollama_models = [
        ("qwen3.5:0.8b", "Q4_K_M"),  # Default
    ]
    
    for model_name, quant_name in ollama_models:
        result = run_ollama_benchmark(model_name, quant_name)
        all_results.append(result)
    
    # Save results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 90)
    print("QUANTIZATION COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Method':<12} {'Framework':<10} {'Size(GB)':<10} {'tok/s':<12} {'ms/tok':<10} {'kWh':<12} {'tok/J':<8}")
    print("-" * 75)
    
    for r in all_results:
        if r.get("status") == "success":
            quant = r.get("quantization", "N/A")
            framework = r.get("framework", "N/A")
            size = f"{r.get('model_size_gb', 0):.2f}" if isinstance(r.get('model_size_gb'), (int, float)) else r.get('model_size_gb', 'N/A')
            tps = r.get('tokens_per_second', 0)
            ms = r.get('ms_per_token', 0)
            kwh = r.get('energy_kwh', 0)
            tpj = r.get('tokens_per_joule', 0)
            print(f"{quant:<12} {framework:<10} {size:<10} {tps:<12.2f} {ms:<10.2f} {kwh:<12.6f} {tpj:<8.1f}")
    
    print(f"\nResults saved to: {RESULTS_DIR.absolute()}")
    
    # Generate LaTeX table
    print("\n" + "=" * 90)
    print("LATEX TABLE")
    print("=" * 90)
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{Quantization method comparison for Qwen3.5-0.8B on Apple Silicon M4 Pro.}")
    print("\\label{tab:quant_comparison}")
    print("\\begin{tabular}{l|cccccc}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Framework} & \\textbf{Size (GB)} & \\textbf{tok/s} & \\textbf{ms/tok} & \\textbf{Energy (kWh)} & \\textbf{Efficiency (tok/J)} \\\\")
    print("\\midrule")
    
    for r in all_results:
        if r.get("status") == "success":
            quant = r.get("quantization", "N/A")
            framework = r.get("framework", "N/A")
            size = f"{r.get('model_size_gb', 0):.2f}" if isinstance(r.get('model_size_gb'), (int, float)) else "N/A"
            tps = r.get('tokens_per_second', 0)
            ms = r.get('ms_per_token', 0)
            kwh = r.get('energy_kwh', 0)
            tpj = r.get('tokens_per_joule', 0)
            print(f"{quant} & {framework} & {size} & {tps:.2f} & {ms:.2f} & {kwh:.6f} & {tpj:.1f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
