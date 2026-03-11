#!/usr/bin/env python3
"""Benchmark Ollama models for efficiency metrics.

Measures latency, throughput, and energy consumption for Ollama models.
NO accuracy benchmarks - pure efficiency metrics only.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("experiments/results_ollama")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Check CodeCarbon availability
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("Warning: CodeCarbon not available. Energy tracking disabled.")


def check_ollama_available():
    """Check if Ollama is installed and accessible."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def list_ollama_models():
    """List available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if parts:
                    models.append(parts[0])
            return models
        return []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def run_ollama_inference(model_name: str, prompt: str, timeout: int = 120):
    """Run inference using Ollama API with thinking disabled.
    
    Returns:
        tuple: (output_text, latency_seconds, tokens_generated) or (None, None, None) on error
    """
    import urllib.request
    import json
    
    # Use Ollama API with think: false for Qwen models
    is_qwen = "qwen" in model_name.lower()
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 128  # Limit output tokens for benchmark
        }
    }
    
    # Disable thinking for Qwen3.5 models
    if is_qwen:
        payload["think"] = False
    
    try:
        start_time = time.perf_counter()
        
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        latency = time.perf_counter() - start_time
        
        if 'error' in result:
            print(f"  API Error: {result['error']}")
            return None, None, None
        
        output_text = result.get('response', '').strip()
        # Get actual token count from API response
        tokens = result.get('eval_count', len(output_text) / 4)
        
        return output_text, latency, tokens
        
    except urllib.error.URLError as e:
        print(f"  Connection error: {e}")
        return None, None, None
    except Exception as e:
        print(f"  Error during inference: {e}")
        return None, None, None


    """Run inference using Ollama CLI.
    
    Returns:
        tuple: (output_text, latency_seconds, tokens_generated) or (None, None, None) on error
    """
    try:
        start_time = time.perf_counter()
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        latency = time.perf_counter() - start_time
        
        if result.returncode != 0:
            return None, None, None
        
        output_text = result.stdout.strip()
        # Estimate tokens (rough approximation: ~4 chars per token)
        tokens = len(output_text) / 4
        
        return output_text, latency, tokens
        
    except subprocess.TimeoutExpired:
        return None, None, None
    except Exception as e:
        print(f"  Error during inference: {e}")
        return None, None, None


def benchmark_model(
    model_name: str,
    prompt: str = "Write a short paragraph about artificial intelligence and its impact on society.",
    max_tokens: int = 64,
    num_iterations: int = 5,
    timeout: int = 120,
    warmup: bool = True
):
    """Benchmark a single Ollama model.
    
    Args:
        model_name: Name of the Ollama model (e.g., "qwen3:4b")
        prompt: Prompt to use for benchmarking
        max_tokens: Target number of tokens to generate
        num_iterations: Number of benchmark iterations
        timeout: Timeout for each inference call in seconds
        warmup: Whether to run a warmup iteration
        
    Returns:
        dict: Benchmark results
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {model_name}")
    print(f"{'='*60}")
    
    result = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "num_iterations": num_iterations,
        "prompt_length_chars": len(prompt),
    }
    
    # Initialize energy tracker if available
    tracker = None
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"ollama_benchmark_{model_name.replace(':', '_')}",
            output_dir=str(RESULTS_DIR),
            save_to_file=False,
            log_level="error"
        )
    
    try:
        # Check if model exists
        available_models = list_ollama_models()
        if model_name not in available_models:
            print(f"  Model '{model_name}' not found in Ollama.")
            print(f"  Available models: {available_models}")
            result["status"] = "model_not_found"
            result["available_models"] = available_models
            return result
        
        # Warmup run
        if warmup:
            print("  Running warmup iteration...")
            _, _, _ = run_ollama_inference(model_name, prompt, timeout)
        
        # Start energy tracking
        if tracker:
            tracker.start()
        
        # Benchmark iterations
        latencies = []
        total_tokens = 0
        successful_runs = 0
        
        print(f"  Running {num_iterations} benchmark iterations...")
        
        for i in range(num_iterations):
            output, latency, tokens = run_ollama_inference(model_name, prompt, timeout)
            
            if output is not None:
                latencies.append(latency)
                total_tokens += tokens
                successful_runs += 1
                print(f"    Iteration {i+1}: {latency:.3f}s, ~{tokens:.0f} tokens")
            else:
                print(f"    Iteration {i+1}: FAILED")
        
        # Stop energy tracking
        energy_consumed = 0.0
        emissions = 0.0
        if tracker:
            emissions = tracker.stop()
            energy_consumed = tracker.final_emissions_data.energy_consumed
        
        if successful_runs == 0:
            result["status"] = "all_iterations_failed"
            return result
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = total_tokens / successful_runs
        
        result.update({
            "avg_latency_seconds": round(avg_latency, 6),
            "ms_per_token": round((avg_latency / avg_tokens) * 1000, 2) if avg_tokens > 0 else 0,
            "tokens_per_second": round(avg_tokens / avg_latency, 2) if avg_latency > 0 else 0,
            "avg_tokens_generated": round(avg_tokens, 1),
            "successful_iterations": successful_runs,
            "total_iterations": num_iterations,
            "energy_consumed_kwh": round(energy_consumed, 6) if energy_consumed else 0,
            "emissions_kg": round(emissions, 6) if emissions else 0,
            "status": "success"
        })
        
        # Additional latency statistics
        if len(latencies) > 1:
            min_lat = min(latencies)
            max_lat = max(latencies)
            std_lat = (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5
            result["min_latency_seconds"] = round(min_lat, 6)
            result["max_latency_seconds"] = round(max_lat, 6)
            result["std_latency_seconds"] = round(std_lat, 6)
        
        print(f"\n  Results:")
        print(f"    Avg latency: {result['avg_latency_seconds']:.3f}s")
        print(f"    Avg tokens: {result['avg_tokens_generated']:.1f}")
        print(f"    ms/token: {result['ms_per_token']:.2f}")
        print(f"    Throughput: {result['tokens_per_second']:.2f} tok/s")
        if energy_consumed:
            print(f"    Energy: {result['energy_consumed_kwh']:.6f} kWh")
            print(f"    Emissions: {result['emissions_kg']:.6f} kg CO2")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)
        
        if tracker:
            tracker.stop()
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama models for efficiency metrics"
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Model names to benchmark (e.g., qwen3:4b llama3:8b). If not specified, uses default list."
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=120,
        help="Timeout per inference in seconds (default: 120)"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="Custom prompt for benchmarking"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup iteration"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available Ollama models and exit"
    )
    
    args = parser.parse_args()
    
    # Check Ollama availability
    if not check_ollama_available():
        print("ERROR: Ollama is not installed or not accessible.")
        print("Please install Ollama from https://ollama.ai")
        sys.exit(1)
    
    # List models mode
    if args.list:
        models = list_ollama_models()
        print("Available Ollama models:")
        for m in models:
            print(f"  - {m}")
        sys.exit(0)
    
    # Default models to benchmark
    default_models = [
        "qwen3.5:0.8b",
        "qwen3.5:2b",
        "qwen3.5:4b",
        "phi3:instruct",
    ]
    
    models_to_benchmark = args.models if args.models else default_models
    
    print(f"{'='*60}")
    print("OLLAMA EFFICIENCY BENCHMARK")
    print(f"{'='*60}")
    print(f"Models: {models_to_benchmark}")
    print(f"Iterations: {args.iterations}")
    print(f"Timeout: {args.timeout}s")
    print(f"CodeCarbon: {'Available' if CODECARBON_AVAILABLE else 'Not available'}")
    print(f"Results directory: {RESULTS_DIR.absolute()}")
    
    # Default prompt - short and constrained for fast benchmarking
    prompt = args.prompt or (
        "Complete this sentence: Machine learning is"
    )
    
    all_results = []
    
    for model_name in models_to_benchmark:
        result = benchmark_model(
            model_name=model_name,
            prompt=prompt,
            num_iterations=args.iterations,
            timeout=args.timeout,
            warmup=not args.no_warmup
        )
        all_results.append(result)
        
        # Save individual result
        safe_name = model_name.replace(":", "_").replace("/", "_")
        result_file = RESULTS_DIR / f"{safe_name}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {result_file}")
    
    # Save combined results
    all_results_file = RESULTS_DIR / "all_results.json"
    with open(all_results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'ms/tok':<12} {'tok/s':<12} {'Energy(kWh)':<14} {'Status':<10}")
    print("-" * 70)
    
    for r in all_results:
        status = r.get("status", "unknown")
        if status == "success":
            print(f"{r['model']:<20} {r['ms_per_token']:<12.2f} {r['tokens_per_second']:<12.2f} {r['energy_consumed_kwh']:<14.6f} {status:<10}")
        else:
            print(f"{r['model']:<20} {'N/A':<12} {'N/A':<12} {'N/A':<14} {status:<10}")
    
    print(f"\nResults saved to: {RESULTS_DIR.absolute()}")
    print(f"Combined results: {all_results_file}")


if __name__ == "__main__":
    main()
