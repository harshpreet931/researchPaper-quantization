#!/usr/bin/env python3
"""
llama.cpp Efficiency Benchmarking Script
=========================================

Benchmarks GGUF models using llama-bench binary for pure efficiency metrics:
- Latency (ms/token)
- Throughput (tokens/s)
- Memory usage
- Load time

NO accuracy tests - pure efficiency metrics only.

Usage:
    python benchmark_llamacpp.py --model /path/to/model.gguf
    python benchmark_llamacpp.py --models-dir /path/to/gguf/models/
    python benchmark_llamacpp.py --model model.gguf --prompt-sizes 128,512 --gen-sizes 32,128
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Results output directory
RESULTS_DIR = Path(__file__).parent / "results_llamacpp"
RESULTS_DIR.mkdir(exist_ok=True)

# Default benchmark parameters
DEFAULT_PROMPT_SIZES = [128, 512, 1024]
DEFAULT_GEN_SIZES = [32, 128]
DEFAULT_REPETITIONS = 3
DEFAULT_THREADS = None  # Auto-detect

# Quantization levels commonly used in GGUF models
QUANTIZATION_LEVELS = {
    "Q4_0": "4-bit, no scaling",
    "Q4_1": "4-bit, with scaling",
    "Q4_K_M": "4-bit K-quantized medium",
    "Q4_K_S": "4-bit K-quantized small",
    "Q5_0": "5-bit, no scaling",
    "Q5_1": "5-bit, with scaling",
    "Q5_K_M": "5-bit K-quantized medium",
    "Q5_K_S": "5-bit K-quantized small",
    "Q6_K": "6-bit K-quantized",
    "Q8_0": "8-bit, no scaling",
    "F16": "16-bit float",
    "F32": "32-bit float",
}


def find_llama_binaries() -> Dict[str, Optional[str]]:
    """Find llama.cpp binaries in PATH."""
    binaries = {}
    for name in ["llama-bench", "llama-cli", "llama-quantize", "llama-server"]:
        path = shutil.which(name)
        binaries[name] = path
    return binaries


def detect_quantization_from_filename(filename: str) -> Optional[str]:
    """Detect quantization level from GGUF filename."""
    filename_upper = filename.upper()
    for quant in QUANTIZATION_LEVELS.keys():
        if quant in filename_upper:
            return quant
    return None


def get_model_info(model_path: Path) -> Dict[str, Any]:
    """Get basic model file info."""
    info = {
        "path": str(model_path),
        "filename": model_path.name,
        "size_bytes": 0,
        "size_gb": 0,
        "quantization": None,
    }
    
    if model_path.exists():
        info["size_bytes"] = model_path.stat().st_size
        info["size_gb"] = info["size_bytes"] / (1024**3)
        info["quantization"] = detect_quantization_from_filename(model_path.name)
    
    return info


def run_llama_bench(
    model_path: Path,
    prompt_sizes: List[int],
    gen_sizes: List[int],
    repetitions: int = 3,
    threads: Optional[int] = None,
    batch_size: int = 2048,
    ubatch_size: int = 512,
    n_gpu_layers: int = 0,  # CPU-only by default
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run llama-bench and parse results.
    
    Returns dict with benchmark results.
    """
    llama_bench = shutil.which("llama-bench")
    if not llama_bench:
        return {"error": "llama-bench binary not found in PATH"}
    
    # Build command
    cmd = [
        llama_bench,
        "-m", str(model_path),
        "-o", "json",
        "-r", str(repetitions),
        "-p", ",".join(map(str, prompt_sizes)),
        "-n", ",".join(map(str, gen_sizes)),
        "-b", str(batch_size),
        "-ub", str(ubatch_size),
        "-ngl", str(n_gpu_layers),
    ]
    
    if threads:
        cmd.extend(["-t", str(threads)])
    
    if verbose:
        cmd.append("-v")
    
    result = {
        "model_path": str(model_path),
        "command": " ".join(cmd),
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "prompt_sizes": prompt_sizes,
            "gen_sizes": gen_sizes,
            "repetitions": repetitions,
            "threads": threads,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "n_gpu_layers": n_gpu_layers,
        },
        "results": [],
        "status": "pending",
    }
    
    try:
        start_time = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        elapsed = time.time() - start_time
        
        result["elapsed_seconds"] = elapsed
        result["return_code"] = proc.returncode
        
        if proc.returncode != 0:
            result["status"] = "failed"
            result["error"] = proc.stderr
            result["stdout"] = proc.stdout
            return result
        
        # Parse JSON output from llama-bench
        # llama-bench outputs a JSON array of results
        stdout = proc.stdout.strip()
        if stdout:
            try:
                # Find the JSON array in output
                json_start = stdout.find("[")
                json_end = stdout.rfind("]") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = stdout[json_start:json_end]
                    bench_results = json.loads(json_str)
                    result["results"] = bench_results
                    result["status"] = "success"
                else:
                    result["status"] = "parse_error"
                    result["error"] = "No JSON array found in output"
                    result["stdout"] = stdout
            except json.JSONDecodeError as e:
                result["status"] = "parse_error"
                result["error"] = f"JSON decode error: {e}"
                result["stdout"] = stdout
        else:
            result["status"] = "no_output"
            result["error"] = "No output from llama-bench"
            
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Benchmark timed out after 1 hour"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def extract_efficiency_metrics(bench_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key efficiency metrics from llama-bench results.
    
    Returns standardized metrics across different test configurations.
    """
    metrics = {
        "model_path": bench_result.get("model_path"),
        "timestamp": bench_result.get("timestamp"),
        "status": bench_result.get("status"),
    }
    
    if bench_result.get("status") != "success":
        metrics["error"] = bench_result.get("error")
        return metrics
    
    results = bench_result.get("results", [])
    if not results:
        metrics["error"] = "No benchmark results"
        return metrics
    
    # Aggregate metrics across different configurations
    prompt_metrics = {}  # keyed by prompt size
    gen_metrics = {}  # keyed by gen size
    
    for r in results:
        # Each result has: n_prompt, n_gen, n_batch, time_total, etc.
        n_prompt = r.get("n_prompt", 0)
        n_gen = r.get("n_gen", 0)
        
        # Prompt processing (prefill) metrics
        if n_prompt > 0 and n_gen == 0:
            # Pure prompt processing test
            pp_speed = r.get("prompt_processing_speed", 0)  # tokens/s
            pp_time = r.get("prompt_processing_time", 0)  # ms
            if n_prompt not in prompt_metrics:
                prompt_metrics[n_prompt] = []
            prompt_metrics[n_prompt].append({
                "tokens_per_second": pp_speed,
                "total_time_ms": pp_time,
                "ms_per_token": pp_time / n_prompt if n_prompt > 0 else 0,
            })
        
        # Token generation metrics
        if n_gen > 0:
            tg_speed = r.get("token_generation_speed", 0)  # tokens/s
            tg_time = r.get("token_generation_time", 0)  # ms
            key = f"p{n_prompt}_g{n_gen}"
            if key not in gen_metrics:
                gen_metrics[key] = []
            gen_metrics[key].append({
                "tokens_per_second": tg_speed,
                "total_time_ms": tg_time,
                "ms_per_token": tg_time / n_gen if n_gen > 0 else 0,
                "prompt_size": n_prompt,
                "gen_size": n_gen,
            })
    
    # Aggregate and compute averages
    metrics["prompt_processing"] = {}
    for size, vals in prompt_metrics.items():
        if vals:
            avg_tps = sum(v["tokens_per_second"] for v in vals) / len(vals)
            avg_mspt = sum(v["ms_per_token"] for v in vals) / len(vals)
            metrics["prompt_processing"][size] = {
                "avg_tokens_per_second": avg_tps,
                "avg_ms_per_token": avg_mspt,
            }
    
    metrics["token_generation"] = {}
    for key, vals in gen_metrics.items():
        if vals:
            avg_tps = sum(v["tokens_per_second"] for v in vals) / len(vals)
            avg_mspt = sum(v["ms_per_token"] for v in vals) / len(vals)
            metrics["token_generation"][key] = {
                "avg_tokens_per_second": avg_tps,
                "avg_ms_per_token": avg_mspt,
                "prompt_size": vals[0]["prompt_size"],
                "gen_size": vals[0]["gen_size"],
            }
    
    # Summary metrics for common use case (512 prompt, 128 gen)
    summary_key = "p512_g128"
    if summary_key in metrics["token_generation"]:
        metrics["summary"] = metrics["token_generation"][summary_key]
    elif metrics["token_generation"]:
        # Use first available
        first_key = list(metrics["token_generation"].keys())[0]
        metrics["summary"] = metrics["token_generation"][first_key]
    
    return metrics


def measure_memory_usage(model_path: Path) -> Dict[str, Any]:
    """
    Estimate memory usage using llama-cli with a minimal run.
    
    This provides a rough estimate of memory requirements.
    """
    llama_cli = shutil.which("llama-cli")
    if not llama_cli:
        return {"error": "llama-cli not found"}
    
    # Run a minimal inference to measure memory
    cmd = [
        llama_cli,
        "-m", str(model_path),
        "-n", "1",  # Generate 1 token
        "-p", "Hello",  # Minimal prompt
        "-ngl", "0",  # CPU only
        "--no-display-prompt",
        "-c", "512",  # Small context
    ]
    
    result = {
        "method": "llama-cli estimation",
        "model_path": str(model_path),
    }
    
    try:
        # Get model size from file
        model_info = get_model_info(model_path)
        result["model_size_gb"] = model_info["size_gb"]
        
        # Run minimal inference
        start_time = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.time() - start_time
        
        result["load_and_inference_time"] = elapsed
        result["success"] = proc.returncode == 0
        
        # Parse memory info from stderr if available
        if proc.stderr:
            result["stderr_sample"] = proc.stderr[:500]
            
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def run_full_benchmark(
    model_path: Path,
    prompt_sizes: List[int] = None,
    gen_sizes: List[int] = None,
    repetitions: int = DEFAULT_REPETITIONS,
    threads: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run full benchmark suite for a single model.
    """
    prompt_sizes = prompt_sizes or DEFAULT_PROMPT_SIZES
    gen_sizes = gen_sizes or DEFAULT_GEN_SIZES
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {model_path.name}")
    print(f"{'='*60}")
    
    # Get model info
    model_info = get_model_info(model_path)
    print(f"Size: {model_info['size_gb']:.2f} GB")
    print(f"Quantization: {model_info['quantization'] or 'Unknown'}")
    
    result = {
        "model_info": model_info,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Check if model exists
    if not model_path.exists():
        result["status"] = "failed"
        result["error"] = f"Model not found: {model_path}"
        print(f"ERROR: {result['error']}")
        return result
    
    # Run llama-bench
    print(f"\nRunning llama-bench...")
    print(f"  Prompt sizes: {prompt_sizes}")
    print(f"  Gen sizes: {gen_sizes}")
    print(f"  Repetitions: {repetitions}")
    
    bench_result = run_llama_bench(
        model_path=model_path,
        prompt_sizes=prompt_sizes,
        gen_sizes=gen_sizes,
        repetitions=repetitions,
        threads=threads,
        verbose=verbose,
    )
    
    result["benchmark"] = bench_result
    
    # Extract efficiency metrics
    metrics = extract_efficiency_metrics(bench_result)
    result["metrics"] = metrics
    
    # Print summary
    if metrics.get("status") == "success":
        print(f"\n--- Results ---")
        if "summary" in metrics:
            s = metrics["summary"]
            print(f"Throughput: {s['avg_tokens_per_second']:.2f} tok/s")
            print(f"Latency: {s['avg_ms_per_token']:.2f} ms/token")
        
        # Print detailed results
        if metrics.get("token_generation"):
            print(f"\nToken Generation:")
            for key, val in metrics["token_generation"].items():
                print(f"  {key}: {val['avg_tokens_per_second']:.2f} tok/s, {val['avg_ms_per_token']:.2f} ms/tok")
        
        if metrics.get("prompt_processing"):
            print(f"\nPrompt Processing:")
            for size, val in metrics["prompt_processing"].items():
                print(f"  {size} tokens: {val['avg_tokens_per_second']:.2f} tok/s")
        
        result["status"] = "success"
    else:
        print(f"ERROR: {metrics.get('error', 'Unknown error')}")
        result["status"] = "failed"
    
    return result


def find_gguf_models(models_dir: Path) -> List[Path]:
    """Find all GGUF models in a directory."""
    models = []
    if models_dir.is_dir():
        for f in models_dir.rglob("*.gguf"):
            models.append(f)
    return sorted(models)


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a summary table of all benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<30} {'Quant':<10} {'Size(GB)':<10} {'tok/s':<10} {'ms/tok':<10} {'Status':<10}")
    print("-" * 80)
    
    for r in results:
        model_info = r.get("model_info", {})
        metrics = r.get("metrics", {})
        
        name = model_info.get("filename", "N/A")[:28]
        quant = model_info.get("quantization") or "N/A"
        size = f"{model_info.get('size_gb', 0):.2f}"
        
        summary = metrics.get("summary", {})
        tps = f"{summary.get('avg_tokens_per_second', 0):.1f}"
        mspt = f"{summary.get('avg_ms_per_token', 0):.1f}"
        status = r.get("status", "unknown")
        
        print(f"{name:<30} {quant:<10} {size:<10} {tps:<10} {mspt:<10} {status:<10}")
    
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GGUF models using llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark a single model
  python benchmark_llamacpp.py --model /path/to/model-q4_k_m.gguf
  
  # Benchmark all GGUF models in a directory
  python benchmark_llamacpp.py --models-dir /path/to/models/
  
  # Custom benchmark parameters
  python benchmark_llamacpp.py --model model.gguf --prompt-sizes 256,512 --gen-sizes 64,128
  
  # Use specific number of threads
  python benchmark_llamacpp.py --model model.gguf --threads 8
        """,
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Path to a single GGUF model file",
    )
    
    parser.add_argument(
        "--models-dir", "-d",
        type=str,
        help="Directory containing GGUF models to benchmark",
    )
    
    parser.add_argument(
        "--prompt-sizes", "-p",
        type=str,
        default=",".join(map(str, DEFAULT_PROMPT_SIZES)),
        help=f"Comma-separated prompt sizes (default: {','.join(map(str, DEFAULT_PROMPT_SIZES))})",
    )
    
    parser.add_argument(
        "--gen-sizes", "-n",
        type=str,
        default=",".join(map(str, DEFAULT_GEN_SIZES)),
        help=f"Comma-separated generation sizes (default: {','.join(map(str, DEFAULT_GEN_SIZES))})",
    )
    
    parser.add_argument(
        "--repetitions", "-r",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=f"Number of repetitions per test (default: {DEFAULT_REPETITIONS})",
    )
    
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=None,
        help="Number of threads (default: auto)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: results_llamacpp/benchmark_<timestamp>.json)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    parser.add_argument(
        "--check-binaries",
        action="store_true",
        help="Check for llama.cpp binaries and exit",
    )
    
    args = parser.parse_args()
    
    # Check binaries
    if args.check_binaries:
        print("Checking llama.cpp binaries...")
        binaries = find_llama_binaries()
        for name, path in binaries.items():
            status = path if path else "NOT FOUND"
            print(f"  {name}: {status}")
        return 0 if all(binaries.values()) else 1
    
    # Parse sizes
    prompt_sizes = [int(x) for x in args.prompt_sizes.split(",")]
    gen_sizes = [int(x) for x in args.gen_sizes.split(",")]
    
    # Collect models to benchmark
    models = []
    
    if args.model:
        model_path = Path(args.model)
        models.append(model_path)
    
    if args.models_dir:
        models_dir = Path(args.models_dir)
        found_models = find_gguf_models(models_dir)
        if found_models:
            print(f"Found {len(found_models)} GGUF models in {models_dir}")
            models.extend(found_models)
    
    if not models:
        parser.error("No models specified. Use --model or --models-dir")
    
    # Print header
    print("=" * 70)
    print("LLAMA.CPP EFFICIENCY BENCHMARK")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to benchmark: {len(models)}")
    
    # Check binaries
    binaries = find_llama_binaries()
    if not binaries.get("llama-bench"):
        print("ERROR: llama-bench not found in PATH")
        return 1
    print(f"llama-bench: {binaries['llama-bench']}")
    print("=" * 70)
    
    # Run benchmarks
    all_results = []
    
    for model_path in models:
        result = run_full_benchmark(
            model_path=model_path,
            prompt_sizes=prompt_sizes,
            gen_sizes=gen_sizes,
            repetitions=args.repetitions,
            threads=args.threads,
            verbose=args.verbose,
        )
        all_results.append(result)
        
        # Save individual result
        model_name = model_path.stem.replace("-", "_").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"{model_name}_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Result saved to: {result_file}")
    
    # Save combined results
    output_file = args.output
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"all_results_{timestamp}.json"
    else:
        output_file = Path(output_file)
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print_summary_table(all_results)
    print(f"All results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
