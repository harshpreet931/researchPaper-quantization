#!/usr/bin/env python3
"""Quick benchmark for Ollama models with energy tracking."""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

# Check CodeCarbon availability
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("Warning: CodeCarbon not available. Energy tracking disabled.")

RESULTS_DIR = Path("results_ollama")
RESULTS_DIR.mkdir(exist_ok=True)

models = ["qwen3:4b", "phi3:mini", "ministral-3:3b"]
prompt = "Reply with only: OK"
iterations = 5

results = []

for model in models:
    print(f"\n=== Benchmarking {model} ===")
    latencies = []
    
    # Initialize energy tracker
    tracker = None
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"ollama_benchmark_{model.replace(':', '_')}",
            output_dir=str(RESULTS_DIR),
            save_to_file=False,
            log_level="error"
        )
    
    # Warmup
    print("  Warmup...")
    try:
        subprocess.run(["ollama", "run", model, prompt], capture_output=True, timeout=30)
    except Exception as e:
        print(f"  Warmup failed: {e}")
    
    # Start energy tracking
    if tracker:
        tracker.start()
    
    for i in range(iterations):
        try:
            start = time.perf_counter()
            result = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True, timeout=30)
            latency = time.perf_counter() - start
            latencies.append(latency)
            output = result.stdout.strip()
            print(f"  Iter {i+1}: {latency:.3f}s - {output[:30]}")
        except Exception as e:
            print(f"  Iter {i+1}: FAILED - {e}")
    
    # Stop energy tracking
    energy_consumed = 0.0
    emissions = 0.0
    if tracker:
        try:
            emissions = tracker.stop()
            energy_consumed = tracker.final_emissions_data.energy_consumed
        except Exception as e:
            print(f"  Energy tracking error: {e}")
    
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        result = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "avg_latency_seconds": round(avg_lat, 4),
            "min_latency_seconds": round(min(latencies), 4),
            "max_latency_seconds": round(max(latencies), 4),
            "successful_iterations": len(latencies),
            "total_iterations": iterations,
            "energy_consumed_kwh": round(energy_consumed, 6) if energy_consumed else 0,
            "emissions_kg": round(emissions, 6) if emissions else 0,
            "status": "success"
        }
    else:
        result = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "successful_iterations": 0,
            "total_iterations": iterations,
            "energy_consumed_kwh": 0,
            "emissions_kg": 0
        }
    
    results.append(result)
    
    # Save individual
    safe_name = model.replace(":", "_")
    with open(RESULTS_DIR / f"{safe_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {RESULTS_DIR / safe_name}.json")
    if energy_consumed:
        print(f"  Energy: {energy_consumed:.6f} kWh, Emissions: {emissions:.6f} kg CO2")

# Save all
with open(RESULTS_DIR / "all_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== SUMMARY ===")
print(f"{'Model':<20} {'Avg Latency':<15} {'Energy (kWh)':<15} {'Status':<10}")
print("-" * 60)
for r in results:
    if r["status"] == "success":
        print(f"{r['model']:<20} {r['avg_latency_seconds']:.3f}s{'':<9} {r['energy_consumed_kwh']:<15.6f} {r['status']:<10}")
    else:
        print(f"{r['model']:<20} {'N/A':<15} {'N/A':<15} {r['status']:<10}")

print(f"\nResults saved to: {RESULTS_DIR.absolute()}")
