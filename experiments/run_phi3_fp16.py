#!/usr/bin/env python3
"""One-off: benchmark Phi-3 Mini FP16 and FP16 energy using the correct repo."""

import json
import time
from datetime import datetime
from pathlib import Path

# ── latency benchmark ──────────────────────────────────────────────────────────
def run_latency():
    from mlx_lm import load, generate
    from mlx.utils import tree_flatten

    model_path = "microsoft/Phi-3-mini-4k-instruct"
    result = {
        "model": "Phi-3-Mini",
        "quant_method": "fp16",
        "model_path": model_path,
        "enable_thinking": False,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        print("Loading model…")
        load_start = time.time()
        model, tokenizer = load(model_path)
        result["load_time_seconds"] = time.time() - load_start

        params = tree_flatten(model.parameters())
        total_params = sum(p.size for _, p in params)
        total_size   = sum(p.size * p.itemsize for _, p in params)
        result["parameters"]    = total_params
        result["model_size_gb"] = total_size / (1024 ** 3)
        result["bits_per_weight"] = total_size * 8 / total_params
        print(f"Size: {result['model_size_gb']:.2f} GB  BPW: {result['bits_per_weight']:.2f}")

        prompt = "Explain the concept of machine learning in simple terms."
        max_tokens = 32

        # warmup
        _ = generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=False)

        latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=False)
            latencies.append(time.perf_counter() - t0)

        avg = sum(latencies) / len(latencies)
        result["total_latency_seconds"] = avg
        result["ms_per_token"]       = (avg / max_tokens) * 1000
        result["tokens_per_second"]  = max_tokens / avg
        print(f"Latency: {result['ms_per_token']:.2f} ms/tok  Throughput: {result['tokens_per_second']:.2f} tok/s")

        # quick MMLU sample
        qs = [
            ("What is the time complexity of binary search?",
             "A) O(n) B) O(log n) C) O(n^2) D) O(1)", "B"),
            ("What is the SI unit of electric current?",
             "A) Volt B) Ohm C) Ampere D) Watt", "C"),
            ("What is the derivative of x^2?",
             "A) x B) 2x C) 2 D) x^2", "B"),
            ("What organelle is responsible for photosynthesis?",
             "A) Mitochondria B) Ribosome C) Chloroplast D) Nucleus", "C"),
            ("What is the chemical symbol for gold?",
             "A) Go B) Au C) Ag D) Gd", "B"),
        ]
        correct = sum(
            1 for q, choices, ans in qs
            if ans in generate(model, tokenizer,
                               f"Question: {q}\nChoices: {choices}\nAnswer with just the letter:\nAnswer:",
                               max_tokens=5, verbose=False).upper()
        )
        result["accuracy"] = correct / len(qs) * 100
        result["correct"]  = correct
        result["total"]    = len(qs)
        result["status"]   = "success"
        print(f"MMLU accuracy: {result['accuracy']:.1f}%")

    except Exception as e:
        import traceback; traceback.print_exc()
        result["status"] = "failed"
        result["error"]  = str(e)

    out = Path("results_gptq_awq") / "Phi_3_Mini_fp16.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out}")
    return result


# ── energy benchmark ───────────────────────────────────────────────────────────
def run_energy():
    from mlx_lm import load, generate
    from mlx.utils import tree_flatten
    from codecarbon import EmissionsTracker

    model_path = "microsoft/Phi-3-mini-4k-instruct"
    num_tokens = 100
    result = {
        "model": "Phi-3-Mini",
        "quant_method": "fp16",
        "model_path": model_path,
        "num_tokens": num_tokens,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        print("\nLoading model for energy run…")
        t0 = time.time()
        model, tokenizer = load(model_path)
        result["load_time_seconds"] = time.time() - t0

        params = tree_flatten(model.parameters())
        total_params = sum(p.size for _, p in params)
        total_size   = sum(p.size * p.itemsize for _, p in params)
        result["parameters"]    = total_params
        result["model_size_gb"] = total_size / (1024 ** 3)

        prompt = "Explain the concept of machine learning in simple terms."

        # warmup
        _ = generate(model, tokenizer, prompt, max_tokens=num_tokens, verbose=False)

        tracker = EmissionsTracker(
            project_name="phi3_mini_fp16_energy",
            output_file="emissions.csv",
            log_level="error",
            save_to_file=True,
        )
        tracker.start()
        t_start = time.perf_counter()
        _ = generate(model, tokenizer, prompt, max_tokens=num_tokens, verbose=False)
        elapsed = time.perf_counter() - t_start
        emissions = tracker.stop()

        result["total_time_seconds"] = elapsed
        result["ms_per_token"]       = (elapsed / num_tokens) * 1000
        result["tokens_per_second"]  = num_tokens / elapsed
        result["emissions_kg"]       = emissions
        result["energy_kwh"]         = tracker._total_energy.kWh  # type: ignore[attr-defined]
        energy_j = result["energy_kwh"] * 3_600_000
        result["tokens_per_joule"]   = num_tokens / energy_j
        result["status"] = "success"

        print(f"Energy run: {result['tokens_per_second']:.2f} tok/s  "
              f"{result['energy_kwh']:.6f} kWh  "
              f"{result['tokens_per_joule']:.4f} tok/J")

    except Exception as e:
        import traceback; traceback.print_exc()
        result["status"] = "failed"
        result["error"]  = str(e)

    out = Path("results_energy") / "Phi_3_Mini_fp16_energy.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out}")
    return result


if __name__ == "__main__":
    r1 = run_latency()
    if r1.get("status") == "success":
        run_energy()
    else:
        print("Latency run failed — skipping energy run.")
