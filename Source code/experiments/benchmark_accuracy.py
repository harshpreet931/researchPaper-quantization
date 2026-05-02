#!/usr/bin/env python3
"""MMLU accuracy benchmark for quantized MLX models.

Uses logit-based scoring (standard academic method) instead of text generation.
For each question, compares the model's next-token log-probabilities for A/B/C/D
and picks the highest. This guarantees zero unparseable responses.

Evaluates 4 models (Qwen3.5-0.8B/2B/4B, Phi-3 Mini) at 3 quantization levels
(FP16, INT8, INT4) on a 100-question subset of the MMLU benchmark.

Usage:
    pip install datasets mlx-lm
    python benchmark_accuracy.py
"""

import gc
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results_accuracy")
RESULTS_DIR.mkdir(exist_ok=True)

# 12 model configurations: (hf_path, display_name, quant_method)
MODELS = [
    # Qwen3.5 FP16
    ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "fp16"),
    ("Qwen/Qwen3.5-2B", "Qwen3.5-2B", "fp16"),
    ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", "fp16"),
    # Qwen3.5 INT8
    ("mlx-community/Qwen3.5-0.8B-8bit", "Qwen3.5-0.8B", "int8"),
    ("mlx-community/Qwen3.5-2B-8bit", "Qwen3.5-2B", "int8"),
    ("mlx-community/Qwen3.5-4B-8bit", "Qwen3.5-4B", "int8"),
    # Qwen3.5 INT4
    ("mlx-community/Qwen3.5-0.8B-4bit", "Qwen3.5-0.8B", "int4"),
    ("mlx-community/Qwen3.5-2B-4bit", "Qwen3.5-2B", "int4"),
    ("mlx-community/Qwen3.5-4B-4bit", "Qwen3.5-4B", "int4"),
    # Phi-3 Mini
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-Mini", "fp16"),
    ("mlx-community/Phi-3-mini-4k-instruct-8bit", "Phi-3-Mini", "int8"),
    ("mlx-community/Phi-3-mini-4k-instruct-4bit", "Phi-3-Mini", "int4"),
]

# 10 MMLU subjects across 4 categories
SUBJECTS = {
    # STEM
    "abstract_algebra": "STEM",
    "college_physics": "STEM",
    "computer_security": "STEM",
    # Humanities
    "philosophy": "Humanities",
    "world_religions": "Humanities",
    # Social Sciences
    "high_school_psychology": "Social Sciences",
    "sociology": "Social Sciences",
    # Other
    "clinical_knowledge": "Other",
    "management": "Other",
    "marketing": "Other",
}

QUESTIONS_PER_SUBJECT = 10
SEED = 42
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTERS = ["A", "B", "C", "D"]


def load_mmlu_questions():
    """Load MMLU questions from HuggingFace datasets."""
    from datasets import load_dataset

    random.seed(SEED)
    questions = []

    for subject, category in SUBJECTS.items():
        print(f"  Loading {subject}...", end=" ")
        ds = load_dataset("cais/mmlu", subject, split="test")

        indices = list(range(len(ds)))
        sample_indices = random.sample(indices, min(QUESTIONS_PER_SUBJECT, len(indices)))

        for idx in sample_indices:
            row = ds[idx]
            questions.append({
                "subject": subject,
                "category": category,
                "question": row["question"],
                "choices": row["choices"],
                "correct_answer": ANSWER_MAP[row["answer"]],
            })
        print(f"{len(sample_indices)} questions")

    random.shuffle(questions)  # mix subjects to avoid ordering bias
    return questions


def make_raw_prompt(q):
    """Build the raw question text for MMLU evaluation."""
    choices = q["choices"]
    return (
        "The following is a multiple choice question. "
        "Answer with just the letter (A, B, C, or D).\n\n"
        f"Question: {q['question']}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        "Answer:"
    )


def get_prompt_tokens(q, model_name, tokenizer):
    """Tokenize an MMLU prompt, using chat template where available."""
    raw = make_raw_prompt(q)

    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": raw}]
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        # Disable thinking for Qwen models to avoid <think> token overhead
        if "Qwen" in model_name:
            kwargs["enable_thinking"] = False
        try:
            return tokenizer.apply_chat_template(msgs, **kwargs)
        except TypeError:
            # Fallback if enable_thinking kwarg not supported
            kwargs.pop("enable_thinking", None)
            try:
                return tokenizer.apply_chat_template(msgs, **kwargs)
            except Exception:
                pass

    return tokenizer.encode(raw)


def build_answer_token_map(tokenizer):
    """Pre-compute token IDs for each answer letter (A/B/C/D).

    Checks multiple representations (with/without space prefix) and returns
    a dict mapping each letter to a set of candidate token IDs.
    """
    token_map = {}
    for letter in LETTERS:
        candidates = set()
        for variant in [letter, f" {letter}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for tid in ids:
                candidates.add(tid)
        token_map[letter] = candidates
    return token_map


def score_by_logits(model, tokens, answer_token_map):
    """Run a single forward pass and return the best A/B/C/D answer.

    Returns (predicted_letter, logit_dict) where logit_dict maps each
    letter to its best logit value. Guaranteed to always return a letter.
    """
    import mlx.core as mx

    input_ids = mx.array(tokens)[None]  # (1, seq_len)
    logits = model(input_ids)
    mx.eval(logits)
    next_logits = logits[0, -1, :]  # (vocab_size,)

    answer_logits = {}
    for letter in LETTERS:
        candidates = answer_token_map[letter]
        if candidates:
            answer_logits[letter] = max(next_logits[tid].item() for tid in candidates)
        else:
            answer_logits[letter] = float("-inf")

    best_letter = max(answer_logits, key=answer_logits.get)
    return best_letter, answer_logits


def run_accuracy_benchmark(model_path, model_name, quant_method, questions):
    """Run MMLU accuracy benchmark on a single model config using logit scoring."""
    from mlx_lm import load

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {model_name} ({quant_method})")
    print(f"{'='*60}")

    result = {
        "model": model_name,
        "quant_method": quant_method,
        "model_path": model_path,
        "method": "logit_scoring",
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
    }

    try:
        # Load model
        load_start = time.time()
        model, tokenizer = load(model_path)
        load_time = time.time() - load_start
        result["load_time_seconds"] = load_time

        # Model metadata
        from mlx.utils import tree_flatten
        params = tree_flatten(model.parameters())
        total_params = sum(p.size for _, p in params)
        total_size = sum(p.size * p.itemsize for _, p in params)
        result["parameters"] = total_params
        result["model_size_gb"] = total_size / (1024**3)

        print(f"Loaded in {load_time:.1f}s | {result['model_size_gb']:.2f} GB | {total_params:,} params")

        # Pre-compute answer token map (once per model)
        answer_token_map = build_answer_token_map(tokenizer)
        print(f"Answer tokens: { {l: sorted(ids) for l, ids in answer_token_map.items()} }")

        # Warmup forward pass
        import mlx.core as mx
        warmup_ids = mx.array(tokenizer.encode("Hello"))[None]
        _ = model(warmup_ids)
        mx.eval(_)

        # Evaluate
        correct = 0
        per_subject = {s: {"correct": 0, "total": 0} for s in SUBJECTS}
        per_category = {c: {"correct": 0, "total": 0} for c in set(SUBJECTS.values())}
        question_results = []

        for i, q in enumerate(questions):
            tokens = get_prompt_tokens(q, model_name, tokenizer)
            model_answer, logits = score_by_logits(model, tokens, answer_token_map)
            is_correct = model_answer == q["correct_answer"]

            if is_correct:
                correct += 1

            subj = q["subject"]
            cat = q["category"]
            per_subject[subj]["total"] += 1
            per_category[cat]["total"] += 1
            if is_correct:
                per_subject[subj]["correct"] += 1
                per_category[cat]["correct"] += 1

            # Round logits for JSON readability
            question_results.append({
                "subject": subj,
                "correct_answer": q["correct_answer"],
                "model_answer": model_answer,
                "is_correct": is_correct,
                "logits": {k: round(v, 4) for k, v in logits.items()},
            })

            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{len(questions)} | Running accuracy: {correct/(i+1)*100:.1f}%")

        # Compute accuracy percentages
        result["overall_accuracy"] = (correct / len(questions)) * 100
        result["correct"] = correct
        result["total"] = len(questions)
        result["unparseable"] = 0  # Logit scoring always produces an answer

        result["per_subject"] = {
            s: {**d, "accuracy": (d["correct"] / d["total"] * 100) if d["total"] > 0 else 0}
            for s, d in per_subject.items()
        }
        result["per_category"] = {
            c: {**d, "accuracy": (d["correct"] / d["total"] * 100) if d["total"] > 0 else 0}
            for c, d in per_category.items()
        }
        result["question_results"] = question_results
        result["status"] = "success"

        print(f"\n  Overall: {result['overall_accuracy']:.1f}% ({correct}/{len(questions)})")
        for cat in sorted(result["per_category"]):
            d = result["per_category"][cat]
            print(f"  {cat}: {d['accuracy']:.1f}%")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)

    # Free memory and delete cached model
    try:
        del model, tokenizer
    except NameError:
        pass
    gc.collect()
    delete_model_cache(model_path)

    return result


def delete_model_cache(model_path):
    """Delete the downloaded model from HuggingFace cache to free disk space."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    # HF cache uses format: models--org--model_name
    cache_name = "models--" + model_path.replace("/", "--")
    cache_dir = hf_cache / cache_name
    if cache_dir.exists():
        size_gb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / (1024**3)
        shutil.rmtree(cache_dir)
        print(f"  Deleted cache: {cache_name} ({size_gb:.2f} GB freed)")


def main():
    print("Loading MMLU questions...")
    questions = load_mmlu_questions()
    print(f"\nLoaded {len(questions)} questions across {len(SUBJECTS)} subjects\n")

    all_results = []

    for model_path, name, quant in MODELS:
        result = run_accuracy_benchmark(model_path, name, quant, questions)
        all_results.append(result)

        # Save individual result
        fname = f"{name.replace('-', '_').replace('.', '_')}_{quant}_accuracy.json"
        with open(RESULTS_DIR / fname, "w") as f:
            json.dump(result, f, indent=2)

    # Save combined results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    categories = sorted(set(SUBJECTS.values()))
    cat_short = {"STEM": "STEM%", "Humanities": "Human%", "Social Sciences": "Social%", "Other": "Other%"}

    print(f"\n{'='*90}")
    print("MMLU ACCURACY SUMMARY (logit-based scoring)")
    print(f"{'='*90}")
    header = f"{'Model':<15} {'Quant':<6} {'Overall%':<9}"
    for c in categories:
        header += f" {cat_short.get(c, c):<8}"
    print(header)
    print("-" * 90)

    for r in all_results:
        if r.get("status") != "success":
            print(f"{r['model']:<15} {r['quant_method']:<6} FAILED: {r.get('error', 'unknown')[:40]}")
            continue
        line = f"{r['model']:<15} {r['quant_method']:<6} {r['overall_accuracy']:<9.1f}"
        for c in categories:
            acc = r["per_category"].get(c, {}).get("accuracy", 0)
            line += f" {acc:<8.1f}"
        print(line)

    print(f"{'='*90}")
    print(f"\nResults saved to: {RESULTS_DIR.absolute()}")


if __name__ == "__main__":
    main()
