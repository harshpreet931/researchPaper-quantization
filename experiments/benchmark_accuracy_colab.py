#!/usr/bin/env python3
"""Full MMLU accuracy benchmark for quantized models - Google Colab compatible.

Uses transformers + bitsandbytes instead of MLX. Works on NVIDIA GPUs in Colab.
Same logit-based scoring — one forward pass per question, picks the letter
(A/B/C/D) with the highest next-token log-probability.

Colab Setup:
    !pip install transformers datasets bitsandbytes accelerate

Usage:
    python benchmark_accuracy_colab.py

    # To resume a partial run:
    python benchmark_accuracy_colab.py --resume
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

RESULTS_DIR = Path("results_accuracy_colab")
RESULTS_DIR.mkdir(exist_ok=True)

# Model configurations for Colab (using HF model IDs)
# Note: We use standard HF models and apply quantization via bitsandbytes
MODELS = [
    # Qwen3.5 FP16
    ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "fp16"),
    ("Qwen/Qwen3.5-2B", "Qwen3.5-2B", "fp16"),
    ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", "fp16"),
    # Qwen3.5 INT8
    ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "int8"),
    ("Qwen/Qwen3.5-2B", "Qwen3.5-2B", "int8"),
    ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", "int8"),
    # Qwen3.5 INT4
    ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", "int4"),
    ("Qwen/Qwen3.5-2B", "Qwen3.5-2B", "int4"),
    ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", "int4"),
    # Phi-3 Mini
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-Mini", "fp16"),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-Mini", "int8"),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-Mini", "int4"),
]

# All 57 MMLU subjects with their categories
SUBJECTS = {
    # STEM
    "abstract_algebra": "STEM",
    "anatomy": "STEM",
    "astronomy": "STEM",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_mathematics": "STEM",
    "high_school_physics": "STEM",
    "high_school_statistics": "STEM",
    "machine_learning": "STEM",
    # Humanities
    "formal_logic": "Humanities",
    "high_school_european_history": "Humanities",
    "high_school_us_history": "Humanities",
    "high_school_world_history": "Humanities",
    "international_law": "Humanities",
    "jurisprudence": "Humanities",
    "logical_fallacies": "Humanities",
    "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities",
    "philosophy": "Humanities",
    "prehistory": "Humanities",
    "professional_law": "Humanities",
    "world_religions": "Humanities",
    # Social Sciences
    "econometrics": "Social Sciences",
    "global_facts": "Social Sciences",
    "high_school_geography": "Social Sciences",
    "high_school_government_and_politics": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences",
    "high_school_microeconomics": "Social Sciences",
    "high_school_psychology": "Social Sciences",
    "human_sexuality": "Social Sciences",
    "professional_psychology": "Social Sciences",
    "public_relations": "Social Sciences",
    "security_studies": "Social Sciences",
    "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    # Other
    "business_ethics": "Other",
    "clinical_knowledge": "Other",
    "college_medicine": "Other",
    "human_aging": "Other",
    "management": "Other",
    "marketing": "Other",
    "medical_genetics": "Other",
    "miscellaneous": "Other",
    "nutrition": "Other",
    "professional_accounting": "Other",
    "professional_medicine": "Other",
    "virology": "Other",
}

SEED = 42
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTERS = ["A", "B", "C", "D"]


def load_mmlu_questions():
    """Load all questions from all 57 MMLU subjects."""
    from datasets import load_dataset

    questions = []
    total = 0

    print(f"Loading all {len(SUBJECTS)} MMLU subjects...")
    for subject, category in SUBJECTS.items():
        ds = load_dataset("cais/mmlu", subject, split="test")
        n = len(ds)
        for i in range(n):
            row = ds[i]
            questions.append({
                "subject": subject,
                "category": category,
                "question": row["question"],
                "choices": row["choices"],
                "correct_answer": ANSWER_MAP[row["answer"]],
            })
        total += n
        print(f"  {subject}: {n} questions  (running total: {total})")

    print(f"\nTotal: {len(questions)} questions across {len(SUBJECTS)} subjects")
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
        if "Qwen" in model_name:
            kwargs["enable_thinking"] = False
        try:
            return tokenizer.apply_chat_template(msgs, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            try:
                return tokenizer.apply_chat_template(msgs, **kwargs)
            except Exception:
                pass

    return tokenizer.encode(raw)


def build_answer_token_map(tokenizer):
    """Pre-compute token IDs for each answer letter (A/B/C/D)."""
    token_map = {}
    for letter in LETTERS:
        candidates = set()
        for variant in [letter, f" {letter}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for tid in ids:
                candidates.add(tid)
        token_map[letter] = candidates
    return token_map


def score_by_logits(model, tokens, answer_token_map, device):
    """Run a single forward pass and return the best A/B/C/D answer."""
    input_ids = torch.tensor([tokens], device=device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # (vocab_size,)

    answer_logits = {}
    for letter in LETTERS:
        candidates = answer_token_map[letter]
        if candidates:
            answer_logits[letter] = max(logits[tid].item() for tid in candidates)
        else:
            answer_logits[letter] = float("-inf")

    best_letter = max(answer_logits, key=answer_logits.get)
    return best_letter, answer_logits


def load_model(model_path, quant_method, device):
    """Load model with specified quantization using transformers + bitsandbytes."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if quant_method == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif quant_method == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif quant_method == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown quantization method: {quant_method}")
    
    model.eval()
    return model, tokenizer


def get_model_size(model):
    """Calculate model size in GB."""
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    return total_size / (1024**3)


def result_filename(name, quant):
    return RESULTS_DIR / f"{name.replace('-', '_').replace('.', '_')}_{quant}_accuracy_colab.json"


def run_accuracy_benchmark(model_path, model_name, quant_method, questions, resume=False):
    """Run full MMLU accuracy benchmark on a single model config."""
    out_file = result_filename(model_name, quant_method)

    if resume and out_file.exists():
        print(f"\nSkipping {model_name} ({quant_method}) — result already exists.")
        with open(out_file) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {model_name} ({quant_method})  [{len(questions)} questions]")
    print(f"{'='*70}")

    result = {
        "model": model_name,
        "quant_method": quant_method,
        "model_path": model_path,
        "method": "logit_scoring_full_mmlu_colab",
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "num_subjects": len(SUBJECTS),
    }

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        load_start = time.time()
        model, tokenizer = load_model(model_path, quant_method, device)
        load_time = time.time() - load_start
        result["load_time_seconds"] = load_time

        model_size_gb = get_model_size(model)
        total_params = sum(p.numel() for p in model.parameters())
        result["parameters"] = total_params
        result["model_size_gb"] = model_size_gb

        print(f"Loaded in {load_time:.1f}s | {model_size_gb:.2f} GB | {total_params:,} params")

        answer_token_map = build_answer_token_map(tokenizer)

        # Warmup
        warmup_ids = torch.tensor([tokenizer.encode("Hello")], device=device)
        with torch.no_grad():
            _ = model(warmup_ids)

        correct = 0
        per_subject = {s: {"correct": 0, "total": 0} for s in SUBJECTS}
        per_category = {c: {"correct": 0, "total": 0} for c in set(SUBJECTS.values())}
        question_results = []

        bench_start = time.time()

        for i, q in enumerate(questions):
            tokens = get_prompt_tokens(q, model_name, tokenizer)
            model_answer, logits = score_by_logits(model, tokens, answer_token_map, device)
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

            question_results.append({
                "subject": subj,
                "correct_answer": q["correct_answer"],
                "model_answer": model_answer,
                "is_correct": is_correct,
                "logits": {k: round(v, 4) for k, v in logits.items()},
            })

            if (i + 1) % 500 == 0:
                elapsed = time.time() - bench_start
                rate = (i + 1) / elapsed
                remaining = (len(questions) - i - 1) / rate
                print(
                    f"  [{i+1}/{len(questions)}] "
                    f"Accuracy: {correct/(i+1)*100:.1f}%  "
                    f"Speed: {rate:.1f} q/s  "
                    f"ETA: {remaining/60:.0f} min"
                )

        bench_time = time.time() - bench_start
        result["benchmark_time_seconds"] = bench_time
        result["questions_per_second"] = len(questions) / bench_time

        result["overall_accuracy"] = (correct / len(questions)) * 100
        result["correct"] = correct
        result["total"] = len(questions)
        result["unparseable"] = 0

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
        print(f"  Time: {bench_time/60:.1f} min  ({result['questions_per_second']:.1f} q/s)")
        for cat in sorted(result["per_category"]):
            d = result["per_category"][cat]
            print(f"  {cat}: {d['accuracy']:.1f}%")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)

    # Save immediately so a partial run is never fully lost
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_file}")

    try:
        del model, tokenizer
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def print_summary(all_results):
    categories = ["STEM", "Humanities", "Social Sciences", "Other"]
    cat_short = {
        "STEM": "STEM%",
        "Humanities": "Human%",
        "Social Sciences": "Social%",
        "Other": "Other%",
    }

    print(f"\n{'='*95}")
    print("FULL MMLU ACCURACY SUMMARY (Colab, logit-based scoring, all 57 subjects)")
    print(f"{'='*95}")
    header = f"{'Model':<17} {'Quant':<6} {'N':>5} {'Overall%':<10}"
    for c in categories:
        header += f" {cat_short[c]:<8}"
    print(header)
    print("-" * 95)

    for r in all_results:
        if r.get("status") != "success":
            print(f"{r['model']:<17} {r['quant_method']:<6} FAILED: {r.get('error', 'unknown')[:40]}")
            continue
        line = (
            f"{r['model']:<17} {r['quant_method']:<6} "
            f"{r['total']:>5} {r['overall_accuracy']:<10.2f}"
        )
        for c in categories:
            acc = r["per_category"].get(c, {}).get("accuracy", 0)
            line += f" {acc:<8.2f}"
        print(line)
    print(f"{'='*95}")


def main():
    parser = argparse.ArgumentParser(description="Full MMLU benchmark for Colab (transformers + bitsandbytes)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip model configs that already have a result file",
    )
    args = parser.parse_args()

    questions = load_mmlu_questions()

    all_results = []
    for model_path, name, quant in MODELS:
        result = run_accuracy_benchmark(model_path, name, quant, questions, resume=args.resume)
        all_results.append(result)

    # Save combined results
    combined_path = RESULTS_DIR / "all_results_colab.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print_summary(all_results)
    print(f"\nAll results saved to: {RESULTS_DIR.absolute()}")


if __name__ == "__main__":
    main()
