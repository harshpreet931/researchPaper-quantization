# Low Power Edge Inference for Green Computing: An Evaluation of Post-Training Quantization on Apple Silicon

## Authors

**Harshpreet Singh** · harshpreet393.be22@chitkara.edu.in  
**Sameer Kumar** · sameer776.be22@chitkara.edu.in  
*Chitkara University, India*

## Overview

This project evaluates Post-Training Quantization (PTQ) on Apple Silicon for edge AI deployment. We test:

- **Models**: Qwen3.5 series (0.8B, 2B, 4B parameters) and Phi-3 Mini (3.82B parameters)
- **Frameworks**: Apple MLX (FP16, INT8, INT4) and Ollama (Q4_K_M)
- **Hardware**: Apple M4 Pro with 24GB unified memory

## Key Results

### Quantization Performance (MLX INT4 vs FP16)
- **Speedup**: 2.0x to 3.8x faster inference with INT4
- **Memory**: 3.5x to 3.6x memory reduction
- **Energy**: 2.2x to 3.9x energy savings within MLX

### Framework Comparison (MLX INT4 vs Ollama Q4_K_M)
- **Throughput**: MLX achieves 6.8x to 13.4x higher throughput
- **Energy Savings**: 6.7x to 19.0x with MLX INT4 vs Ollama

### Peak Performance
| Model | MLX INT4 Throughput | Memory |
|-------|---------------------|--------|
| Qwen3.5-0.8B | 213.11 tok/s | 0.39 GB |
| Qwen3.5-2B | 131.86 tok/s | 0.99 GB |
| Qwen3.5-4B | 71.21 tok/s | 2.20 GB |
| Phi-3 Mini | 88.40 tok/s | 2.00 GB |

All models fit within 2.5GB with MLX INT4, enabling deployment on devices with 8GB RAM.

## Project Structure

```
├── paper_minified.tex  # Main paper
├── references.bib      # Bibliography
├── experiments/        # Benchmark scripts & results
│   ├── benchmark_*.py  # Benchmarking scripts
│   └── results_*/      # Experiment results
└── docs/               # Documentation
```

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run benchmarks
python experiments/run_experiments.py
```

## Repository

Code and experimental data: [github.com/harshpreet931/researchPaper-quantization](https://github.com/harshpreet931/researchPaper-quantization)

## License

MIT
