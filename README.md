# Low Power Edge Inference for Green Computing

An evaluation of Post-Training Quantization and Unstructured Pruning for edge AI deployment.

## Authors

**Harshpreet Singh** · harshpreet393.be22@chitkara.edu.in  
**Sameer Kumar** · sameer776.be22@chitkara.edu.in  
*Chitkara University, India*

## Overview

This project compares four PTQ techniques—INT8 uniform quantization, GPTQ, AWQ, and INT4 weight-only—on Qwen3-4B and Phi-2 models to identify optimal accuracy-energy trade-offs for edge deployments.

## Project Structure

```
├── paper.tex          # Main paper
├── abstract.tex       # Abstract
├── references.bib     # Bibliography
├── experiments/       # Benchmark scripts & results
│   ├── benchmark_*.py # Benchmarking scripts
│   └── results_*/     # Experiment results
└── docs/              # Documentation
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

## License

MIT