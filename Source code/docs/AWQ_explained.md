# AWQ: Activation-Aware Weight Quantization

## Overview

AWQ (Activation-Aware Weight Quantization) is a post-training quantization technique specifically designed for Large Language Models (LLMs). It was introduced in the paper *"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"* by Lin et al. (2023), which received the **Best Paper Award at MLSys 2024**.

## The Core Problem

Standard quantization methods treat all weights equally, but in neural networks:

- **Not all weights are equally important** for the output
- Quantizing important weights causes significant accuracy degradation
- Uniform quantization across all weights leads to suboptimal results

## Key Insight

AWQ's breakthrough observation: **protecting just 1% of salient weights can significantly reduce quantization error.**

The key question is: *How do we identify which weights are salient?*

## How AWQ Works

### 1. Identifying Salient Weights

AWQ identifies salient weights based on **activation magnitudes**, not weight magnitudes.

**Why activations?**
- Weights connected to high-activation channels have more impact on output
- A small weight connected to a high-activation feature can contribute significantly
- Looking at weights alone misses this relationship

Mathematically, the importance of weight $w_{ij}$ is determined by the activation $x_j$:

$$\text{Importance}(w_{ij}) \propto |x_j| \cdot |w_{ij}|$$

### 2. Per-Channel Scaling

Instead of using a single scale factor for all weights, AWQ applies **per-channel scaling**:

$$\hat{W} = \text{quant}(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1}$$

Where:
- $W$ is the weight matrix
- $s$ is a per-channel scaling factor
- $\text{quant}(\cdot)$ is the quantization function

**Intuition:** By scaling up salient channels before quantization, they occupy more quantization "bins" and thus lose less precision.

### 3. Automatic Scale Search

AWQ finds optimal scaling factors by minimizing the reconstruction error:

$$s^* = \argmin_s \mathcal{L}(W, s)$$

Where the loss is:

$$\mathcal{L} = \sum_{i,j} \left| X_{i,j} \cdot (w_{i,j} - \hat{w}_{i,j}) \right|$$

This is computed efficiently using a small calibration dataset (e.g., 512 samples from The Pile).

## Comparison with Other Methods

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Naive INT4** | Direct quantization | Simple | High accuracy loss |
| **GPTQ** | Hessian-based weight update | Good accuracy | Slow for large models |
| **AWQ** | Activation-aware scaling | Fast, accurate | Needs calibration data |
| **bitsandbytes** | Dynamic quantization | Easy to use | CUDA-only |

## AWQ vs GPTQ

| Aspect | AWQ | GPTQ |
|--------|-----|------|
| **Speed** | Faster (no Hessian computation) | Slower (requires inverse Hessian) |
| **Memory** | Lower overhead | Higher overhead during quantization |
| **Calibration** | Uses activations | Uses Hessian approximations |
| **Hardware** | Works on CPU/GPU | Primarily GPU-optimized |

## Implementation Example

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load model
model_path = "mistralai/Mistral-7B-v0.1"
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define quantization config
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Quantize (requires calibration data)
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval"  # Calibration dataset
)

# Save quantized model
model.save_quantized("mistral-7b-awq-4bit")
```

## Quantization Parameters

### w_bit (Weight Bits)
- **4-bit**: Best compression (72% memory reduction), slight accuracy loss
- **8-bit**: Moderate compression (47% memory reduction), minimal accuracy loss

### q_group_size (Quantization Group Size)
- **32**: More granular, higher quality
- **64**: Balance of quality and speed
- **128**: Faster, slightly lower quality (default)

### zero_point
- **True**: Uses asymmetric quantization (better for non-symmetric distributions)
- **False**: Uses symmetric quantization (simpler, slightly faster)

## When to Use AWQ

### Best For:
- Deploying LLMs on edge devices with limited memory
- Production inference where quantization time isn't critical
- Models where calibration data is available

### Not Ideal For:
- Quick experiments (calibration adds overhead)
- Models without good calibration data
- Very small models (overhead may not be worth it)

## Performance Impact

Based on original paper benchmarks (LLaMA-7B on NVIDIA GPU):

| Precision | Memory | Perplexity (WikiText2) |
|-----------|--------|------------------------|
| FP16 | 13.5 GB | 5.68 |
| INT8 | 7.0 GB | 5.72 |
| AWQ-INT4 | 3.5 GB | 5.78 |
| GPTQ-INT4 | 3.5 GB | 5.82 |
| Naive INT4 | 3.5 GB | 6.50+ |

**Key takeaway**: AWQ-INT4 achieves near-FP16 perplexity with 4× memory reduction.

## Current Limitations

1. **Calibration Required**: Needs representative data for optimal results
2. **Model-Specific**: May need tuning for different architectures
3. **Hardware Dependent**: Best performance on NVIDIA GPUs with specific kernels

## Resources

- **Paper**: [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- **Code**: [https://github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
- **Hugging Face Integration**: Available through `autoawq` package

## Quick Reference

```bash
# Install AWQ
pip install autoawq

# Quantize a model
python -m awq.entry --model_path mistral-7b \
    --w_bit 4 \
    --q_group_size 128 \
    --output_path mistral-7b-awq-4bit
```

---

*Last updated: 2026-03-12*
