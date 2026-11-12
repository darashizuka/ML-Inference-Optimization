# ML Inference Optimization

Benchmarking inference optimization techniques across vision and NLP models, measuring the tradeoffs between model size, latency, and accuracy for on-device deployment.

## Models

| Model | Task | Dataset |
|-------|------|---------|
| MobileNetV3-Small | Image Classification | ImageNette (100 samples, 10 classes) |
| EfficientNet-V2-S | Image Classification | ImageNette (100 samples, 10 classes) |
| DistilBERT | Sentiment Classification | SST-2 (100 validation samples) |

## Optimization Techniques

- **FP32** -- Full-precision baseline
- **FP16** -- Half-precision floating point
- **INT8** -- Dynamic quantization via `torch.ao.quantization`
- **Core ML Export** -- Converted to `.mlpackage` (iOS16 target) with linear symmetric INT8 quantization via `coremltools`

## Results

| Model | Precision | Size (MB) | Latency (ms) | Accuracy | Acc Delta |
|-------|-----------|-----------|--------------|----------|-----------|
| MobileNetV3-Small | FP32 | 9.8 | 12.1 | 71.0% | Baseline |
| MobileNetV3-Small | FP16 | 4.9 | 130.8 | 71.0% | +0.0% |
| MobileNetV3-Small | INT8 | 5.2 | 11.8 | 69.0% | -2.0% |
| EfficientNet-V2-S | FP32 | 82.7 | 143.8 | 87.0% | Baseline |
| EfficientNet-V2-S | FP16 | 41.5 | 4930.3 | 87.0% | +0.0% |
| EfficientNet-V2-S | INT8 | 79.0 | 143.7 | 87.0% | +0.0% |
| DistilBERT (SST-2) | FP32 | 255.5 | 155.7 | 94.0% | Baseline |
| DistilBERT (SST-2) | INT8 | 132.3 | 113.7 | 93.0% | -1.0% |

> FP16 latency is higher on CPU due to lack of native FP16 support -- gains are realized on GPU/ANE hardware.

## Key Findings

- **EfficientNet-V2-S** is the safest candidate for aggressive optimization -- zero accuracy loss at INT8
- **MobileNetV3-Small** shows highest quantization sensitivity (-2.0% at INT8) due to narrow bottleneck layers in squeeze-excitation blocks
- **DistilBERT INT8** achieves 1.37x speedup and ~48% size reduction with only 1% accuracy drop
- **Core ML INT8** packages achieve ~50% compression vs FP32 across all models

## Setup

```bash
pip install torch torchvision transformers datasets coremltools pandas tabulate
```

Run `ML_inference.ipynb` in Google Colab or locally.

## Environment

Benchmarked on Google Colab (Intel Xeon CPU, NVIDIA T4 GPU). Latency measured as mean of 100 forward passes after 10 warmup runs.
