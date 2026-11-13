# Assignment 03 - Edge Computing & Intelligent Systems
## 23i-3022 Section A

## Overview
ResNet-18 model optimization on CIFAR-10 using quantization and pruning.

**Q1: Quantization** - PTQ (Post-Training Quantization) and QAT (Quantization-Aware Training)
**Q2: Pruning** - Unstructured magnitude-based pruning and structured channel pruning

## How to Run
1. Open `Assignment3.ipynb` on Kaggle
2. Set Accelerator to GPU T4 x2, enable Internet
3. Run All

## Models Produced
| # | Model | File |
|---|-------|------|
| 1 | FP32 Baseline | `models/resnet18_fp32.pth` |
| 2 | PTQ INT8 | `models/resnet18_ptq_int8.pth` |
| 3 | QAT INT8 | `models/resnet18_qat_int8.pth` |
| 4 | Unstructured Pruned (70%) | `models/resnet18_unstructured_pruned.pth` |
| 5 | Structured Pruned (30%) | `models/resnet18_structured_pruned.pth` |

## Results
- `results/benchmark_results.csv` - Accuracy, latency, size, sparsity
- `results/inference_summary.csv` - Per-model inference results
- `results/onnx_benchmark.csv` - ONNX export verification
- `results/predictions_*.csv` - Per-sample predictions

## Structure
```
├── Assignment3.ipynb          # Main notebook (all code)
├── README.md
├── report.pdf                 # Technical report
├── models/                    # Trained checkpoints
│   ├── resnet18_fp32.pth
│   ├── resnet18_ptq_int8.pth
│   ├── resnet18_qat_int8.pth
│   ├── resnet18_unstructured_pruned.pth
│   ├── resnet18_structured_pruned.pth
│   └── onnx/                  # ONNX exports
└── results/                   # CSV results
```
