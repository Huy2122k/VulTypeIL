# VulTypeIL++: Scalable Replay for Vulnerability Type Identification

This directory contains the implementation of VulTypeIL++ with scalable replay strategies, extending the original VulTypeIL framework.

## Overview

VulTypeIL++ addresses the scalability limitations of the original VulTypeIL by introducing:
- **Fixed-size replay buffer** (no dataset scaling with tasks)
- **Mixed batch iterator** (constant compute per step)
- **Advanced selection strategies** (MCSS, GCR-approx)
- **Consolidation phase** (phasic replay for reduced forgetting)

## Directory Structure

```
vultypeil_pp/
├── configs/                    # Configuration files for 6 ablations
│   ├── 1_vultypeil_fixed200.yaml
│   ├── 2_vultypeil_ratio_allold.yaml
│   ├── 3_vultypeilpp_er.yaml
│   ├── 4_vultypeilpp_mcss.yaml
│   ├── 5_vultypeilpp_gcr_approx.yaml
│   └── 6_vultypeilpp_mcss_cons.yaml
├── outputs/                    # Experiment outputs (created automatically)
├── data_utils.py              # Data loading utilities
├── replay_buffer.py           # Replay buffer implementation
├── mixed_dataloader.py        # Mixed batch iterator
├── selectors.py               # Selection strategies (Mahalanobis, MCSS, GCR)
├── trainer.py                 # Training functions (2-phase + consolidation)
├── metrics.py                 # Continual learning metrics
├── run.py                     # Main experiment runner
├── run_all.py                 # Script to run all ablations
└── README.md                  # This file
```

## Ablation Configurations

### 1. VulTypeIL Original (Fixed 200, Mahalanobis)
- **Config**: `configs/1_vultypeil_fixed200.yaml`
- **Description**: Baseline from original paper
- **Replay**: Fixed 200 samples selected by Mahalanobis distance
- **Scalability**: ❌ (merges all previous data for selection)

### 2. VulTypeIL + Replay Ratio All-Old (20%)
- **Config**: `configs/2_vultypeil_ratio_allold.yaml`
- **Description**: Non-scalable baseline with high quality
- **Replay**: 20% of all previous data (random selection)
- **Scalability**: ❌ (dataset grows with tasks)

### 3. VulTypeIL++ (Random ER)
- **Config**: `configs/3_vultypeilpp_er.yaml`
- **Description**: Scalable replay with reservoir sampling
- **Replay**: Fixed buffer (M=2000), reservoir sampling
- **Scalability**: ✅ (constant buffer size, mixed batches)

### 4. VulTypeIL++-MCSS
- **Config**: `configs/4_vultypeilpp_mcss.yaml`
- **Description**: Scalable replay with multi-criteria selection
- **Replay**: Fixed buffer (M=2000), MCSS selection
- **Selection**: Prototype distance + loss + diversity
- **Scalability**: ✅

### 5. VulTypeIL++-GCR (Approx)
- **Config**: `configs/5_vultypeilpp_gcr_approx.yaml`
- **Description**: Scalable replay with gradient coreset
- **Replay**: Fixed buffer (M=2000), GCR-approx selection
- **Selection**: Gradient matching in output space
- **Scalability**: ✅

### 6. VulTypeIL++-MCSS + Consolidation
- **Config**: `configs/6_vultypeilpp_mcss_cons.yaml`
- **Description**: MCSS + phasic consolidation
- **Replay**: Fixed buffer (M=2000), MCSS + consolidation phase
- **Consolidation**: 500 steps on buffer after each task
- **Scalability**: ✅

## Installation

1. Install dependencies (same as VulTypeIL):
```bash
pip install -r ../requirements.txt
```

2. Ensure data is prepared:
```bash
# Data should be in: ../incremental_tasks_csv/
# - task1_train.csv, task1_test.csv, task1_valid.csv
# - task2_train.csv, task2_test.csv, task2_valid.csv
# - ... (task3, task4, task5)
```

## Usage

### Run Single Experiment

```bash
python run.py --config configs/1_vultypeil_fixed200.yaml
```

### Run All Ablations

```bash
python run_all.py
```

This will run all 6 configurations sequentially and save results to `outputs/`.

### Run Specific Ablations

```bash
python run_all.py --ablations 3 4 5 6
```

## Output Structure

Each experiment creates:
```
outputs/<method_name>/
├── checkpoints/
│   ├── task_1_phase1_best.ckpt
│   ├── task_1_phase2_best.ckpt
│   ├── task_2_phase1_best.ckpt
│   └── ...
├── results/
│   ├── task1_after_task1.pred.csv
│   ├── task1_after_task1.gold.csv
│   └── ...
└── metrics/
    ├── <method>_acc_matrix.csv
    ├── <method>_f1_matrix.csv
    ├── <method>_summary.csv
    └── <method>_forgetting.csv
```

## Key Metrics

The experiments track:
- **Accuracy Matrix**: Acc[i][j] = accuracy on task j after training task i
- **Average Forgetting**: Mean forgetting across all tasks
- **Task 1 Final Accuracy**: Performance on first task at the end
- **Backward Transfer (BWT)**: Average performance change on old tasks
- **Forward Transfer (FWT)**: Zero-shot performance on new tasks

## Configuration Parameters

### Key Parameters to Tune

```yaml
# Buffer size (for scalable methods)
buffer:
  size: 2000  # Try: 500, 1000, 2000, 5000

# Replay ratio (for batch mixing)
replay:
  replay_ratio: 0.2  # Try: 0.1, 0.2, 0.3

# EWC regularization
loss:
  ewc_lambda: 0.4  # Try: 0.1, 0.2, 0.4, 0.5

# Consolidation steps
consolidation:
  steps: 500  # Try: 200, 500, 1000
```

## Comparison Matrix

| Method | Buffer Size | Selection | Scalable | Expected Quality |
|--------|-------------|-----------|----------|------------------|
| (1) Fixed 200 | - | Mahalanobis | ❌ | Baseline |
| (2) Ratio All-Old | Growing | Random | ❌ | High (non-scalable) |
| (3) ER | 2000 | Reservoir | ✅ | Moderate |
| (4) MCSS | 2000 | Multi-criteria | ✅ | High |
| (5) GCR | 2000 | Gradient | ✅ | High |
| (6) MCSS+Cons | 2000 | Multi-criteria | ✅ | Highest |

## Expected Results

Based on continual learning literature:
- **(1) Fixed 200**: High forgetting, especially on Task 1
- **(2) Ratio All-Old**: Low forgetting but not scalable
- **(3) ER**: Moderate forgetting, scalable
- **(4) MCSS**: Lower forgetting than ER, scalable
- **(5) GCR**: Similar to MCSS, potentially better on tail classes
- **(6) MCSS+Cons**: Lowest forgetting, scalable

## Citation

If you use this code, please cite:

```bibtex
@article{vultypeil2024,
  title={Learning never stops: Improving software vulnerability type identification via incremental learning},
  author={[Original Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Notes

- All methods use the same 2-phase training (Phase 1: Focal+LS, Phase 2: Focal+LS+EWC)
- Consolidation phase (ablation 6) adds a 3rd phase after Phase 2
- Buffer size M=2000 is chosen to balance quality and memory
- Replay ratio r=0.2 means 20% of each batch comes from buffer
- Early stopping with patience=5 is used for all methods

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config (try 8 or 4)
- Reduce `buffer.size` (try 1000 or 500)

### Slow Training
- Reduce `num_epochs` (try 50 instead of 100)
- Use smaller buffer for MCSS/GCR (try 1000)

### Poor Performance
- Increase `buffer.size` (try 5000)
- Increase `replay_ratio` (try 0.3)
- Tune `ewc_lambda` (try 0.2 or 0.5)
