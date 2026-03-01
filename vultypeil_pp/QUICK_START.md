# VulTypeIL++ Quick Start Guide

## Prerequisites

1. **Data Preparation**: Ensure you have the incremental task data in CSV format:
   ```
   ../incremental_tasks_csv/
   ├── task1_train.csv
   ├── task1_test.csv
   ├── task1_valid.csv
   ├── task2_train.csv
   ├── ...
   └── task5_valid.csv
   ```

2. **Environment**: Python 3.8+ with dependencies installed:
   ```bash
   pip install -r ../requirements.txt
   ```

## Running Experiments

### Option 1: Run Single Ablation (Recommended for Testing)

Start with the simplest ablation to verify setup:

```bash
cd vultypeil_pp
python run.py --config configs/3_vultypeilpp_er.yaml
```

This runs VulTypeIL++ with random experience replay (fastest to run).

### Option 2: Run All Ablations

```bash
python run_all.py
```

This will run all 6 ablations sequentially. Expected time: 10-20 hours depending on hardware.

### Option 3: Run Specific Ablations

Run only ablations 3, 4, and 5:
```bash
python run_all.py --ablations 3 4 5
```

Skip ablations 1 and 2 (non-scalable baselines):
```bash
python run_all.py --skip 1 2
```

## Understanding the Output

After running an experiment, you'll find:

```
outputs/<method_name>/
├── checkpoints/          # Model checkpoints
├── results/             # Predictions (pred.csv, gold.csv)
└── metrics/             # Performance metrics
    ├── <method>_acc_matrix.csv      # Accuracy matrix
    ├── <method>_f1_matrix.csv       # F1 matrix
    ├── <method>_summary.csv         # Summary metrics
    └── <method>_forgetting.csv      # Forgetting per task
```

### Key Metrics to Check

1. **Average Forgetting**: Lower is better (less catastrophic forgetting)
2. **Task 1 Final Accuracy**: Performance on first task at the end
3. **Final Average Accuracy**: Overall performance across all tasks

## Recommended Workflow

### Step 1: Quick Test (30 min - 1 hour)
```bash
# Test with reduced epochs to verify everything works
# Edit config file: num_epochs: 10 (instead of 100)
python run.py --config configs/3_vultypeilpp_er.yaml
```

### Step 2: Run Scalable Methods (6-8 hours)
```bash
# Run ablations 3-6 (all scalable methods)
python run_all.py --ablations 3 4 5 6
```

### Step 3: Run Baselines (4-6 hours)
```bash
# Run ablations 1-2 (original methods for comparison)
python run_all.py --ablations 1 2
```

### Step 4: Analyze Results
```bash
# Check the summary
cat outputs/experiment_summary.txt

# Compare accuracy matrices
python -c "
import pandas as pd
for i in range(1, 7):
    method = ['fixed200', 'ratio_allold', 'er', 'mcss', 'gcr', 'mcss_cons'][i-1]
    try:
        df = pd.read_csv(f'outputs/{i}_{method}/metrics/*_summary.csv')
        print(f'Ablation {i}: Avg Forg = {df[\"avg_forgetting\"].values[0]:.4f}')
    except:
        pass
"
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```yaml
# Edit config file
training:
  batch_size: 8  # or 4
```

**Solution 2**: Reduce buffer size
```yaml
buffer:
  size: 1000  # instead of 2000
```

### Issue: Training Too Slow

**Solution 1**: Reduce epochs
```yaml
training:
  num_epochs: 50  # instead of 100
```

**Solution 2**: Use smaller buffer for MCSS/GCR
```yaml
buffer:
  size: 1000  # instead of 2000
```

### Issue: Data Not Found

Make sure data paths are correct:
```yaml
data:
  data_dir: "/admin/dataset"  # or "../incremental_tasks_csv"
```

If data is in Excel format, convert to CSV first:
```bash
cd ..
python quick_excel_to_csv.py
```

## Expected Timeline

| Ablation | Method | Expected Time | GPU Memory |
|----------|--------|---------------|------------|
| 1 | Fixed 200 | 2-3 hours | ~8GB |
| 2 | Ratio All-Old | 3-4 hours | ~10GB |
| 3 | ER | 2-3 hours | ~8GB |
| 4 | MCSS | 3-4 hours | ~8GB |
| 5 | GCR | 3-4 hours | ~8GB |
| 6 | MCSS+Cons | 3-4 hours | ~8GB |

Total: ~15-20 hours for all ablations

## Customization

### Change Buffer Size
```yaml
buffer:
  size: 5000  # Try: 500, 1000, 2000, 5000
```

### Change Replay Ratio
```yaml
replay:
  replay_ratio: 0.3  # Try: 0.1, 0.2, 0.3, 0.5
```

### Change EWC Lambda
```yaml
loss:
  ewc_lambda: 0.5  # Try: 0.1, 0.2, 0.4, 0.5
```

### Add More Tasks
```yaml
data:
  num_tasks: 10  # If you have more tasks
```

## Next Steps

1. **Analyze Results**: Compare forgetting metrics across ablations
2. **Visualize**: Create plots from accuracy matrices
3. **Tune Hyperparameters**: Adjust buffer size, replay ratio, EWC lambda
4. **Extend**: Add new selection strategies or consolidation methods

## Getting Help

- Check `README.md` for detailed documentation
- Review config files in `configs/` for parameter descriptions
- Examine `run.py` for implementation details
- Check output logs for error messages

## Citation

If you use this code in your research, please cite the original VulTypeIL paper and mention VulTypeIL++.
