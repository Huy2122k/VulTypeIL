# Discussion Experiments Guide

This document provides a comprehensive guide to the experiments conducted in the Discussion folder (discussion1 to discussion5). Each section describes the purpose of the experiments, expected results, and commands to run training and evaluation.

## Discussion1: Hard Prompt vs Soft Prompt

### Purpose
This discussion compares the effectiveness of hard prompts (ManualTemplate) and soft prompts (SoftTemplate) in prompt learning for continual learning on vulnerability type classification. Hard prompts use fixed text templates, while soft prompts learn continuous embeddings.

### Expected Results
- Performance metrics (accuracy, precision, recall, F1, MCC) for each task after training.
- Comparison of forgetting rates and stability across tasks.
- Soft prompts may show better adaptability but require more parameters.

### Commands
#### Training and Evaluation for Hard Prompt
```bash
python "Discussion/discussion1/hard prompt.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

#### Training and Evaluation for Soft Prompt
```bash
python "Discussion/discussion1/soft prompt.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

## Discussion2: Verbalizer Comparison (1 vs 5 Label Words)

### Purpose
This discussion investigates the impact of the number of label words in the verbalizer on classification performance. verbalizer1.py uses 1 label word per class, while verbalizer5.py uses 5 label words per class.

### Expected Results
- More label words may improve performance by providing richer semantic information.
- Trade-off between expressiveness and computational cost.

### Commands
#### Training and Evaluation with 1 Label Word
```bash
python "Discussion/discussion2/verbalizer1.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

#### Training and Evaluation with 5 Label Words
```bash
python "Discussion/discussion2/verbalizer5.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

## Discussion3: Focal Alpha Variations

### Purpose
This discussion explores different focal alpha values (w) in the Focal Loss for continual learning. Files test focal_alpha=0.1, 0.3, 0.7, 0.9.

### Expected Results
- Focal alpha controls the weight of hard examples. Lower values focus more on hard examples.
- Optimal alpha depends on data difficulty and task sequence.

### Commands
#### focal_alpha=0.1
```bash
python "Discussion/discussion3/w0.1.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

#### focal_alpha=0.3
```bash
python "Discussion/discussion3/w0.3.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

#### focal_alpha=0.7
```bash
python "Discussion/discussion3/w0.7.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

#### focal_alpha=0.9
```bash
python "Discussion/discussion3/w0.9.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

## Discussion4: EWC Lambda Variations

### Purpose
This discussion examines different lambda values for EWC regularization. Lambda controls the strength of the penalty on parameter changes. Files test lambda=0.1, 0.2, 0.3, 0.5.

### Expected Results
- Higher lambda values reduce forgetting but may hinder learning new tasks.
- Lower lambda allows better adaptation at the cost of stability.

### Commands
#### lambda=0.1
```bash
python "Discussion/discussion4/lamda0.1.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

#### lambda=0.2
```bash
python "Discussion/discussion4/lamda0.2.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

#### lambda=0.3
```bash
python "Discussion/discussion4/lamda0.3.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

#### lambda=0.5
```bash
python "Discussion/discussion4/lamda0.5.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

## Discussion5: Replay Size Variations

### Purpose
This discussion studies the effect of different replay buffer sizes (rs) in experience replay for continual learning. Files test rs=0.05, 0.1, 0.15, 0.25 (as fraction of dataset size for replay selection).

### Expected Results
- Larger replay sizes improve retention of old knowledge but increase memory usage.
- Smaller sizes are more efficient but may lead to higher forgetting.

### Commands
#### rs=0.05
```bash
python "Discussion/discussion5/rs0.05.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

#### rs=0.1
```bash
python "Discussion/discussion5/rs0.1.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

#### rs=0.15
```bash
python "Discussion/discussion5/rs0.15.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

#### rs=0.25
```bash
python "Discussion/discussion5/rs0.25.py" --data_dir incremental_tasks_csv --pretrained_model_path Salesforce/codet5-base --batch_size 16 --num_epochs 100 --lr 5e-5 --use_cuda
```

## Notes
- All scripts use incremental learning across 5 tasks (task1 to task5).
- Data is loaded from CSV files in the specified --data_dir.
- Models are saved as 'best.ckpt' locally.
- Ensure OpenPrompt and required dependencies are installed.
- Adjust parameters as needed for your environment.