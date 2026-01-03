# Hướng Dẫn Chạy VulExplainer_CodeBERT

## Tổng Quan
VulExplainer_CodeBERT là baseline sử dụng kiến trúc distillation: Teacher model (CNN-based) được train trước, sau đó distill kiến thức vào Student model (CodeBERT-based) để cải thiện hiệu suất phát hiện lỗ hổng bảo mật.

## Chuẩn Bị Data
Data đã được xử lý và hợp nhất từ incremental_tasks_csv:
- **Train**: 11,508 samples (`mydata/updated_train.csv`)
- **Validation**: 1,413 samples (`mydata/updated_val.csv`)
- **Test**: 1,314 samples (`mydata/updated_test.csv`)
- **CWE Label Map**: 157 classes (`mydata/cwe_label_map.pkl`)

Data có các cột: `func_before`, `cwe_ids`, `cwe_abstract_group`

## Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

Các thư viện chính: torch, transformers, sklearn, pandas, tqdm

## Kịch Bản Chạy

### Bước 1: Train Teacher Model (CNN-based)
Teacher model sử dụng TextCNN với CodeBERT embeddings.

**Lệnh chạy:**
```bash
cd RQ1/baselines/VulExplainer/VulExplainer_CodeBERT
python teacher_main.py \
    --output_dir=./saved_models \
    --model_name=cnnteacher.bin \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --train_data_file=../../../mydata/updated_train.csv \
    --eval_data_file=../../../mydata/updated_val.csv \
    --test_data_file=../../../mydata/updated_test.csv \
    --do_train \
    --do_test \
    --block_size 512 \
    --epochs 50 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 5e-3 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

**Thời gian dự kiến:** ~2-4 giờ (tùy GPU)
**Output:** Model lưu tại `./saved_models/checkpoint-best-acc/cnnteacher.bin`

### Bước 2: Train Student Model (CodeBERT với Distillation)
Student model học từ teacher qua knowledge distillation.

**Lệnh chạy (Soft Distillation):**
```bash
cd RQ1/baselines/VulExplainer/VulExplainer_CodeBERT
python student_codebert_main.py \
    --alpha 0.7 \
    --output_dir=./saved_models \
    --model_name=soft_distil_model_07.bin \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --train_data_file=../../../mydata/updated_train.csv \
    --eval_data_file=../../../mydata/updated_val.csv \
    --test_data_file=../../../mydata/updated_test.csv \
    --do_train \
    --do_test \
    --block_size 512 \
    --epochs 50 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

**Hyperparameters quan trọng:**
- `--alpha`: Trọng số distillation loss (0.7 = 70% distillation, 30% student loss)
- `--beta`: Beta cho inference (tự động chọn best từ eval)
- Batch size: 8 cho student (nhỏ hơn teacher)
- Learning rate: 2e-5 cho student (nhỏ hơn teacher)

**Thời gian dự kiến:** ~4-6 giờ
**Output:** Model lưu tại `./saved_models/checkpoint-best-acc/soft_distil_model_07.bin`

### Bước 3: Evaluation/Test
Để test model đã train:

```bash
cd RQ1/baselines/VulExplainer/VulExplainer_CodeBERT
python student_codebert_main.py \
    --alpha 0.7 \
    --beta 0.8 \
    --output_dir=./saved_models \
    --model_name=soft_distil_model_07.bin \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --train_data_file=../../../mydata/updated_train.csv \
    --eval_data_file=../../../mydata/updated_val.csv \
    --test_data_file=../../../mydata/updated_test.csv \
    --do_test \
    --block_size 512 \
    --epochs 50 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

**Metrics output:**
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- MCC (Matthews Correlation Coefficient)

## Các Tùy Chọn Khác

### Hard Distillation
Thay `--alpha 0.7` bằng `--use_hard_distil` để dùng hard labels thay vì soft.

### Focal Loss cho Teacher
Thêm `--use_focal_loss` vào lệnh teacher để dùng focal loss.

### Logit Adjustment
Thêm `--use_logit_adjustment --tau 1.0` để cân bằng class imbalance.

## Troubleshooting

### Out of Memory
- Giảm batch_size: `--train_batch_size 4` cho student
- Giảm block_size: `--block_size 256`

### Model không converge
- Tăng epochs: `--epochs 100`
- Điều chỉnh learning rate

### Import Errors
- Đảm bảo transformers >= 4.0
- Cài đặt torch phù hợp với CUDA version

## Notes
- Đảm bảo có GPU với CUDA để training nhanh
- Model tự động download pretrained CodeBERT nếu chưa có
- Best beta được chọn tự động từ validation set
- Predictions được lưu vào `./raw_prediction/teacher_preds.csv` cho teacher