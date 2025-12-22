# Hướng dẫn sử dụng Checkpoint System

## Tổng quan

File `vul2.py` đã được cập nhật để lưu checkpoint riêng biệt cho từng task, giúp dễ dàng đánh giá và so sánh hiệu suất của model tại các giai đoạn khác nhau.

## Cấu trúc Checkpoint

### Thư mục lưu trữ
- `model/best/`: Checkpoint tốt nhất (tương thích ngược)
- `model/checkpoints/`: Tất cả checkpoint theo task

### Quy ước đặt tên
- `task_X_phase1_best.ckpt`: Model tốt nhất từ Phase 1 của Task X
- `task_X_phase2_best.ckpt`: Model tốt nhất từ Phase 2 của Task X  
- `task_X_final.ckpt`: Model cuối cùng sau khi hoàn thành Task X

## Các hàm tiện ích mới

### 1. `save_task_checkpoint(prompt_model, task_id, phase="final")`
Lưu checkpoint cho task và phase cụ thể.

```python
# Lưu checkpoint cuối cùng của task 3
save_task_checkpoint(prompt_model, 3, "final")

# Lưu checkpoint phase 1 của task 2
save_task_checkpoint(prompt_model, 2, "phase1_best")
```

### 2. `load_task_checkpoint(prompt_model, task_id, phase="final")`
Load checkpoint cho task và phase cụ thể.

```python
# Load checkpoint cuối cùng của task 3
success = load_task_checkpoint(prompt_model, 3, "final")

# Load checkpoint phase 2 của task 4
success = load_task_checkpoint(prompt_model, 4, "phase2_best")
```

### 3. `list_available_checkpoints()`
Hiển thị tất cả checkpoint có sẵn.

```python
checkpoints = list_available_checkpoints()
```

## Script đánh giá: `evaluate_checkpoints.py`

### Chức năng
- Đánh giá bất kỳ checkpoint nào trên tất cả các task
- So sánh hiệu suất giữa các checkpoint
- Xuất kết quả chi tiết và tổng hợp

### Cách sử dụng

```bash
python evaluate_checkpoints.py
```

### Tùy chọn đánh giá
1. **Evaluate all checkpoints**: Đánh giá tất cả checkpoint có sẵn
2. **Evaluate specific checkpoint**: Chọn checkpoint cụ thể để đánh giá
3. **Evaluate final checkpoints only**: Chỉ đánh giá các checkpoint cuối cùng

### Kết quả đầu ra
- `results/checkpoint_evaluation/evaluation_comprehensive.csv`: Kết quả chi tiết
- `results/checkpoint_evaluation/summary_comprehensive.csv`: Kết quả tổng hợp

## Ví dụ sử dụng

### 1. Đánh giá checkpoint cụ thể
```python
from vul2 import *

# Setup model
prompt_model, mytemplate, tokenizer, WrapperClass = setup_model()

# Load checkpoint của task 3
load_task_checkpoint(prompt_model, 3, "final")

# Test trên task 1
test_dataloader = create_test_dataloader(test_paths[0], mytemplate, tokenizer, WrapperClass)
results = test(prompt_model, test_dataloader, "task3_model_on_task1")
```

### 2. So sánh hiệu suất giữa các phase
```python
# So sánh Phase 1 vs Phase 2 của Task 2
phases = ["phase1_best", "phase2_best"]
for phase in phases:
    load_task_checkpoint(prompt_model, 2, phase)
    # Đánh giá trên tất cả task...
```

### 3. Phân tích catastrophic forgetting
```python
# Đánh giá model của task 5 trên các task trước đó
load_task_checkpoint(prompt_model, 5, "final")
for task_id in range(1, 6):
    test_dataloader = create_test_dataloader(test_paths[task_id-1], ...)
    results = test(prompt_model, test_dataloader, f"task5_model_on_task{task_id}")
```

## Lợi ích

### 1. Phân tích chi tiết
- So sánh hiệu suất giữa Phase 1 và Phase 2
- Đánh giá tác động của EWC regularization
- Phân tích catastrophic forgetting

### 2. Reproducibility
- Có thể tái tạo kết quả tại bất kỳ thời điểm nào
- Lưu trữ trạng thái model tại các milestone quan trọng

### 3. Debugging và tối ưu
- Xác định task nào gây ra forgetting nhiều nhất
- So sánh các chiến lược training khác nhau
- Fine-tune hyperparameters dựa trên checkpoint cụ thể

## Lưu ý quan trọng

1. **Dung lượng**: Mỗi checkpoint có thể chiếm ~500MB-1GB tùy thuộc vào model size
2. **Tương thích**: Checkpoint chỉ tương thích với cùng architecture và configuration
3. **Backup**: Nên backup các checkpoint quan trọng để tránh mất dữ liệu

## Troubleshooting

### Lỗi thường gặp
1. **FileNotFoundError**: Checkpoint không tồn tại
   - Kiểm tra đường dẫn và tên file
   - Sử dụng `list_available_checkpoints()` để xem checkpoint có sẵn

2. **CUDA out of memory**: 
   - Giảm batch_size trong evaluation
   - Sử dụng CPU thay vì GPU cho evaluation

3. **Model architecture mismatch**:
   - Đảm bảo sử dụng cùng configuration khi load checkpoint
   - Kiểm tra version của các thư viện

### Debug tips
```python
# Kiểm tra checkpoint có load thành công không
success = load_task_checkpoint(prompt_model, task_id, phase)
if not success:
    print("Failed to load checkpoint")
    list_available_checkpoints()
```