# Hướng dẫn Đánh giá Phase-based Continual Learning

## Tổng quan

File `evl_vul2.py` đã được cập nhật để đánh giá hiệu quả của các phase trong continual learning. Script này sẽ so sánh hiệu suất giữa Phase 1 và Phase 2 cho mỗi task.

## Cấu trúc Checkpoint Files

Script tìm kiếm các checkpoint files với định dạng:
- `task_1_phase1_best.ckpt`, `task_1_phase2_best.ckpt`
- `task_2_phase1_best.ckpt`, `task_2_phase2_best.ckpt`
- `task_3_phase1_best.ckpt`, `task_3_phase2_best.ckpt`
- `task_4_phase1_best.ckpt`, `task_4_phase2_best.ckpt`

### Vị trí tìm kiếm

Script sẽ tự động tìm kiếm trong các thư mục sau:
1. `best/best/`
2. `best/`
3. `checkpoints/`
4. `model/checkpoints/`
5. `./` (thư mục gốc)

## Cách chạy

```bash
python evl_vul2.py
```

## Các phân tích được thực hiện

### 1. So sánh Phase 1 vs Phase 2
- So sánh accuracy và F1-score giữa hai phase
- Tính toán mức độ cải thiện (improvement)
- Phân tích cải thiện trung bình cho mỗi task

**Output files:**
- `evaluation_results/phase_comparison_detailed.csv`
- `evaluation_results/phase_improvement_summary.csv`

### 2. Phân tích Catastrophic Forgetting
- Tính toán forgetting matrix cho cả Phase 1 và Phase 2
- So sánh mức độ forgetting giữa hai phase
- Đánh giá hiệu quả của replay strategy trong việc giảm forgetting

**Output files:**
- `evaluation_results/results_matrix_phase1.csv`
- `evaluation_results/results_matrix_phase2.csv`
- `evaluation_results/forgetting_matrix_phase1.csv`
- `evaluation_results/forgetting_matrix_phase2.csv`
- `evaluation_results/forgetting_comparison_phases.csv`

### 3. Hiệu quả Replay Strategy
- Phân tích cải thiện trên previous tasks (replay targets)
- Tính toán tỷ lệ thành công của replay
- So sánh performance trên current task vs previous tasks

**Output files:**
- `evaluation_results/replay_effectiveness.csv`

### 4. Biểu đồ trực quan
- Performance heatmaps cho từng phase
- So sánh side-by-side giữa Phase 1 và Phase 2
- Learning curves với phase comparison
- Bar chart thể hiện mức độ cải thiện

**Output files:**
- `evaluation_results/plots/performance_heatmap_phase1.png`
- `evaluation_results/plots/performance_heatmap_phase2.png`
- `evaluation_results/plots/phase_comparison_heatmaps.png`
- `evaluation_results/plots/learning_curves_phase_comparison.png`
- `evaluation_results/plots/phase_improvement_bars.png`

### 5. Báo cáo tổng kết
- Thống kê tổng quan về tất cả các metrics
- Summary JSON file với các chỉ số chính

**Output files:**
- `evaluation_results/summary_report.json`

## Giải thích các Metrics

### Accuracy Improvement
```
Improvement = Phase2_Accuracy - Phase1_Accuracy
```
- Giá trị dương: Phase 2 tốt hơn Phase 1
- Giá trị âm: Phase 2 kém hơn Phase 1

### Catastrophic Forgetting
```
Forgetting[i][j] = max_performance[j] - current_performance[j]
```
Với i > j (đánh giá task j sau khi học task i)
- Giá trị cao: Forgetting nhiều
- Giá trị thấp hoặc 0: Giữ được kiến thức

### Replay Success Rate
```
Success_Rate = (Số previous tasks cải thiện) / (Tổng số previous tasks)
```
- 100%: Tất cả previous tasks đều cải thiện
- 0%: Không có previous task nào cải thiện

## Ví dụ Output

### Phase Comparison
```
Task 1 - So sánh Phase 1 vs Phase 2:
  task_1:
    Accuracy: 0.8500 → 0.8750 (+0.0250)
    F1-Score: 0.8400 → 0.8650 (+0.0250)
  📈 Cải thiện trung bình: +0.0250
```

### Catastrophic Forgetting
```
Ma trận Catastrophic Forgetting Phase 1:
              Task 1  Task 2  Task 3  Task 4
After Task 1  0.0000  0.0000  0.0000  0.0000
After Task 2  0.0150  0.0000  0.0000  0.0000
After Task 3  0.0280  0.0120  0.0000  0.0000
After Task 4  0.0350  0.0200  0.0100  0.0000
```

### Replay Effectiveness
```
Task 2 - Hiệu quả Replay:
  task_1 (replay): 0.8350 → 0.8500 (+0.0150)
  task_2 (current): 0.8200 → 0.8400 (+0.0200)
  📈 Cải thiện trung bình trên previous tasks: +0.0150
  🎯 Tỷ lệ thành công replay: 100.00%
```

## Troubleshooting

### Không tìm thấy checkpoint files
```
❌ Không tìm thấy checkpoint phase nào!
```
**Giải pháp:**
1. Kiểm tra xem files đã được giải nén chưa
2. Đảm bảo tên files đúng format: `task_X_phaseY_best.ckpt`
3. Đặt files trong một trong các thư mục được tìm kiếm

### CUDA out of memory
**Giải pháp:**
1. Giảm `batch_size` trong script (mặc định: 16)
2. Sử dụng CPU: đặt `use_cuda = False`
3. Đánh giá từng checkpoint một thay vì tất cả cùng lúc

### Missing test data
```
FileNotFoundError: incremental_tasks/task1_test.xlsx
```
**Giải pháp:**
Đảm bảo các file test data tồn tại trong thư mục `incremental_tasks/`

## Tùy chỉnh

### Thay đổi số lượng tasks
Sửa trong hàm `evaluate_all_checkpoints()`:
```python
for task_id in range(1, 5):  # Thay 5 thành số tasks + 1
```

### Thêm metrics khác
Sửa trong hàm `evaluate_model()` để thêm metrics mới vào dictionary trả về.

### Thay đổi batch size
```python
batch_size = 8  # Giảm nếu gặp memory issues
```

## Kết quả mong đợi

Sau khi chạy thành công, bạn sẽ có:
- 7 CSV files với dữ liệu chi tiết
- 5 PNG files với biểu đồ trực quan
- 1 JSON file với báo cáo tổng kết

Tất cả được lưu trong thư mục `evaluation_results/`
