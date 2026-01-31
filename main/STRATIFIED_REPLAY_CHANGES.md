# Stratified Replay with Mahalanobis Distance - Cải Tiến Cho Long-Tail Distribution

## Tổng Quan Thay Đổi

File `vul_main.py` đã được cập nhật với chiến lược replay mới kết hợp:

### 1. **Mahalanobis Distance** (giữ nguyên) - Đo uncertainty
### 2. **Stratified Sampling by Class** (mới) - Đảm bảo class coverage
### 3. **Ratio-based Replay Budget** (mới) - Scale theo tổng dataset

---

## Chi Tiết Cải Tiến

### 1. Kết Hợp Mahalanobis Distance + Stratified Sampling

**Vấn đề cũ:**
- Mahalanobis chọn 100 tail + 100 head samples
- Không đảm bảo coverage cho tất cả classes
- Bias towards recent tasks
- Rare classes bị bỏ qua

**Giải pháp mới:**
```python
def select_uncertain_samples_with_stratified_class(
    prompt_model, dataloader, examples, num_samples, min_samples_per_class=2
)
```

**Chiến lược:**
1. **Tính Mahalanobis distance** cho tất cả previous samples (uncertainty measure)
2. **Group samples theo class**
3. **Phase 1**: Mỗi class chọn `min_samples_per_class` samples có **uncertainty cao nhất**
4. **Phase 2**: Phân bổ remaining budget theo **tỷ lệ class frequency**, chọn uncertain samples

**Ví dụ với Task 1 (23 classes) và replay budget = 468 samples:**

**Phase 1** (min 2 samples/class):
- CWE-119: Chọn 2 samples có Mahalanobis distance cao nhất
- CWE-787: Chọn 2 samples có Mahalanobis distance cao nhất
- ...
- CWE-617: Chọn 2 samples có Mahalanobis distance cao nhất
- Total: 46 samples (23 classes × 2)

**Phase 2** (remaining 422 samples):
- CWE-119 (13% frequency): 2 + 55 = 57 samples (uncertain ones)
- CWE-787 (11% frequency): 2 + 46 = 48 samples (uncertain ones)
- ...
- CWE-617 (0.6% frequency): 2 + 3 = 5 samples (uncertain ones)

**Lợi ích:**
- ✅ Giữ nguyên ưu điểm của Mahalanobis (chọn uncertain samples)
- ✅ Đảm bảo tất cả classes được đại diện (min 2 samples/class)
- ✅ Phản ánh đúng distribution thực tế (phân bổ theo frequency)
- ✅ Đặc biệt tốt cho long-tail: rare classes có ít nhất 2 uncertain samples

---

### 2. Ratio-based Replay Budget

**Vấn đề cũ:**
- Cố định 200 samples cho mọi task
- Không scale với kích thước dataset

**Giải pháp mới:**
```python
replay_budget = int(total_prev_samples * args.replay_ratio)
```

**Ví dụ với replay_ratio = 0.2 (20%):**
- Task 2: 2344 prev samples → 469 replay samples (20%)
- Task 3: 4637 prev samples → 927 replay samples (20%)
- Task 4: 6987 prev samples → 1397 replay samples (20%)
- Task 5: 9257 prev samples → 1851 replay samples (20%)

**So sánh với cố định 200:**
```
Old (200 fixed):
Task 2: 200/2344 = 8.5%
Task 3: 200/4637 = 4.3%  ← Quá ít!
Task 4: 200/6987 = 2.9%  ← Rất ít!
Task 5: 200/9257 = 2.2%  ← Cực ít!

New (20% ratio):
Task 2: 469/2344 = 20%
Task 3: 927/4637 = 20%
Task 4: 1397/6987 = 20%
Task 5: 1851/9257 = 20%  ← Consistent!
```

**Lợi ích:**
- ✅ Scale tự nhiên với dataset size
- ✅ Tỷ lệ coverage nhất quán qua các tasks
- ✅ Không tăng vô hạn (giới hạn bởi tỷ lệ %)
- ✅ Có thể tune dễ dàng (giảm ratio nếu cần)

---

### 3. Không Cân Bằng Task Distribution

**Quyết định thiết kế:**
- **KHÔNG** chia đều budget cho mỗi task
- **GIỮ NGUYÊN** logic chọn từ all previous data
- Mahalanobis tự nhiên sẽ chọn samples từ các tasks khác nhau dựa trên uncertainty

**Lý do:**
- Phân phối giữa các tasks không đồng đều trong thực tế
- Task gần đây có thể cần nhiều replay hơn (vì model đang học)
- Task xa có thể cần ít hơn (đã được consolidate)
- Để Mahalanobis quyết định dựa trên uncertainty thực tế

---

## Tham Số Mới

### Command Line Arguments

```bash
python main/vul_main.py \
  --replay_ratio 0.2 \              # 20% của tổng previous dataset
  --min_samples_per_class 2         # Minimum 2 samples/class
```

### Tuning Guidelines

**replay_ratio (default=0.2):**
- Tăng nếu: Forgetting nghiêm trọng (0.25-0.3)
- Giảm nếu: Training time quá lâu hoặc memory hạn chế (0.1-0.15)
- Recommended range: 0.15-0.25

**Ví dụ với các ratio khác nhau (Task 5):**
```
ratio=0.1 (10%): 926 replay samples
ratio=0.15 (15%): 1389 replay samples
ratio=0.2 (20%): 1851 replay samples  ← Default
ratio=0.25 (25%): 2314 replay samples
ratio=0.3 (30%): 2777 replay samples
```

**min_samples_per_class (default=2):**
- Tăng nếu: Classes rất quan trọng, cần coverage tốt hơn (3-4)
- Giảm nếu: Có quá nhiều classes (1)
- Recommended range: 1-3

---

## So Sánh Với Phương Pháp Cũ

### Old Method (Mahalanobis Only)

**Task 5 với 200 samples:**
```
Task 1: 114 samples (57%) - Nhiều nhưng không đủ cho 23 classes
Task 2: 66 samples (33%)
Task 3: 20 samples (10%)
Task 4: 0 samples (0%)

Class coverage:
- Chỉ chọn 100 tail + 100 head
- Nhiều classes bị bỏ qua
- Không đảm bảo min samples/class
```

### New Method (Mahalanobis + Stratified + Ratio)

**Task 5 với 1851 samples (20% of 9257):**
```
Task distribution: Tự nhiên theo uncertainty (không cố định)
Dự đoán:
  Task 1: ~600-800 samples (uncertain + old)
  Task 2: ~400-600 samples
  Task 3: ~300-400 samples
  Task 4: ~200-300 samples

Class coverage:
- Tất cả 23 classes của Task 1: min 2 samples/class
- Phân bổ theo frequency: head classes nhiều hơn
- Chọn uncertain samples trong mỗi class
- Total: 1851 samples thay vì 200 (tăng 9.25x!)
```

**Cải thiện:**
- Coverage tăng từ 2.2% → 20% (tăng 9x)
- Mỗi class đảm bảo có ít nhất 2 uncertain samples
- Phản ánh đúng distribution thực tế
- Vẫn giữ ưu điểm của Mahalanobis (chọn uncertain)

---

## Kết Quả Mong Đợi

### Task 1 Performance

**Trước (200 samples cố định):**
```
After Task 1: acc = 0.6836
After Task 2: acc = 0.6360 (↓ 4.76%)
After Task 3: acc = 0.5132 (↓ 17.04%) ⚠️ CATASTROPHIC
After Task 4: acc = 0.5482 (↓ 13.54%)
After Task 5: acc = 0.5789 (↓ 10.47%)
```

**Sau (20% ratio + stratified):**
```
After Task 1: acc = 0.6836
After Task 2: acc = 0.6500 (↓ 3.36%) ✓ Better
After Task 3: acc = 0.6300 (↓ 5.36%) ✓ Much better
After Task 4: acc = 0.6200 (↓ 6.36%) ✓ Stable
After Task 5: acc = 0.6100 (↓ 7.36%) ✓ Good
```

**Cải thiện dự đoán:**
- Giảm catastrophic forgetting từ 17% → ~7%
- Performance ổn định hơn qua các tasks
- Đặc biệt tốt cho rare classes (đảm bảo min coverage)

---

## Logging Mới

```
================================================================================
REPLAY BUFFER CONFIGURATION FOR TASK 5
================================================================================
Total previous samples: 9257
Replay ratio: 0.2 (20.0%)
Replay budget: 1851 samples
================================================================================

================================================================================
REPLAY BUFFER STATISTICS FOR TASK 5
================================================================================
Total replay samples selected: 1851

Replay samples by task origin:
  Task 1: 720 samples (38.90%)  ← Natural distribution by uncertainty
  Task 2: 550 samples (29.71%)
  Task 3: 380 samples (20.53%)
  Task 4: 201 samples (10.86%)

Replay samples by class (top 10):
  CWE-119: 95 samples (5.13%)   ← Uncertain samples from this class
  CWE-787: 82 samples (4.43%)
  CWE-125: 75 samples (4.05%)
  ...
================================================================================
```

---

## Testing

### Chạy với default parameters (20% ratio):
```bash
python main/vul_main.py
```

### Chạy với custom ratio:
```bash
# Conservative (10%)
python main/vul_main.py --replay_ratio 0.1

# Aggressive (30%)
python main/vul_main.py --replay_ratio 0.3

# Custom min samples per class
python main/vul_main.py --replay_ratio 0.2 --min_samples_per_class 3
```

### So sánh với phương pháp cũ:
```bash
# Backup và test
cp main/vul_main.py main/vul_main_new.py
# Restore old version và chạy
# Compare metrics
```

---

## Implementation Details

### Core Algorithm

```python
# 1. Compute Mahalanobis distance for all previous samples
distances, features, labels = compute_mahalanobis(model, dataloader)

# 2. Group by class
class_groups = group_by_class(distances, labels)

# 3. Phase 1: Min samples per class (most uncertain)
for class_id, samples in class_groups:
    select top-k uncertain samples (k = min_samples_per_class)

# 4. Phase 2: Stratified by frequency (most uncertain)
for class_id, samples in class_groups:
    budget = remaining * (class_frequency)
    select top-budget uncertain samples
```

### Key Differences from Old

| Aspect | Old | New |
|--------|-----|-----|
| Selection | 100 tail + 100 head | Stratified by class + uncertainty |
| Budget | Fixed 200 | Ratio-based (20% of prev data) |
| Class coverage | No guarantee | Min 2 samples/class |
| Task distribution | Natural (biased) | Natural (uncertainty-based) |
| Scalability | Poor (2.2% at Task 5) | Good (20% consistent) |

---

## Notes

- Mahalanobis distance computation giữ nguyên (không thay đổi)
- Stratified sampling áp dụng **sau khi** tính uncertainty
- Replay ratio có thể tune dễ dàng dựa trên resources
- Min samples per class đảm bảo rare classes không bị quên

---

## Future Improvements

1. **Adaptive replay ratio**: Tự động điều chỉnh dựa trên forgetting rate
2. **Class importance weighting**: Rare classes có thể cần nhiều samples hơn min
3. **Herding within class**: Thay random bằng herding để chọn representative
4. **Temperature-based uncertainty**: Combine Mahalanobis với prediction entropy
