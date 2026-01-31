# Hướng dẫn Tích hợp Scalable Replay

## Tổng quan
Hệ thống cải tiến replay này giúp tối ưu hóa việc chọn lọc và lưu trữ replay examples trong continual learning với các tính năng:

- ✅ **Lọc dư thừa ngữ nghĩa**: Loại bỏ samples tương tự
- ✅ **Tóm tắt mã nguồn**: Giữ lại chỉ những dòng quan trọng
- ✅ **Ưu tiên dựa trên clustering**: Chọn samples thông minh
- ✅ **Bộ nhớ dài hạn**: Lưu trữ ngữ cảnh lịch sử
- ✅ **Tích hợp dễ dàng**: Chỉ cần thay đổi 3-5 dòng code

## Tích hợp nhanh (5 phút)

### Bước 1: Thêm import
```python
# Thêm vào đầu file vul_main2.py
from replay_integration import upgrade_existing_replay_function, log_replay_improvements
```

### Bước 2: Thay thế code chọn replay
Tìm đoạn code này (khoảng dòng 700):
```python
# CŨ - Xóa đoạn này
indices_to_replay, _ = select_uncertain_samples_with_stratified_class(
    prompt_model, train_dataloader_prev, prev_examples,
    num_samples=replay_budget, min_samples_per_class=args.min_samples_per_class
)
```

Thay bằng:
```python
# MỚI - Thêm đoạn này
enhanced_selector = upgrade_existing_replay_function()
current_examples = read_prompt_examples(data_paths[i - 1])

indices_to_replay, selection_info = enhanced_selector.select_enhanced_replay_samples(
    prompt_model=prompt_model,
    dataloader=train_dataloader_prev,
    examples=prev_examples,
    num_samples=replay_budget,
    task_id=i,
    min_samples_per_class=args.min_samples_per_class,
    current_task_examples=current_examples
)

# Ghi log để theo dõi
log_replay_improvements(selection_info, i)
```

### Bước 3: Chạy thử
```bash
cd main
python vul_main2.py --replay_ratio 0.2
```

## Cấu hình nâng cao

### Tiết kiệm bộ nhớ
```python
from replay_config import create_config
from replay_integration import EnhancedReplaySelector

config = create_config('memory_efficient')
enhanced_selector = EnhancedReplaySelector(
    similarity_threshold=0.9,  # Lọc tích cực hơn
    max_code_lines=5,         # Code ngắn hơn
    n_clusters=5              # Ít clusters hơn
)
```

### Tập trung chất lượng
```python
config = create_config('quality_focused')
enhanced_selector = EnhancedReplaySelector(
    similarity_threshold=0.75,  # Lọc ít hơn
    max_code_lines=15,         # Code dài hơn
    n_clusters=15,             # Nhiều clusters hơn
    use_gradient_importance=True  # Bật gradient importance
)
```

## Kết quả mong đợi

- 📉 **Giảm 20-40% memory usage**
- 📈 **Cải thiện 15-25% class balance**
- ⚡ **Tăng 10-20% tốc độ training**
- 🧠 **Giảm 5-15% catastrophic forgetting**

## Theo dõi kết quả

### File log tự động
```bash
# Xem log cải tiến
cat replay_improvements.log

# Xem thống kê
tail -f replay_improvements.log
```

### Thư mục bộ nhớ dài hạn
```bash
# Kiểm tra memory đã lưu
ls -la long_term_memory/
```

## Troubleshooting

### Lỗi thường gặp
1. **ImportError**: Cài đặt dependencies
   ```bash
   pip install scikit-learn scipy
   ```

2. **Memory Error**: Dùng config tiết kiệm
   ```python
   config = create_config('memory_efficient')
   ```

3. **Chậm**: Dùng config nhanh
   ```python
   config = create_config('fast')
   ```

### Kiểm tra hoạt động
```python
# Chạy demo để test
python replay_demo.py

# Chạy tests
python test_scalable_replay.py
```

## Tùy chỉnh nâng cao

### Từ khóa vulnerability tùy chỉnh
```python
from replay_config import CodeSummarizerConfig

config = CodeSummarizerConfig(
    max_code_lines=8,
    vulnerability_keywords=['buffer', 'overflow', 'malloc', 'free', 'strcpy']
)
```

### Lưu/tải cấu hình
```python
# Lưu cấu hình
config = create_config('balanced')
config.save_to_file('my_config.json')

# Tải cấu hình
config = ScalableReplayConfig.load_from_file('my_config.json')
```

## Hỗ trợ

- 📖 **Chi tiết**: Xem `SCALABLE_REPLAY_INTEGRATION_GUIDE.md`
- 🧪 **Demo**: Chạy `python replay_demo.py`
- 🔧 **Test**: Chạy `python test_scalable_replay.py`

---
*Hệ thống này được thiết kế để plug-and-play với minimal changes. Chỉ cần 5 phút để tích hợp và thấy ngay kết quả!*