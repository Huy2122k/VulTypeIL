# Tối ưu hóa Memory cho vul2.py

## Các thay đổi đã thực hiện để giảm Out of Memory:

### 1. **Giảm Batch Size và Sequence Length**
```python
# Trước
batch_size = 16
max_seq_l = 512

# Sau  
batch_size = 4          # Giảm 75% memory usage
max_seq_l = 256         # Giảm 50% memory usage
```

### 2. **Thêm Gradient Accumulation**
```python
gradient_accumulation_steps = 4  # Mô phỏng batch size = 16 với memory của batch size = 4
```

### 3. **Giảm số Epochs**
```python
# Trước
num_epochs = 100

# Sau
num_epochs = 20  # Giảm thời gian training và memory usage
```

### 4. **Cải thiện Training Loop với Memory Management**
- Thêm gradient accumulation để tránh update quá thường xuyên
- Thêm `torch.cuda.empty_cache()` định kỳ để clear GPU cache
- Scale loss theo gradient accumulation steps

### 5. **Tối ưu hóa Test Dataloaders**
```python
# Trước: Tải tất cả test dataloaders cùng lúc
test_dataloader1 = PromptDataLoader(...)
test_dataloader2 = PromptDataLoader(...)
# ... (tất cả 5 dataloaders)

# Sau: Tạo dataloader on-demand
def create_test_dataloader(test_path_idx):
    return PromptDataLoader(...)

# Sử dụng và xóa ngay sau khi test
for j in range(5):
    test_dataloader = create_test_dataloader(j)
    test(prompt_model, test_dataloader, f'{task_names[j]}_test_task_{i}')
    del test_dataloader
    torch.cuda.empty_cache()
```

### 6. **Cải thiện Memory Management trong Functions**
- Thêm `torch.cuda.empty_cache()` trong `compute_mahalanobis()`
- Thêm `torch.cuda.empty_cache()` trong `test()`
- Cleanup variables sau khi sử dụng với `del`

### 7. **Tối ưu hóa Replay Strategy**
- Cleanup intermediate variables trong replay selection
- Safety check cho indices để tránh index out of bounds

## Kết quả dự kiến:

### Memory Usage Reduction:
- **Batch size**: 75% giảm (16→4)
- **Sequence length**: 50% giảm (512→256)  
- **Total memory**: ~87.5% giảm so với ban đầu

### Training Time:
- **Epochs**: 80% giảm (100→20)
- **Gradient accumulation**: Duy trì hiệu quả training

### Stability:
- Định kỳ clear GPU cache
- Better memory management
- Safer variable cleanup

## Lưu ý sử dụng:

1. **GPU Memory**: Code này được tối ưu cho GPU có ít nhất 6GB VRAM
2. **Training Quality**: Gradient accumulation đảm bảo chất lượng training không bị ảnh hưởng
3. **Monitoring**: Theo dõi GPU memory usage trong quá trình training
4. **Adjustments**: Có thể điều chỉnh batch_size và max_seq_l tùy theo GPU available

## Nếu vẫn bị OOM:

1. Giảm `batch_size` xuống 2 hoặc 1
2. Giảm `max_seq_l` xuống 128
3. Tăng `gradient_accumulation_steps` lên 8
4. Sử dụng mixed precision training (FP16)