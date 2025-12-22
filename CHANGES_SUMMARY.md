# Tóm tắt các thay đổi đã thực hiện trong vul2.py

## Các vấn đề đã được sửa:

### 1. **Cải thiện Template Prompting**
- **Trước**: Template đơn giản không phản ánh đúng phương pháp trong bài báo
- **Sau**: Cải thiện template để phù hợp hơn với task vulnerability classification:
```python
# Trước
template_text = ('The code snippet: {"placeholder":"text_a"} '
                 'The vulnerability description:  {"placeholder":"text_b"} '
                 'Identify the vulnerability type: {"mask"}.')

# Sau  
template_text = ('Given the following vulnerable code snippet: {"placeholder":"text_a"} '
                 'and its vulnerability description: {"placeholder":"text_b"}, '
                 'classify the vulnerability type as: {"mask"}.')
```

### 2. **Sửa Verbalizer Mapping**
- **Vấn đề**: Thứ tự các CWE trong verbalizer không khớp với danh sách classes
- **Giải pháp**: Sắp xếp lại verbalizer theo đúng thứ tự trong danh sách classes để đảm bảo mapping chính xác

### 3. **Sửa đường dẫn lưu model cho Windows**
- **Trước**: Sử dụng đường dẫn Linux `/workspace/model/best/best.ckpt`
- **Sau**: Sử dụng đường dẫn tương đối `model/best/best.ckpt` và tạo thư mục tự động:
```python
os.makedirs('model/best', exist_ok=True)
torch.save(prompt_model.state_dict(), 'model/best/best.ckpt')
```

### 4. **Tạo thư mục results tự động**
- **Vấn đề**: Code cố gắng ghi file vào thư mục `results` nhưng không tạo thư mục trước
- **Giải pháp**: Thêm `os.makedirs('results', exist_ok=True)` trước khi ghi file

### 5. **Sửa xử lý labels**
- **Vấn đề**: Code xử lý labels như string nhưng model cần integer labels
- **Giải pháp**: 
  - Chuyển đổi CWE IDs thành class indices trong hàm `read_prompt_examples()`
  - Xử lý cả tensor và list labels trong các hàm training và testing
  - Đảm bảo labels được convert thành tensor đúng cách

### 6. **Cải thiện xử lý dữ liệu**
- Thêm kiểm tra và xử lý trường hợp CWE ID không có trong danh sách classes
- Cải thiện xử lý trong hàm `compute_mahalanobis()` để làm việc với integer labels
- Đảm bảo tính nhất quán trong việc xử lý labels xuyên suốt pipeline

## Kết quả:
- Code hiện tại đã được sửa để phù hợp với môi trường Windows
- Template prompting được cải thiện để phản ánh đúng phương pháp trong bài báo
- Xử lý labels được sửa để đảm bảo tính chính xác của model
- Tất cả đường dẫn và thư mục được xử lý tự động
- Code không còn lỗi syntax và sẵn sàng để chạy

## Lưu ý:
- Đảm bảo có đủ GPU memory để chạy CodeT5-base model
- Kiểm tra các file dataset trong thư mục `incremental_tasks/` có đúng format không
- Model sẽ được lưu trong thư mục `model/best/` và kết quả trong thư mục `results/`