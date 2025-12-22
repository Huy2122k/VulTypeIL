# Fix cho File Excel Corrupt - task1_train.xlsx

## Vấn đề:
- File `task1_train.xlsx` bị corrupt/có vấn đề gây segfault
- File `task2_train.xlsx` và các file khác hoạt động bình thường
- `pd.read_excel()` bị crash khi đọc file task1

## Giải pháp đã áp dụng:

### 1. **Robust Excel Reading**
```python
# Thử nhiều phương pháp đọc Excel
methods = [
    lambda f: pd.read_excel(f, engine='openpyxl', sheet_name=0),
    lambda f: pd.read_excel(f, engine='xlrd', sheet_name=0),
    lambda f: pd.read_excel(f, sheet_name=0),
    lambda f: pd.read_excel(f, engine='openpyxl'),
    lambda f: pd.read_excel(f, engine='xlrd'),
    lambda f: pd.read_excel(f)
]
```

### 2. **Pre-training Data Test**
```python
def test_data_loading():
    # Test tất cả files trước khi training
    # Identify working files
    # Skip corrupt files
```

### 3. **Dynamic Task Selection**
```python
# Chỉ train trên các task có file hoạt động
working_files = [1, 2, 3, 4]  # Skip task 0 (task1) if corrupt
for task_idx in working_files:
    # Training logic
```

### 4. **Smart Previous Data Merging**
```python
def read_and_merge_previous_datasets(current_task_idx, data_paths, working_files):
    # Chỉ merge từ các file hoạt động trước đó
    previous_working = [idx for idx in working_files if idx < current_task_idx]
```

### 5. **Graceful Degradation**
- Nếu task1 bị lỗi → Bắt đầu từ task2
- Nếu task2 bị lỗi → Bắt đầu từ task3
- Vẫn duy trì incremental learning logic

## Kết quả:

### **Trước fix:**
```
Reading data from incremental_tasks/task1_train.xlsx...
Segmentation fault (core dumped)
```

### **Sau fix:**
```
Testing data loading...
Testing file 1: incremental_tasks/task1_train.xlsx
❌ CRITICAL: Failed to read task1_train.xlsx with all methods!
This file may be corrupted. Skipping...
✗ File 1 FAILED: No examples loaded

Testing file 2: incremental_tasks/task2_train.xlsx
✓ File 2 OK: 2344 examples

⚠️  Warning: Only 4/5 files are working
Working tasks: [2, 3, 4, 5]
Starting training on tasks: [2, 3, 4, 5]
```

## Cách hoạt động:

### 1. **Startup Phase:**
- Test tất cả 5 files Excel
- Identify working files
- Report corrupt files

### 2. **Training Phase:**
- Bắt đầu từ task đầu tiên hoạt động
- Treat nó như "task 1" trong incremental sequence
- Merge data chỉ từ previous working tasks

### 3. **Testing Phase:**
- Test chỉ trên các task có data
- Skip corrupt tasks

## Lợi ích:

1. **No More Segfault**: Tránh được crash do file corrupt
2. **Automatic Recovery**: Tự động skip file lỗi
3. **Maintain Logic**: Vẫn giữ incremental learning workflow
4. **Clear Reporting**: Báo cáo rõ ràng file nào bị lỗi

## Nếu muốn fix file task1_train.xlsx:

### Option 1: Re-export từ source
```python
# Nếu có file gốc, export lại
df.to_excel('task1_train_fixed.xlsx', index=False)
```

### Option 2: Copy format từ task2
```python
# Copy structure từ task2, replace data
task2_df = pd.read_excel('task2_train.xlsx')
# Modify data...
task2_df.to_excel('task1_train_fixed.xlsx', index=False)
```

### Option 3: Use CSV instead
```python
# Convert to CSV to avoid Excel issues
df.to_csv('task1_train.csv', index=False)
# Update code to read CSV
```

## Monitoring:

Code sẽ báo cáo:
- Số file hoạt động vs tổng số file
- Danh sách task được train
- Warning nếu có file bị skip

Với fix này, training sẽ chạy được ngay cả khi có file corrupt!