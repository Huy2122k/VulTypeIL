# Segmentation Fault Fixes - Cho GPU 40GB VRAM

## Nguyên nhân chính gây Segfault:

### 1. **DataLoader Multiprocessing** ⚠️ CRITICAL
- **Vấn đề**: PromptDataLoader sử dụng multiprocessing workers gây segfault
- **Fix**: Thêm `num_workers=0` vào tất cả PromptDataLoader
```python
PromptDataLoader(..., num_workers=0)  # Tắt multiprocessing
```

### 2. **Tokenizer Parallelism** ⚠️ CRITICAL  
- **Vấn đề**: Tokenizer parallelism conflict với multiprocessing
- **Fix**: `os.environ['TOKENIZERS_PARALLELISM'] = 'false'`

### 3. **Multiprocessing Start Method**
- **Vấn đề**: Default fork method có thể gây segfault
- **Fix**: `multiprocessing.set_start_method('spawn', force=True)`

## Các thay đổi đã thực hiện:

### 1. **Khôi phục Config cho GPU mạnh**
```python
# Khôi phục config gốc cho 40GB VRAM
batch_size = 16         # Từ 2 → 16
max_seq_l = 512         # Từ 128 → 512  
num_epochs = 100        # Từ 10 → 100
gradient_accumulation_steps = 1  # Không cần accumulation
```

### 2. **Fix DataLoader Critical**
```python
# Thêm num_workers=0 vào TẤT CẢ PromptDataLoader
PromptDataLoader(
    dataset=...,
    template=...,
    # ... other params
    num_workers=0  # ← CRITICAL FIX
)
```

### 3. **Simplify Training Loop**
```python
# Loại bỏ gradient accumulation phức tạp
# Trở về standard training loop
loss.backward()
optimizer1.step()
optimizer2.step()
scheduler1.step() 
scheduler2.step()
prompt_model.zero_grad()
```

### 4. **Environment Safety**
```python
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Critical
os.environ['OMP_NUM_THREADS'] = '4'  # Allow more threads for powerful GPU
```

### 5. **Khôi phục Text Processing**
```python
# Khôi phục về giá trị gốc
text_a = ' '.join(code[idx].split(' ')[:384])  # 384 tokens
text_b = ' '.join(desc[idx].split(' ')[:64])   # 64 tokens
```

## Tại sao các fix này hoạt động:

### **num_workers=0** là fix quan trọng nhất:
- Segfault thường xảy ra do conflict giữa PyTorch multiprocessing và OpenPrompt
- Với GPU 40GB, single-process loading vẫn đủ nhanh
- Tránh được race conditions và memory corruption

### **TOKENIZERS_PARALLELISM=false**:
- Hugging Face tokenizers có thể conflict với DataLoader workers
- Tắt parallelism tránh được threading issues

### **spawn method**:
- Fork method có thể gây memory corruption với CUDA
- Spawn method an toàn hơn nhưng chậm hơn (không quan trọng với GPU mạnh)

## Test để verify:

### 1. **Chạy test cơ bản**:
```bash
python test_basic.py
```

### 2. **Chạy training với debug**:
```bash
python -u vul2.py 2>&1 | tee training.log
```

### 3. **Monitor GPU**:
```bash
watch -n 1 nvidia-smi
```

## Nếu vẫn bị segfault:

### **Backup plan 1**: Force CPU mode
```python
use_cuda = False
```

### **Backup plan 2**: Giảm batch size
```python
batch_size = 8  # Thay vì 16
```

### **Backup plan 3**: Sử dụng model nhỏ hơn
```python
pretrainedmodel_path = "t5-base"  # Thay vì codet5-base
```

## Kết luận:

Với GPU 40GB VRAM, vấn đề không phải memory mà là:
1. **DataLoader multiprocessing conflicts** (fix chính)
2. **Tokenizer parallelism issues** 
3. **Threading/multiprocessing race conditions**

Các fix này giữ nguyên performance cao nhưng tránh được segfault.