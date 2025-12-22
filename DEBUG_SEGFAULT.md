# Debug Segmentation Fault - Hướng dẫn khắc phục

## Nguyên nhân có thể gây Segmentation Fault:

### 1. **Vấn đề về thư viện và dependencies**
- Xung đột phiên bản PyTorch/CUDA
- Thư viện OpenMP threading issues
- Tokenizer parallelism conflicts

### 2. **Vấn đề về memory**
- GPU memory corruption
- CPU memory overflow
- Memory fragmentation

### 3. **Vấn đề về multiprocessing**
- DataLoader workers conflicts
- Threading issues trong tokenizer

## Các bước debug:

### Bước 1: Test cơ bản
```bash
python test_basic.py
```

### Bước 2: Kiểm tra environment
```bash
# Kiểm tra CUDA
nvidia-smi

# Kiểm tra PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# Kiểm tra memory
free -h
```

### Bước 3: Chạy với debug mode
```bash
# Chạy với gdb để catch segfault
gdb python
(gdb) run vul2.py
(gdb) bt  # khi segfault xảy ra

# Hoặc với valgrind
valgrind --tool=memcheck python vul2.py
```

## Các fix đã áp dụng trong code:

### 1. **Environment Variables**
```python
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Tắt tokenizer parallelism
os.environ['OMP_NUM_THREADS'] = '1'  # Giới hạn OpenMP threads
```

### 2. **Multiprocessing Fix**
```python
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
```

### 3. **Memory Management**
```python
# Giảm batch size và sequence length
batch_size = 2
max_seq_l = 128

# Thêm memory cleanup
torch.cuda.empty_cache()
gc.collect()
```

### 4. **Error Handling**
- Wrap tất cả operations trong try-catch
- Graceful degradation khi có lỗi
- Proper cleanup khi exit

## Nếu vẫn bị segfault:

### Option 1: Chạy trên CPU
```python
use_cuda = False  # Force CPU mode
```

### Option 2: Sử dụng model nhỏ hơn
```python
# Thay vì CodeT5-base, dùng model nhỏ hơn
pretrainedmodel_path = "t5-small"
```

### Option 3: Cài đặt lại environment
```bash
# Tạo environment mới
conda create -n vul_new python=3.8
conda activate vul_new

# Cài đặt PyTorch stable
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt dependencies
pip install transformers openprompt pandas numpy scikit-learn scipy tqdm
```

### Option 4: Chạy từng phần
```python
# Test từng component riêng biệt
# 1. Test data loading
# 2. Test model loading  
# 3. Test training loop
```

## Monitoring commands:

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Monitor system memory
watch -n 1 free -h

# Monitor processes
htop
```

## Log files để check:
- `/var/log/kern.log` - kernel messages
- `dmesg` - system messages
- Core dump files nếu có

## Liên hệ support:
Nếu vẫn không giải quyết được, cung cấp:
1. Output của `test_basic.py`
2. `nvidia-smi` output
3. `python --version` và `pip list`
4. Exact error message và stack trace