#!/usr/bin/env python3
"""Phân tích đơn giản các file trong thư mục dataset"""

import pandas as pd
import os

print("=" * 80)
print("PHÂN TÍCH CÁC FILE TRONG THƯ MỤC DATASET")
print("=" * 80)

# 1. Phân tích output.csv
print("\n1. FILE: output.csv")
print("-" * 60)
try:
    csv_df = pd.read_csv('dataset/output.csv')
    print(f"   Số dòng: {len(csv_df):,}")
    print(f"   Số cột: {len(csv_df.columns)}")
    print(f"   Tên cột: {list(csv_df.columns)}")
    print(f"\n   3 dòng đầu tiên:")
    print(csv_df.head(3).to_string(index=False))
except Exception as e:
    print(f"   ❌ Lỗi: {e}")

# 2. Phân tích các file Excel
excel_files = {
    'train.xlsx': 'Dữ liệu huấn luyện',
    'valid.xlsx': 'Dữ liệu validation',
    'test.xlsx': 'Dữ liệu test',
    'cpp_processed.xlsx': 'Dữ liệu C++ đã xử lý'
}

for idx, (filename, description) in enumerate(excel_files.items(), 2):
    filepath = f'dataset/{filename}'
    print(f"\n{idx}. FILE: {filename} - {description}")
    print("-" * 60)
    
    if not os.path.exists(filepath):
        print(f"   ❌ File không tồn tại")
        continue
    
    try:
        df = pd.read_excel(filepath)
        print(f"   Số dòng: {len(df):,}")
        print(f"   Số cột: {len(df.columns)}")
        print(f"   Tên cột: {list(df.columns)}")
        
        # Hiển thị 2 dòng đầu
        print(f"\n   2 dòng đầu tiên:")
        for col in df.columns:
            print(f"     - {col}: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")
        
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")

print("\n" + "=" * 80)
print("KIỂM TRA MỐI QUAN HỆ GIỮA CÁC FILE")
print("=" * 80)

try:
    train_df = pd.read_excel('dataset/train.xlsx')
    valid_df = pd.read_excel('dataset/valid.xlsx')
    test_df = pd.read_excel('dataset/test.xlsx')
    
    print(f"\nTổng số mẫu:")
    print(f"  Train:      {len(train_df):>6,} mẫu")
    print(f"  Validation: {len(valid_df):>6,} mẫu")
    print(f"  Test:       {len(test_df):>6,} mẫu")
    print(f"  {'─' * 25}")
    print(f"  Tổng cộng:  {len(train_df) + len(valid_df) + len(test_df):>6,} mẫu")
    
    # Kiểm tra trùng lặp nếu có cột 'func'
    if 'func' in train_df.columns:
        print(f"\nKiểm tra trùng lặp (dựa trên cột 'func'):")
        
        train_set = set(train_df['func'].dropna())
        valid_set = set(valid_df['func'].dropna())
        test_set = set(test_df['func'].dropna())
        
        train_valid = len(train_set & valid_set)
        train_test = len(train_set & test_set)
        valid_test = len(valid_set & test_set)
        
        print(f"  Train ∩ Valid: {train_valid:>6} mẫu trùng")
        print(f"  Train ∩ Test:  {train_test:>6} mẫu trùng")
        print(f"  Valid ∩ Test:  {valid_test:>6} mẫu trùng")
        
        if train_valid == 0 and train_test == 0 and valid_test == 0:
            print(f"  ✅ Không có trùng lặp giữa các tập!")
        else:
            print(f"  ⚠️  Có trùng lặp giữa các tập!")
    
    # Kiểm tra cpp_processed
    if os.path.exists('dataset/cpp_processed.xlsx'):
        cpp_df = pd.read_excel('dataset/cpp_processed.xlsx')
        print(f"\nMối quan hệ với cpp_processed.xlsx:")
        print(f"  Số mẫu: {len(cpp_df):,}")
        
        if 'func' in cpp_df.columns and 'func' in train_df.columns:
            cpp_set = set(cpp_df['func'].dropna())
            cpp_in_train = len(cpp_set & train_set)
            cpp_in_valid = len(cpp_set & valid_set)
            cpp_in_test = len(cpp_set & test_set)
            
            print(f"  Có trong Train: {cpp_in_train:>6} mẫu")
            print(f"  Có trong Valid: {cpp_in_valid:>6} mẫu")
            print(f"  Có trong Test:  {cpp_in_test:>6} mẫu")

except Exception as e:
    print(f"❌ Lỗi khi phân tích: {e}")

print("\n" + "=" * 80)
print("KẾT LUẬN VỀ LUỒNG XỬ LÝ DỮ LIỆU")
print("=" * 80)
print("""
Dựa trên các script xử lý dữ liệu:

1. DATASET GỐC:
   - train.xlsx, valid.xlsx, test.xlsx: Dữ liệu gốc đã được chia sẵn
   - cpp_processed.xlsx: Dữ liệu C++ đã được xử lý
   - output.csv: File CSV chứa thông tin CVE (CVSS scores)

2. LUỒNG XỬ LÝ:
   
   split_data.py:
   ├─ Đọc train.xlsx, valid.xlsx, test.xlsx
   ├─ Lấy thông tin commit time từ GitHub API
   ├─ Sắp xếp theo thời gian commit
   └─ Chia thành 5 tasks theo thứ tự thời gian
      └─ Lưu vào: incremental_tasks/task{1-5}_{train,valid,test}.xlsx

   excel_to_csv_converter.py / quick_excel_to_csv.py:
   ├─ Đọc các file Excel từ incremental_tasks/
   └─ Chuyển đổi sang CSV
      └─ Lưu vào: incremental_tasks_csv/task{1-5}_{train,valid,test}.csv

   process_incremental_data.py:
   ├─ Đọc các file CSV từ incremental_tasks_csv/
   ├─ Parse CWE IDs và thêm abstract group
   ├─ Lưu vào: processed_data/task{1-5}_{train,valid,test}.csv
   ├─ Merge tất cả tasks
   │  └─ Lưu: processed_data/merged_{train,valid,test}.csv
   └─ Tạo CWE label map
      └─ Lưu: processed_data/cwe_label_map.pkl

3. MỤC ĐÍCH:
   - Chia dữ liệu theo thời gian để mô phỏng continual learning
   - Mỗi task đại diện cho một giai đoạn thời gian
   - Model học dần qua các task (incremental learning)
""")

print("=" * 80)
