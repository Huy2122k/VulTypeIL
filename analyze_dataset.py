import pandas as pd
import os

print("=== PHÂN TÍCH CÁC FILE TRONG DATASET ===\n")

# Phân tích output.csv
print("1. output.csv:")
csv_df = pd.read_csv('dataset/output.csv')
print(f"   - Số dòng: {len(csv_df)}")
print(f"   - Số cột: {len(csv_df.columns)}")
print(f"   - Tên cột: {list(csv_df.columns)}")
print(f"   - Mẫu dữ liệu (3 dòng đầu):")
print(csv_df.head(3))
print()

# Phân tích các file Excel
excel_files = {
    'train.xlsx': 'File huấn luyện',
    'valid.xlsx': 'File validation',
    'test.xlsx': 'File test',
    'cpp_processed.xlsx': 'File C++ đã xử lý'
}

for filename, description in excel_files.items():
    filepath = f'dataset/{filename}'
    if os.path.exists(filepath):
        print(f"2. {filename} ({description}):")
        df = pd.read_excel(filepath)
        print(f"   - Số dòng: {len(df)}")
        print(f"   - Số cột: {len(df.columns)}")
        print(f"   - Tên cột: {list(df.columns)}")
        print(f"   - Mẫu dữ liệu (3 dòng đầu):")
        print(df.head(3))
        print()

# Kiểm tra tính trùng lặp
print("\n=== KIỂM TRA TÍNH TRÙNG LẶP ===\n")

# Đọc các file để so sánh
train_df = pd.read_excel('dataset/train.xlsx')
valid_df = pd.read_excel('dataset/valid.xlsx')
test_df = pd.read_excel('dataset/test.xlsx')
cpp_df = pd.read_excel('dataset/cpp_processed.xlsx')

print(f"Tổng số mẫu:")
print(f"  - Train: {len(train_df)}")
print(f"  - Valid: {len(valid_df)}")
print(f"  - Test: {len(test_df)}")
print(f"  - CPP Processed: {len(cpp_df)}")
print(f"  - Tổng: {len(train_df) + len(valid_df) + len(test_df)}")
print()

# Kiểm tra trùng lặp giữa các tập
if 'func' in train_df.columns:
    train_valid_overlap = set(train_df['func']).intersection(set(valid_df['func']))
    train_test_overlap = set(train_df['func']).intersection(set(test_df['func']))
    valid_test_overlap = set(valid_df['func']).intersection(set(test_df['func']))
    
    print(f"Số mẫu trùng lặp:")
    print(f"  - Train ∩ Valid: {len(train_valid_overlap)}")
    print(f"  - Train ∩ Test: {len(train_test_overlap)}")
    print(f"  - Valid ∩ Test: {len(valid_test_overlap)}")
    print()

# So sánh cpp_processed với các file khác
if len(cpp_df) > 0:
    print(f"Mối quan hệ cpp_processed.xlsx:")
    if 'func' in cpp_df.columns and 'func' in train_df.columns:
        cpp_in_train = len(set(cpp_df['func']).intersection(set(train_df['func'])))
        cpp_in_valid = len(set(cpp_df['func']).intersection(set(valid_df['func'])))
        cpp_in_test = len(set(cpp_df['func']).intersection(set(test_df['func'])))
        print(f"  - Có trong Train: {cpp_in_train}")
        print(f"  - Có trong Valid: {cpp_in_valid}")
        print(f"  - Có trong Test: {cpp_in_test}")
