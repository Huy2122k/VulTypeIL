import pandas as pd
import ast

# Danh sách các CWE classes cần lọc
classes = ['CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20', 'CWE-416',
           'CWE-190', 'CWE-200', 'CWE-120', 'CWE-399', 'CWE-401', 'CWE-264', 
           'CWE-772', 'CWE-189', 'CWE-362', 'CWE-835', 'CWE-369', 'CWE-617', 
           'CWE-400', 'CWE-415', 'CWE-122', 'CWE-770', 'CWE-22']

# Đọc file CSV
df = pd.read_csv('incremental_tasks_csv/task1_test.csv')

print(f"Tổng số rows ban đầu: {len(df)}")

# Hàm kiểm tra xem cwe_ids có match với classes không
def check_cwe_match(cwe_ids_str):
    try:
        # Chuyển string thành list
        cwe_list = ast.literal_eval(cwe_ids_str)
        # Kiểm tra xem có CWE nào trong list match với classes không
        return any(cwe in classes for cwe in cwe_list)
    except:
        return False

# Lọc các rows có cwe_ids match với classes
df_filtered = df[df['cwe_ids'].apply(check_cwe_match)]

print(f"Số rows sau khi lọc: {len(df_filtered)}")
print(f"\nMột vài ví dụ về cwe_ids được lọc:")
print(df_filtered['cwe_ids'].head(10).values)

# Lưu kết quả ra file mới (tùy chọn)
df_filtered.to_csv('incremental_tasks_csv/task1_test_filtered.csv', index=False)
print(f"\nĐã lưu kết quả vào: incremental_tasks_csv/task1_test_filtered.csv")
