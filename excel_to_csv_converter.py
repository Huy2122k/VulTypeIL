#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel to CSV Converter for Incremental Tasks
Chuyển đổi tất cả file Excel trong thư mục incremental_tasks sang CSV
"""

import os
import pandas as pd
from pathlib import Path
import glob

def convert_excel_to_csv(input_dir="incremental_tasks", output_dir="incremental_tasks_csv"):
    """
    Chuyển đổi tất cả file Excel trong thư mục input_dir sang CSV
    
    Args:
        input_dir (str): Thư mục chứa file Excel
        output_dir (str): Thư mục đầu ra cho file CSV
    """
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm tất cả file Excel trong thư mục
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx")) + glob.glob(os.path.join(input_dir, "*.xls"))
    
    if not excel_files:
        print(f"❌ Không tìm thấy file Excel nào trong thư mục: {input_dir}")
        return
    
    print(f"🔍 Tìm thấy {len(excel_files)} file Excel trong {input_dir}")
    print("="*60)
    
    converted_count = 0
    failed_count = 0
    
    for excel_file in sorted(excel_files):
        try:
            # Lấy tên file không có extension
            file_name = Path(excel_file).stem
            csv_file = os.path.join(output_dir, f"{file_name}.csv")
            
            print(f"📊 Đang chuyển đổi: {os.path.basename(excel_file)}")
            
            # Đọc file Excel
            df = pd.read_excel(excel_file)
            
            # Hiển thị thông tin cơ bản
            print(f"   - Số dòng: {len(df)}")
            print(f"   - Số cột: {len(df.columns)}")
            print(f"   - Các cột: {list(df.columns)}")
            
            # Lưu thành CSV
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            print(f"   ✅ Đã lưu: {csv_file}")
            converted_count += 1
            
        except Exception as e:
            print(f"   ❌ Lỗi khi chuyển đổi {excel_file}: {str(e)}")
            failed_count += 1
        
        print("-" * 40)
    
    # Tổng kết
    print(f"\n📋 KẾT QUẢ CHUYỂN ĐỔI:")
    print(f"✅ Thành công: {converted_count} files")
    print(f"❌ Thất bại: {failed_count} files")
    print(f"📁 File CSV được lưu trong: {output_dir}")

def convert_with_data_analysis(input_dir="incremental_tasks", output_dir="incremental_tasks_csv"):
    """
    Chuyển đổi Excel sang CSV với phân tích dữ liệu chi tiết
    """
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm file Excel
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx")) + glob.glob(os.path.join(input_dir, "*.xls"))
    
    if not excel_files:
        print(f"❌ Không tìm thấy file Excel nào trong thư mục: {input_dir}")
        return
    
    print(f"🔍 PHÂN TÍCH VÀ CHUYỂN ĐỔI {len(excel_files)} FILE EXCEL")
    print("="*80)
    
    summary_data = []
    
    for excel_file in sorted(excel_files):
        try:
            file_name = Path(excel_file).stem
            csv_file = os.path.join(output_dir, f"{file_name}.csv")
            
            print(f"\n📊 File: {os.path.basename(excel_file)}")
            print("-" * 50)
            
            # Đọc Excel
            df = pd.read_excel(excel_file)
            
            # Phân tích dữ liệu
            num_rows = len(df)
            num_cols = len(df.columns)
            columns = list(df.columns)
            
            print(f"📈 Thông tin cơ bản:")
            print(f"   - Số dòng: {num_rows:,}")
            print(f"   - Số cột: {num_cols}")
            print(f"   - Kích thước: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            print(f"\n📋 Các cột:")
            for i, col in enumerate(columns, 1):
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                print(f"   {i:2d}. {col:<25} | Type: {dtype:<10} | Null: {null_count:>4} | Unique: {unique_count:>6}")
            
            # Phân tích cột đặc biệt (nếu có)
            if 'cwe_ids' in df.columns:
                print(f"\n🔍 Phân tích CWE IDs:")
                try:
                    # Đếm các CWE ID
                    cwe_counts = {}
                    for cwe_str in df['cwe_ids'].dropna():
                        try:
                            import ast
                            cwe_list = ast.literal_eval(str(cwe_str))
                            if isinstance(cwe_list, list):
                                for cwe in cwe_list:
                                    cwe_counts[cwe] = cwe_counts.get(cwe, 0) + 1
                        except:
                            continue
                    
                    print(f"   - Tổng số CWE types: {len(cwe_counts)}")
                    print(f"   - Top 5 CWE phổ biến:")
                    for cwe, count in sorted(cwe_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"     {cwe}: {count} samples")
                        
                except Exception as e:
                    print(f"   - Lỗi phân tích CWE: {e}")
            
            # Lưu CSV
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"\n✅ Đã lưu CSV: {os.path.basename(csv_file)}")
            
            # Lưu thông tin tổng kết
            summary_data.append({
                'File': file_name,
                'Rows': num_rows,
                'Columns': num_cols,
                'Size_KB': df.memory_usage(deep=True).sum() / 1024,
                'Status': 'Success'
            })
            
        except Exception as e:
            print(f"\n❌ Lỗi: {str(e)}")
            summary_data.append({
                'File': Path(excel_file).stem,
                'Rows': 0,
                'Columns': 0,
                'Size_KB': 0,
                'Status': f'Failed: {str(e)}'
            })
    
    # Tạo báo cáo tổng kết
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "conversion_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n" + "="*80)
    print(f"📋 TỔNG KẾT CHUYỂN ĐỔI")
    print("="*80)
    print(summary_df.to_string(index=False))
    print(f"\n💾 Báo cáo chi tiết: {summary_file}")
    print(f"📁 Thư mục CSV: {output_dir}")

def batch_convert_specific_files():
    """
    Chuyển đổi các file cụ thể trong incremental_tasks
    """
    
    # Danh sách file cần chuyển đổi
    target_files = [
        "task1_train.xlsx", "task1_test.xlsx", "task1_valid.xlsx",
        "task2_train.xlsx", "task2_test.xlsx", "task2_valid.xlsx", 
        "task3_train.xlsx", "task3_test.xlsx", "task3_valid.xlsx",
        "task4_train.xlsx", "task4_test.xlsx", "task4_valid.xlsx",
        "task5_train.xlsx", "task5_test.xlsx", "task5_valid.xlsx"
    ]
    
    input_dir = "incremental_tasks"
    output_dir = "incremental_tasks_csv"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎯 CHUYỂN ĐỔI CÁC FILE TASK CỤ THỂ")
    print("="*60)
    
    found_files = []
    missing_files = []
    
    # Kiểm tra file tồn tại
    for file_name in target_files:
        file_path = os.path.join(input_dir, file_name)
        if os.path.exists(file_path):
            found_files.append(file_path)
        else:
            missing_files.append(file_name)
    
    print(f"✅ Tìm thấy: {len(found_files)} files")
    print(f"❌ Thiếu: {len(missing_files)} files")
    
    if missing_files:
        print(f"\n📋 File thiếu:")
        for file in missing_files:
            print(f"   - {file}")
    
    # Chuyển đổi các file tìm thấy
    converted = 0
    for file_path in found_files:
        try:
            file_name = Path(file_path).stem
            csv_file = os.path.join(output_dir, f"{file_name}.csv")
            
            df = pd.read_excel(file_path)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            print(f"✅ {os.path.basename(file_path)} → {os.path.basename(csv_file)} ({len(df)} rows)")
            converted += 1
            
        except Exception as e:
            print(f"❌ Lỗi {os.path.basename(file_path)}: {e}")
    
    print(f"\n🎉 Hoàn thành: {converted}/{len(found_files)} files")

def main():
    """
    Hàm main với menu lựa chọn
    """
    print("🔄 EXCEL TO CSV CONVERTER")
    print("="*50)
    print("1. Chuyển đổi cơ bản")
    print("2. Chuyển đổi với phân tích chi tiết") 
    print("3. Chuyển đổi file task cụ thể")
    print("4. Chuyển đổi tất cả (auto)")
    
    try:
        choice = input("\nChọn option (1-4, Enter=4): ").strip()
        if not choice:
            choice = "4"
            
        if choice == "1":
            convert_excel_to_csv()
        elif choice == "2":
            convert_with_data_analysis()
        elif choice == "3":
            batch_convert_specific_files()
        elif choice == "4":
            print("🚀 Chạy chuyển đổi tự động...")
            convert_with_data_analysis()
        else:
            print("❌ Lựa chọn không hợp lệ!")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Đã dừng chương trình.")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")

if __name__ == "__main__":
    main()