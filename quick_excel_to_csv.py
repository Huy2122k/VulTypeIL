#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Excel to CSV Converter
Chuyển đổi nhanh tất cả file Excel trong incremental_tasks sang CSV
"""

import os
import pandas as pd
import glob

def quick_convert():
    """Chuyển đổi nhanh tất cả file Excel sang CSV"""
    
    input_dir = "incremental_tasks"
    output_dir = "incremental_tasks_csv"
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm tất cả file Excel
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx")) + glob.glob(os.path.join(input_dir, "*.xls"))
    
    if not excel_files:
        print(f"❌ Không tìm thấy file Excel trong {input_dir}")
        return
    
    print(f"🔄 Chuyển đổi {len(excel_files)} file Excel...")
    
    success = 0
    for excel_file in excel_files:
        try:
            # Đọc Excel và lưu CSV
            df = pd.read_excel(excel_file)
            
            # Tạo tên file CSV
            base_name = os.path.splitext(os.path.basename(excel_file))[0]
            csv_file = os.path.join(output_dir, f"{base_name}.csv")
            
            # Lưu CSV
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            print(f"✅ {os.path.basename(excel_file)} → {base_name}.csv ({len(df)} rows)")
            success += 1
            
        except Exception as e:
            print(f"❌ Lỗi {os.path.basename(excel_file)}: {e}")
    
    print(f"\n🎉 Hoàn thành: {success}/{len(excel_files)} files")
    print(f"📁 File CSV trong: {output_dir}")

if __name__ == "__main__":
    quick_convert()