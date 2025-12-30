import pandas as pd
import os

def prepare_teacher_data():
    """Nối các file tasks theo groups để tạo dữ liệu cho 3 teacher models"""
    
    # Tạo thư mục
    os.makedirs('RQ1/baselines/LFME/teacher_data', exist_ok=True)
    
    # Dictionary để lưu data theo group
    group_data = {'g1': [], 'g2': [], 'g3': []}
    
    # Đọc tất cả files
    for task_id in range(1, 6):
        for split in ['train', 'valid', 'test']:
            file_path = f'incremental_tasks_csv/task{task_id}_{split}.csv'
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                # Phân chia theo groups
                for group in ['g1', 'g2', 'g3']:
                    group_subset = df[df['groups'] == group].copy()
                    if len(group_subset) > 0:
                        group_data[group].append(group_subset)
    
    # Gộp và lưu data cho từng group
    for group in ['g1', 'g2', 'g3']:
        if group_data[group]:
            combined_df = pd.concat(group_data[group], ignore_index=True)
            
            # Chia train/val/test
            from sklearn.model_selection import train_test_split
            train_df, temp_df = train_test_split(combined_df, test_size=0.3, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
            
            # Lưu files
            train_df.to_csv(f'RQ1/baselines/LFME/teacher_data/{group}_train.csv', index=False)
            val_df.to_csv(f'RQ1/baselines/LFME/teacher_data/{group}_val.csv', index=False)
            test_df.to_csv(f'RQ1/baselines/LFME/teacher_data/{group}_test.csv', index=False)
            
            print(f"{group}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    prepare_teacher_data()