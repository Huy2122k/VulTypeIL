import os
import pickle
from collections import defaultdict

import pandas as pd

# Mapping CWE ID to abstract group (based on CWE taxonomy)
# This is a simplified mapping - in practice, you'd need the full CWE database
CWE_TO_GROUP = {
    # Base weaknesses
    '119': 'base', '125': 'variant', '787': 'variant', '476': 'variant',
    '20': 'class', '416': 'variant', '190': 'variant', '200': 'variant',
    '120': 'variant', '399': 'class', '401': 'variant', '264': 'class',
    '772': 'variant', '189': 'class', '362': 'class', '835': 'variant',
    '369': 'variant', '617': 'variant', '400': 'variant', '415': 'variant',
    '122': 'variant', '770': 'variant', '22': 'variant',
    # Add more as needed
}

def parse_cwe_ids(cwe_str):
    """Parse cwe_ids string like "['CWE-119']" to get first CWE ID"""
    try:
        cwe_list = eval(cwe_str)
        if isinstance(cwe_list, list) and len(cwe_list) > 0:
            return cwe_list[0]  # Take first CWE
    except:
        pass
    return 'CWE-119'  # Default

def get_abstract_group(cwe_id):
    """Get abstract group for CWE ID"""
    cwe_num = cwe_id.replace('CWE-', '')
    return CWE_TO_GROUP.get(cwe_num, 'variant')  # Default to variant

def process_task_data(task_num):
    """Process data for a single task"""
    base_path = f"incremental_tasks_csv/task{task_num}"

    train_df = pd.read_csv(f"{base_path}_train.csv")
    valid_df = pd.read_csv(f"{base_path}_valid.csv")
    test_df = pd.read_csv(f"{base_path}_test.csv")

    # Process each dataframe
    for df_name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        # Parse CWE IDs
        df['cwe_ids'] = df['cwe_ids'].apply(parse_cwe_ids)

        # Add abstract group
        df['cwe_abstract_group'] = df['cwe_ids'].apply(get_abstract_group)

        # Save processed file
        output_path = f"processed_data/task{task_num}_{df_name}.csv"
        os.makedirs("processed_data", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}: {len(df)} samples")

def merge_all_data():
    """Merge all processed data into single files"""
    all_train = []
    all_valid = []
    all_test = []

    for task_num in range(1, 6):
        train_df = pd.read_csv(f"processed_data/task{task_num}_train.csv")
        valid_df = pd.read_csv(f"processed_data/task{task_num}_valid.csv")
        test_df = pd.read_csv(f"processed_data/task{task_num}_test.csv")

        all_train.append(train_df)
        all_valid.append(valid_df)
        all_test.append(test_df)

    # Concatenate
    merged_train = pd.concat(all_train, ignore_index=True)
    merged_valid = pd.concat(all_valid, ignore_index=True)
    merged_test = pd.concat(all_test, ignore_index=True)

    # Save merged files
    merged_train.to_csv("processed_data/merged_train.csv", index=False)
    merged_valid.to_csv("processed_data/merged_valid.csv", index=False)
    merged_test.to_csv("processed_data/merged_test.csv", index=False)

    print(f"Merged train: {len(merged_train)} samples")
    print(f"Merged valid: {len(merged_valid)} samples")
    print(f"Merged test: {len(merged_test)} samples")

def create_cwe_label_map(train_file, val_file, test_file):
    """Create CWE label map pickle file from all data"""
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    # Get all unique CWE IDs from all datasets
    all_cwe_ids = set(train_df['cwe_ids'].unique())
    all_cwe_ids.update(val_df['cwe_ids'].unique())
    all_cwe_ids.update(test_df['cwe_ids'].unique())
    
    cwe_label_map = {}
    for idx, cwe in enumerate(sorted(all_cwe_ids)):
        # Count frequency from train data only
        freq = len(train_df[train_df['cwe_ids'] == cwe])

        # Create one-hot vector
        one_hot = [0] * len(all_cwe_ids)
        one_hot[idx] = 1

        cwe_label_map[cwe] = [idx, one_hot, freq]

    # Save pickle
    with open("processed_data/cwe_label_map.pkl", "wb") as f:
        pickle.dump(cwe_label_map, f)

    print(f"Created CWE label map with {len(cwe_label_map)} classes")

if __name__ == "__main__":
    # Process each task
    for task in range(1, 6):
        print(f"Processing task {task}...")
        process_task_data(task)

    # Merge all data
    print("Merging all data...")
    merge_all_data()

    # Create label map
    print("Creating CWE label map...")
    create_cwe_label_map("processed_data/merged_train.csv", "processed_data/merged_valid.csv", "processed_data/merged_test.csv")

    print("Data processing complete!")