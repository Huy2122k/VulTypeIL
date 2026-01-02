import pandas as pd
import json
import re
import os
import numpy as np
from tqdm import tqdm

# CWE to class mapping (31 classes)
CWE_TO_CLASS = {
    '79': 0, '89': 1, '119': 2, '125': 3, '190': 4, '200': 5, '209': 6,
    '252': 7, '269': 8, '287': 9, '352': 10, '362': 11, '400': 12, '401': 13,
    '416': 14, '426': 15, '476': 16, '502': 17, '522': 18, '566': 19, '571': 20,
    '601': 21, '617': 22, '662': 23, '665': 24, '667': 25, '754': 26, '762': 27,
    '763': 28, '832': 29, 'Other': 30
}

def extract_cwe_label(cwe_ids_str):
    """Extract CWE label from cwe_ids string"""
    try:
        cwe_list = eval(cwe_ids_str)
        if isinstance(cwe_list, list) and len(cwe_list) > 0:
            cwe = str(cwe_list[0]).replace('CWE-', '').strip()
            return CWE_TO_CLASS.get(cwe, 30)  # Default to 'Other' if not found
    except:
        pass
    return 30

def tokenize_code(code_str):
    """Simple tokenization of code"""
    if pd.isna(code_str):
        return []
    code_str = str(code_str)
    # Remove comments
    code_str = re.sub(r'//.*', '', code_str)
    code_str = re.sub(r'/\*.*?\*/', '', code_str, flags=re.DOTALL)
    # Extract identifiers and keywords
    tokens = re.findall(r'\b[a-zA-Z_]\w*\b', code_str)
    return tokens[:512]  # Limit to 512 tokens

def generate_simple_graph(code_str, num_nodes=128):
    """Generate a simple graph representation from code"""
    tokens = tokenize_code(code_str)
    
    # Create node features (one-hot encoded token presence + padding)
    num_tokens = len(tokens)
    if num_tokens == 0:
        # If no tokens, create empty graph
        node_features = [[0.0] * 128 for _ in range(num_nodes)]
        edges = []
    else:
        # Create simple features based on tokens
        node_features = []
        for i in range(num_nodes):
            if i < num_tokens:
                # Token-based feature (simplified)
                feat = [1.0 if j % len(tokens) == i % len(tokens) else 0.0 for j in range(128)]
            else:
                feat = [0.0] * 128
            node_features.append(feat)
        
        # Create simple edges (sequential connection + some random)
        edges = []
        for i in range(min(num_nodes - 1, num_tokens - 1)):
            edges.append([i, 0, i + 1])  # Sequential edge
        
        # Add some random edges for connectivity
        for i in range(min(10, num_tokens // 2)):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                edges.append([src, 0, dst])
    
    return node_features, edges

def csv_to_json(csv_file, json_file, num_samples=None):
    """Convert CSV to LIVABLE JSON format"""
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    if num_samples is not None:
        df = df.head(num_samples)
    
    data = []
    print(f"Converting {len(df)} samples to JSON...")
    
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        # Extract target label from CWE IDs
        target = extract_cwe_label(row['cwe_ids'])
        
        # Extract code (try multiple column names)
        code = ""
        for col in ['func_before', 'func', 'abstract_func_before', 'abstract_func']:
            if col in row and pd.notna(row[col]):
                code = str(row[col])
                break
        
        # Generate graph representation
        node_features, edges = generate_simple_graph(code, num_nodes=128)
        
        # Tokenize code for sequence
        sequence = tokenize_code(code)
        
        # Create entry
        entry = {
            'node_features': node_features,
            'graph': edges,
            'targets': [target],
            'sequence': sequence
        }
        data.append(entry)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    
    print(f"Saving to {json_file}...")
    with open(json_file, 'w') as f:
        json.dump(data, f)
    
    print(f"✓ Saved {len(data)} samples to {json_file}")

# Main conversion
if __name__ == '__main__':
    base_dir = 'data/livable_input'
    os.makedirs(base_dir, exist_ok=True)
    
    print("=" * 80)
    print("Converting CSV to LIVABLE JSON format")
    print("=" * 80)
    
    csv_to_json('incremental_tasks_csv/task1_train.csv', 
                f'{base_dir}/diverse-train-v0.json')
    csv_to_json('incremental_tasks_csv/task1_valid.csv', 
                f'{base_dir}/diverse-valid-v0.json')
    csv_to_json('incremental_tasks_csv/task1_test.csv', 
                f'{base_dir}/diverse-test-v0.json')
    
    print("=" * 80)
    print("✓ Conversion complete!")
    print("=" * 80)