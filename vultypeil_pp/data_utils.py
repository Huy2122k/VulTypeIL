"""
Data utilities for VulTypeIL++
"""
import ast
import pandas as pd
from typing import List
from openprompt.data_utils import InputExample


def read_prompt_examples(filename: str, classes: List[str]) -> List[InputExample]:
    """
    Read examples from CSV file.
    
    Args:
        filename: Path to CSV file
        classes: List of class names
    
    Returns:
        List of InputExample objects
    """
    examples = []
    data = pd.read_csv(filename).astype(str)
    desc = data['description'].tolist()
    code = data['abstract_func_before'].tolist()
    type_col = data['cwe_ids'].tolist()
    
    for idx in range(len(data)):
        # Convert CWE IDs to class index
        cwe_list = ast.literal_eval(type_col[idx])
        if cwe_list and cwe_list[0] in classes:
            class_idx = classes.index(cwe_list[0])
        else:
            class_idx = 0  # Default to first class
        
        examples.append(
            InputExample(
                guid=idx,
                text_a=' '.join(code[idx].split(' ')[:384]),
                text_b=' '.join(desc[idx].split(' ')[:64]),
                tgt_text=class_idx,
            )
        )
    return examples


def read_and_merge_previous_datasets(current_index: int, data_paths: List[str], 
                                     classes: List[str], 
                                     return_task_info: bool = False):
    """
    Read and merge all previous task datasets.
    
    Args:
        current_index: Current task index (1-based)
        data_paths: List of paths to task data files
        classes: List of class names
        return_task_info: Whether to return task origin info
    
    Returns:
        List of InputExample objects, optionally with task origins
    """
    merged_data = pd.DataFrame()
    examples = []
    task_origins = []
    
    for i in range(current_index - 1):
        data = pd.read_csv(data_paths[i]).astype(str)
        task_id = i + 1
        data['task_origin'] = task_id
        merged_data = pd.concat([merged_data, data], ignore_index=True)
    
    if len(merged_data) == 0:
        if return_task_info:
            return [], []
        return []
    
    desc = merged_data['description'].tolist()
    code = merged_data['abstract_func_before'].tolist()
    type_col = merged_data['cwe_ids'].tolist()
    task_origin_list = merged_data['task_origin'].tolist()
    
    for idx in range(len(merged_data)):
        cwe_list = ast.literal_eval(type_col[idx])
        if cwe_list and cwe_list[0] in classes:
            class_idx = classes.index(cwe_list[0])
        else:
            class_idx = 0
        
        examples.append(
            InputExample(
                guid=idx,
                text_a=' '.join(code[idx].split(' ')[:384]),
                text_b=' '.join(desc[idx].split(' ')[:64]),
                tgt_text=class_idx,
            )
        )
        task_origins.append(int(task_origin_list[idx]))
    
    if return_task_info:
        return examples, task_origins
    return examples
