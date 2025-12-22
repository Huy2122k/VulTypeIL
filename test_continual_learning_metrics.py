#!/usr/bin/env python3
"""
Test script để kiểm tra các continual learning metrics mới được thêm vào evaluate_checkpoints.py
"""

import numpy as np
import pandas as pd
import os
import sys

# Import functions từ evaluate_checkpoints
from evaluate_checkpoints import calculate_continual_learning_metrics, create_visualization_curves

def create_mock_results():
    """Tạo dữ liệu giả để test các metrics."""
    # Giả lập kết quả cho 5 tasks, 5 checkpoints
    mock_results = {}
    
    # Tạo accuracy matrix giả lập với pattern realistic
    # Task 1: Học tốt, sau đó bị quên một chút
    # Task 2: Học tốt, ít bị quên hơn
    # Task 3: Học khá, bị quên nhiều
    # Task 4: Học tốt, không bị quên
    # Task 5: Học cuối, chưa bị quên
    
    acc_patterns = {
        'task_1': [0.85, 0.82, 0.78, 0.75, 0.73],  # Forgetting pattern
        'task_2': [0.0, 0.88, 0.86, 0.84, 0.83],   # Less forgetting
        'task_3': [0.0, 0.0, 0.90, 0.82, 0.79],    # More forgetting
        'task_4': [0.0, 0.0, 0.0, 0.87, 0.86],     # Minimal forgetting
        'task_5': [0.0, 0.0, 0.0, 0.0, 0.89]       # Just learned
    }
    
    checkpoints = [
        'task1_final.ckpt',
        'task2_final.ckpt', 
        'task3_final.ckpt',
        'task4_final.ckpt',
        'task5_final.ckpt'
    ]
    
    for i, checkpoint in enumerate(checkpoints):
        mock_results[checkpoint] = {}
        for task_id in range(1, 6):
            task_key = f'task_{task_id}'
            acc = acc_patterns[task_key][i]
            
            # Tạo các metrics khác dựa trên accuracy
            mock_results[checkpoint][task_key] = {
                'accuracy': acc,
                'precision_macro': acc * 0.95,  # Slightly lower
                'recall_macro': acc * 0.98,
                'f1_weighted': acc * 0.96,
                'f1_macro': acc * 0.94
            }
    
    return mock_results

def test_continual_learning_metrics():
    """Test các continual learning metrics."""
    print("Testing Continual Learning Metrics")
    print("="*50)
    
    # Tạo mock data
    mock_results = create_mock_results()
    
    # Test calculate_continual_learning_metrics
    try:
        cl_metrics = calculate_continual_learning_metrics(mock_results)
        
        print("✓ Successfully calculated continual learning metrics")
        print(f"  • Average Forgetting: {cl_metrics['avg_forgetting']:.4f}")
        print(f"  • Backward Transfer: {cl_metrics['avg_bwt']:.4f}")
        print(f"  • Forward Transfer: {cl_metrics['avg_fwt']:.4f}")
        
        print(f"\nPer-task forgetting:")
        for i, f in enumerate(cl_metrics['forgetting_measures']):
            print(f"  • Task {i+1}: {f:.4f}")
        
        print(f"\nAccuracy Matrix Shape: {cl_metrics['acc_matrix'].shape}")
        print("Accuracy Matrix:")
        print(cl_metrics['acc_matrix'])
        
        return cl_metrics
        
    except Exception as e:
        print(f"✗ Error calculating metrics: {e}")
        return None

def test_visualization():
    """Test visualization functions."""
    print("\nTesting Visualization Functions")
    print("="*50)
    
    mock_results = create_mock_results()
    cl_metrics = calculate_continual_learning_metrics(mock_results)
    
    if cl_metrics is None:
        print("✗ Cannot test visualization - metrics calculation failed")
        return
    
    try:
        # Tạo thư mục test
        test_dir = "test_results"
        os.makedirs(test_dir, exist_ok=True)
        
        # Test visualization
        create_visualization_curves(mock_results, cl_metrics, test_dir)
        
        # Kiểm tra files được tạo
        fig_dir = os.path.join(test_dir, 'figures')
        expected_files = [
            'continual_learning_analysis.png',
            'accuracy_matrix_detailed.png'
        ]
        
        for file in expected_files:
            file_path = os.path.join(fig_dir, file)
            if os.path.exists(file_path):
                print(f"✓ Created: {file_path}")
            else:
                print(f"✗ Missing: {file_path}")
        
        print("✓ Visualization test completed")
        
    except Exception as e:
        print(f"✗ Error in visualization: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("Continual Learning Metrics Test Suite")
    print("="*60)
    
    # Test metrics calculation
    cl_metrics = test_continual_learning_metrics()
    
    # Test visualization
    test_visualization()
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()