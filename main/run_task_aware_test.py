#!/usr/bin/env python3
"""
Script Test Task-Aware Replay Strategy
=====================================

Script này chạy thử nghiệm với vul_main5.py để kiểm tra
Task-Aware Replay Strategy có giải quyết được vấn đề
catastrophic forgetting của Task 1 hay không.

Tác giả: AI Assistant
"""

import os
import subprocess
import time
import json
from datetime import datetime


def run_task_aware_experiment(config_name, extra_args=""):
    """
    Chạy thử nghiệm với vul_main5.py
    
    Args:
        config_name: Tên cấu hình để lưu kết quả
        extra_args: Tham số bổ sung
    """
    print(f"\n🎯 CHẠY TASK-AWARE REPLAY TEST: {config_name}")
    print(f"{'='*70}")
    
    # Tạo thư mục kết quả riêng
    results_dir = f"results_task_aware_{config_name}_{int(time.time())}"
    checkpoint_dir = f"model_task_aware_{config_name}_{int(time.time())}"
    
    # Command để chạy
    cmd = [
        "python", "vul_main5.py",
        "--results_dir", results_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--num_epochs", "10",  # Giảm epochs để test nhanh
        "--batch_size", "8",   # Giảm batch size để tiết kiệm memory
        "--replay_ratio", "0.25",  # Tăng replay ratio để có nhiều samples hơn
        "--min_samples_per_class", "3"  # Tăng min samples per class
    ]
    
    # Thêm tham số bổ sung
    if extra_args:
        cmd.extend(extra_args.split())
    
    print(f"Command: {' '.join(cmd)}")
    
    # Ghi log thời gian bắt đầu
    start_time = time.time()
    start_datetime = datetime.now()
    
    try:
        # Chạy script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # Timeout 1 giờ
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Lưu kết quả
        experiment_result = {
            "script": "vul_main5.py",
            "config": config_name,
            "start_time": start_datetime.isoformat(),
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "results_dir": results_dir,
            "checkpoint_dir": checkpoint_dir,
            "command": " ".join(cmd)
        }
        
        # Lưu vào file JSON
        result_file = f"task_aware_experiment_{config_name}_{int(start_time)}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_result, f, indent=2, ensure_ascii=False)
        
        if result.returncode == 0:
            print(f"✅ THÀNH CÔNG! Thời gian: {duration/60:.2f} phút")
            print(f"📁 Kết quả lưu tại: {results_dir}")
            print(f"💾 Checkpoint lưu tại: {checkpoint_dir}")
            
            # Phân tích kết quả nhanh
            analyze_results_quick(results_dir)
        else:
            print(f"❌ THẤT BẠI! Return code: {result.returncode}")
            print(f"Lỗi: {result.stderr}")
        
        print(f"📊 Chi tiết lưu tại: {result_file}")
        
        return experiment_result
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT sau 1 giờ!")
        return {
            "script": "vul_main5.py",
            "config": config_name,
            "success": False,
            "error": "Timeout after 1 hour"
        }
    except Exception as e:
        print(f"💥 LỖI: {str(e)}")
        return {
            "script": "vul_main5.py",
            "config": config_name,
            "success": False,
            "error": str(e)
        }


def analyze_results_quick(results_dir):
    """Phân tích nhanh kết quả để xem có cải thiện Task 1 không"""
    try:
        import pandas as pd
        from sklearn.metrics import accuracy_score
        
        print(f"\n📊 PHÂN TÍCH NHANH KẾT QUẢ:")
        print(f"{'='*50}")
        
        # Tìm file kết quả của task cuối cùng
        task_files = []
        for i in range(1, 6):  # Task 1-5
            pred_file = os.path.join(results_dir, f"task{i}_test_task_5.pred.csv")
            gold_file = os.path.join(results_dir, f"task{i}_test_task_5.gold.csv")
            
            if os.path.exists(pred_file) and os.path.exists(gold_file):
                # Đọc predictions và gold labels
                pred = pd.read_csv(pred_file, header=None)[0].tolist()
                gold = pd.read_csv(gold_file, header=None)[0].tolist()
                
                # Tính accuracy
                acc = accuracy_score(gold, pred)
                task_files.append((i, acc))
                
                print(f"Task {i}: {acc:.4f} ({acc*100:.2f}%)")
        
        if task_files:
            # Tìm task có accuracy thấp nhất và cao nhất
            worst_task, worst_acc = min(task_files, key=lambda x: x[1])
            best_task, best_acc = max(task_files, key=lambda x: x[1])
            
            print(f"\n📈 TỔNG KẾT:")
            print(f"Task tệ nhất: Task {worst_task} ({worst_acc:.2%})")
            print(f"Task tốt nhất: Task {best_task} ({best_acc:.2%})")
            
            # Kiểm tra xem Task 1 có cải thiện không
            task1_acc = next((acc for task, acc in task_files if task == 1), None)
            if task1_acc:
                if task1_acc > 0.65:
                    print(f"✅ Task 1 đã cải thiện: {task1_acc:.2%} (> 65%)")
                elif task1_acc > 0.60:
                    print(f"🔶 Task 1 cải thiện nhẹ: {task1_acc:.2%} (> 60%)")
                else:
                    print(f"❌ Task 1 vẫn bị forgetting: {task1_acc:.2%} (< 60%)")
            
            # Tính độ lệch chuẩn
            accuracies = [acc for _, acc in task_files]
            avg_acc = sum(accuracies) / len(accuracies)
            std_acc = (sum((acc - avg_acc)**2 for acc in accuracies) / len(accuracies))**0.5
            
            print(f"Accuracy trung bình: {avg_acc:.2%}")
            print(f"Độ lệch chuẩn: {std_acc:.3f}")
            
            if std_acc < 0.08:
                print(f"✅ Performance cân bằng tốt (std < 0.08)")
            else:
                print(f"⚠️  Performance chưa cân bằng (std = {std_acc:.3f})")
        
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"⚠️  Không thể phân tích kết quả: {e}")


def run_task_aware_tests():
    """Chạy các test với Task-Aware Replay Strategy"""
    
    print(f"🎯 BẮT ĐẦU TEST TASK-AWARE REPLAY STRATEGY")
    print(f"{'='*70}")
    print(f"Mục tiêu: Cải thiện Task 1 accuracy từ 59% lên >65%")
    print(f"Thời gian bắt đầu: {datetime.now()}")
    print(f"{'='*70}")
    
    experiments = []
    
    # 1. Test với task_decay_factor cao (ưu tiên task cũ mạnh)
    print(f"\n📋 TEST 1: HIGH TASK PRIORITY (task_decay_factor=0.9)")
    result1 = run_task_aware_experiment(
        config_name="high_task_priority",
        extra_args="--task_decay_factor 0.9 --min_task_ratio 0.2 --replay_config_type balanced"
    )
    experiments.append(result1)
    
    # 2. Test với min_task_ratio cao (đảm bảo mỗi task có đủ samples)
    print(f"\n📋 TEST 2: BALANCED TASK RATIO (min_task_ratio=0.25)")
    result2 = run_task_aware_experiment(
        config_name="balanced_task_ratio",
        extra_args="--task_decay_factor 0.8 --min_task_ratio 0.25 --replay_config_type quality_focused"
    )
    experiments.append(result2)
    
    # 3. Test với cấu hình tối ưu cho Task 1
    print(f"\n📋 TEST 3: TASK1 OPTIMIZED (decay=0.85, ratio=0.22)")
    result3 = run_task_aware_experiment(
        config_name="task1_optimized",
        extra_args="--task_decay_factor 0.85 --min_task_ratio 0.22 --replay_config_type balanced --enable_gradient_importance"
    )
    experiments.append(result3)
    
    # Tóm tắt kết quả
    print(f"\n📊 TÓM TẮT KẾT QUẢ TASK-AWARE TESTS")
    print(f"{'='*70}")
    
    successful_experiments = [exp for exp in experiments if exp.get('success', False)]
    failed_experiments = [exp for exp in experiments if not exp.get('success', False)]
    
    print(f"✅ Thành công: {len(successful_experiments)}/{len(experiments)}")
    print(f"❌ Thất bại: {len(failed_experiments)}/{len(experiments)}")
    
    if successful_experiments:
        print(f"\n⏱️  THỜI GIAN THỰC THI:")
        for exp in successful_experiments:
            duration = exp.get('duration_minutes', 0)
            print(f"  {exp['config']}: {duration:.2f} phút")
    
    if failed_experiments:
        print(f"\n💥 CÁC TEST THẤT BẠI:")
        for exp in failed_experiments:
            print(f"  {exp['config']}: {exp.get('error', 'Unknown error')}")
    
    # Lưu tóm tắt
    summary = {
        "test_time": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "successful": len(successful_experiments),
        "failed": len(failed_experiments),
        "experiments": experiments,
        "objective": "Improve Task 1 accuracy from 59% to >65%"
    }
    
    summary_file = f"task_aware_test_summary_{int(time.time())}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Tóm tắt chi tiết lưu tại: {summary_file}")
    
    # Hướng dẫn phân tích kết quả
    print(f"\n🔍 HƯỚNG DẪN PHÂN TÍCH KẾT QUẢ:")
    print(f"1. Kiểm tra các file task_aware_experiment_*.json để xem log chi tiết")
    print(f"2. So sánh Task 1 accuracy giữa các test")
    print(f"3. Kiểm tra task_aware_replay_improvements.log để xem replay statistics")
    print(f"4. Tìm cấu hình tốt nhất cho Task 1")
    
    return summary


def run_quick_task_aware_test():
    """Chạy test nhanh với ít epochs"""
    print(f"⚡ CHẠY QUICK TASK-AWARE TEST (5 epochs, 3 tasks)")
    
    result = run_task_aware_experiment(
        config_name="quick_task_aware_test",
        extra_args="--num_epochs 5 --num_tasks 3 --task_decay_factor 0.9 --min_task_ratio 0.25"
    )
    
    return [result]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        print("🚀 CHẠY QUICK TASK-AWARE TEST")
        results = run_quick_task_aware_test()
    else:
        print("🚀 CHẠY TASK-AWARE TESTS ĐẦY ĐỦ")
        print("💡 Để chạy quick test: python run_task_aware_test.py quick")
        results = run_task_aware_tests()
    
    print(f"\n🎉 HOÀN THÀNH TASK-AWARE TESTS!")
    print(f"Thời gian kết thúc: {datetime.now()}")
    print(f"\n💡 Nếu Task 1 accuracy vẫn < 65%, hãy thử:")
    print(f"  - Tăng task_decay_factor lên 0.95")
    print(f"  - Tăng min_task_ratio lên 0.3")
    print(f"  - Tăng replay_ratio lên 0.3")
    print(f"  - Giảm similarity_threshold xuống 0.75")