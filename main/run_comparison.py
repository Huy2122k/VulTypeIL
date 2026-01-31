#!/usr/bin/env python3
"""
Script So sánh vul_main2.py vs vul_main4.py
==========================================

Script này chạy cả hai phiên bản để so sánh hiệu suất:
- vul_main2.py: Phiên bản gốc với Mahalanobis replay
- vul_main4.py: Phiên bản mới với Enhanced Scalable Replay

Tác giả: AI Assistant
"""

import os
import subprocess
import time
import json
from datetime import datetime


def run_experiment(script_name, config_name, extra_args=""):
    """
    Chạy thử nghiệm với script và cấu hình cụ thể
    
    Args:
        script_name: Tên script (vul_main2.py hoặc vul_main4.py)
        config_name: Tên cấu hình để lưu kết quả
        extra_args: Tham số bổ sung
    """
    print(f"\n🚀 CHẠY THỬ NGHIỆM: {script_name} - {config_name}")
    print(f"{'='*70}")
    
    # Tạo thư mục kết quả riêng cho mỗi thử nghiệm
    results_dir = f"results_{config_name}_{int(time.time())}"
    checkpoint_dir = f"model_{config_name}_{int(time.time())}"
    
    # Command để chạy
    cmd = [
        "python", script_name,
        "--results_dir", results_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--num_epochs", "10",  # Giảm epochs để test nhanh
        "--batch_size", "8",   # Giảm batch size để tiết kiệm memory
        "--replay_ratio", "0.2",
        "--min_samples_per_class", "2"
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
            "script": script_name,
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
        result_file = f"experiment_{config_name}_{int(start_time)}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_result, f, indent=2, ensure_ascii=False)
        
        if result.returncode == 0:
            print(f"✅ THÀNH CÔNG! Thời gian: {duration/60:.2f} phút")
            print(f"📁 Kết quả lưu tại: {results_dir}")
            print(f"💾 Checkpoint lưu tại: {checkpoint_dir}")
        else:
            print(f"❌ THẤT BẠI! Return code: {result.returncode}")
            print(f"Lỗi: {result.stderr}")
        
        print(f"📊 Chi tiết lưu tại: {result_file}")
        
        return experiment_result
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT sau 1 giờ!")
        return {
            "script": script_name,
            "config": config_name,
            "success": False,
            "error": "Timeout after 1 hour"
        }
    except Exception as e:
        print(f"💥 LỖI: {str(e)}")
        return {
            "script": script_name,
            "config": config_name,
            "success": False,
            "error": str(e)
        }


def run_comparison_experiments():
    """Chạy các thử nghiệm so sánh"""
    
    print(f"🎯 BẮT ĐẦU SO SÁNH VUL_MAIN2 VS VUL_MAIN4")
    print(f"{'='*70}")
    print(f"Thời gian bắt đầu: {datetime.now()}")
    print(f"{'='*70}")
    
    experiments = []
    
    # 1. Chạy vul_main2.py (baseline)
    print(f"\n📋 THỬ NGHIỆM 1: BASELINE (vul_main2.py)")
    result1 = run_experiment(
        script_name="vul_main2.py",
        config_name="baseline_mahalanobis"
    )
    experiments.append(result1)
    
    # 2. Chạy vul_main4.py với cấu hình balanced
    print(f"\n📋 THỬ NGHIỆM 2: ENHANCED BALANCED (vul_main4.py)")
    result2 = run_experiment(
        script_name="vul_main4.py",
        config_name="enhanced_balanced",
        extra_args="--replay_config_type balanced"
    )
    experiments.append(result2)
    
    # 3. Chạy vul_main4.py với cấu hình memory efficient
    print(f"\n📋 THỬ NGHIỆM 3: ENHANCED MEMORY EFFICIENT (vul_main4.py)")
    result3 = run_experiment(
        script_name="vul_main4.py",
        config_name="enhanced_memory_efficient",
        extra_args="--replay_config_type memory_efficient"
    )
    experiments.append(result3)
    
    # 4. Chạy vul_main4.py với cấu hình quality focused
    print(f"\n📋 THỬ NGHIỆM 4: ENHANCED QUALITY FOCUSED (vul_main4.py)")
    result4 = run_experiment(
        script_name="vul_main4.py",
        config_name="enhanced_quality_focused",
        extra_args="--replay_config_type quality_focused --enable_gradient_importance"
    )
    experiments.append(result4)
    
    # Tóm tắt kết quả
    print(f"\n📊 TÓM TẮT KẾT QUẢ SO SÁNH")
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
        
        # So sánh tốc độ
        baseline_time = next((exp['duration_minutes'] for exp in successful_experiments 
                            if 'baseline' in exp['config']), None)
        if baseline_time:
            print(f"\n📈 SO SÁNH VỚI BASELINE:")
            for exp in successful_experiments:
                if 'enhanced' in exp['config']:
                    speedup = baseline_time / exp['duration_minutes']
                    if speedup > 1:
                        print(f"  {exp['config']}: {speedup:.2f}x nhanh hơn")
                    else:
                        print(f"  {exp['config']}: {1/speedup:.2f}x chậm hơn")
    
    if failed_experiments:
        print(f"\n💥 CÁC THỬ NGHIỆM THẤT BẠI:")
        for exp in failed_experiments:
            print(f"  {exp['config']}: {exp.get('error', 'Unknown error')}")
    
    # Lưu tóm tắt
    summary = {
        "comparison_time": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "successful": len(successful_experiments),
        "failed": len(failed_experiments),
        "experiments": experiments
    }
    
    summary_file = f"comparison_summary_{int(time.time())}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Tóm tắt chi tiết lưu tại: {summary_file}")
    
    # Hướng dẫn phân tích kết quả
    print(f"\n🔍 HƯỚNG DẪN PHÂN TÍCH KẾT QUẢ:")
    print(f"1. Kiểm tra các file experiment_*.json để xem log chi tiết")
    print(f"2. So sánh các file results_*/task*_test_task_*.pred.csv")
    print(f"3. Kiểm tra replay_improvements.log để xem cải tiến replay")
    print(f"4. So sánh memory usage và training time")
    
    return summary


def run_quick_test():
    """Chạy test nhanh với ít epochs"""
    print(f"⚡ CHẠY TEST NHANH (2 epochs)")
    
    # Test vul_main2.py
    result1 = run_experiment(
        script_name="vul_main2.py",
        config_name="quick_test_baseline",
        extra_args="--num_epochs 2 --num_tasks 2"
    )
    
    # Test vul_main4.py
    result2 = run_experiment(
        script_name="vul_main4.py", 
        config_name="quick_test_enhanced",
        extra_args="--num_epochs 2 --num_tasks 2 --replay_config_type fast"
    )
    
    return [result1, result2]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        print("🚀 CHẠY QUICK TEST")
        results = run_quick_test()
    else:
        print("🚀 CHẠY SO SÁNH ĐẦY ĐỦ")
        print("💡 Để chạy quick test: python run_comparison.py quick")
        results = run_comparison_experiments()
    
    print(f"\n🎉 HOÀN THÀNH!")
    print(f"Thời gian kết thúc: {datetime.now()}")