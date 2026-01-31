"""
Phân tích kết quả test để so sánh hiệu suất giữa task1 và task5 trên test set của task1
"""
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Định nghĩa các classes
classes = [
    'CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20', 'CWE-416',
    'CWE-190', 'CWE-200', 'CWE-120', 'CWE-399', 'CWE-401', 'CWE-264', 'CWE-772',
    'CWE-189', 'CWE-362', 'CWE-835', 'CWE-369', 'CWE-617', 'CWE-400', 'CWE-415',
    'CWE-122', 'CWE-770', 'CWE-22'
]

def load_predictions(pred_file, gold_file):
    """Đọc file predictions và gold labels"""
    with open(pred_file, 'r') as f:
        predictions = [int(line.strip()) for line in f.readlines()]
    
    with open(gold_file, 'r') as f:
        gold_labels = [int(line.strip()) for line in f.readlines()]
    
    return predictions, gold_labels

def load_test_data(test_csv):
    """Đọc file test CSV để lấy thông tin chi tiết"""
    df = pd.read_csv(test_csv)
    return df

def analyze_performance(pred_task1, gold_task1, pred_task5, gold_task5, test_df):
    """Phân tích chi tiết hiệu suất"""
    
    print("="*80)
    print("PHÂN TÍCH KẾT QUẢ TEST TASK1")
    print("="*80)
    
    # 1. Thống kê cơ bản
    print("\n1. THỐNG KÊ CƠ BẢN:")
    print(f"   Tổng số samples: {len(gold_task1)}")
    
    # Accuracy
    acc_task1 = sum([1 for p, g in zip(pred_task1, gold_task1) if p == g]) / len(gold_task1)
    acc_task5 = sum([1 for p, g in zip(pred_task5, gold_task5) if p == g]) / len(gold_task5)
    
    print(f"   Accuracy Task1 (test ngay sau train): {acc_task1:.4f} ({acc_task1*100:.2f}%)")
    print(f"   Accuracy Task5 (test sau khi train task 2-5): {acc_task5:.4f} ({acc_task5*100:.2f}%)")
    print(f"   Độ giảm accuracy: {(acc_task1 - acc_task5):.4f} ({(acc_task1 - acc_task5)*100:.2f}%)")
    print(f"   Tỷ lệ giảm: {((acc_task1 - acc_task5)/acc_task1)*100:.2f}%")
    
    # 2. Phân tích theo từng class
    print("\n2. PHÂN TÍCH THEO TỪNG CLASS:")
    print(f"{'Class':<12} {'Count':<8} {'Acc T1':<10} {'Acc T5':<10} {'Giảm':<10} {'Tỷ lệ giảm':<12}")
    print("-"*80)
    
    class_stats = []
    for class_idx in range(len(classes)):
        # Lấy indices của class này
        indices = [i for i, g in enumerate(gold_task1) if g == class_idx]
        if len(indices) == 0:
            continue
        
        # Tính accuracy cho class này
        correct_task1 = sum([1 for i in indices if pred_task1[i] == class_idx])
        correct_task5 = sum([1 for i in indices if pred_task5[i] == class_idx])
        
        acc_t1 = correct_task1 / len(indices)
        acc_t5 = correct_task5 / len(indices)
        decrease = acc_t1 - acc_t5
        decrease_pct = (decrease / acc_t1 * 100) if acc_t1 > 0 else 0
        
        class_stats.append({
            'class': classes[class_idx],
            'class_idx': class_idx,
            'count': len(indices),
            'acc_t1': acc_t1,
            'acc_t5': acc_t5,
            'decrease': decrease,
            'decrease_pct': decrease_pct
        })
        
        print(f"{classes[class_idx]:<12} {len(indices):<8} {acc_t1:<10.4f} {acc_t5:<10.4f} {decrease:<10.4f} {decrease_pct:<12.2f}%")
    
    # Sắp xếp theo độ giảm
    class_stats_sorted = sorted(class_stats, key=lambda x: x['decrease'], reverse=True)
    
    print("\n3. TOP 10 CLASSES BỊ GIẢM NHIỀU NHẤT:")
    print(f"{'Class':<12} {'Count':<8} {'Acc T1':<10} {'Acc T5':<10} {'Giảm':<10}")
    print("-"*70)
    for stat in class_stats_sorted[:10]:
        print(f"{stat['class']:<12} {stat['count']:<8} {stat['acc_t1']:<10.4f} {stat['acc_t5']:<10.4f} {stat['decrease']:<10.4f}")
    
    # 4. Phân tích confusion - những class nào bị nhầm lẫn
    print("\n4. PHÂN TÍCH CONFUSION (Task5):")
    print("   Những samples bị dự đoán sai ở Task5:")
    
    # Tìm các samples bị sai ở task5 nhưng đúng ở task1
    wrong_in_task5 = []
    for i in range(len(gold_task1)):
        if pred_task1[i] == gold_task1[i] and pred_task5[i] != gold_task5[i]:
            wrong_in_task5.append({
                'index': i,
                'true_label': gold_task5[i],
                'pred_task1': pred_task1[i],
                'pred_task5': pred_task5[i],
                'true_class': classes[gold_task5[i]],
                'pred_class_task5': classes[pred_task5[i]]
            })
    
    print(f"   Số samples đúng ở Task1 nhưng sai ở Task5: {len(wrong_in_task5)}")
    
    # Thống kê confusion patterns
    confusion_patterns = Counter()
    for item in wrong_in_task5:
        pattern = f"{item['true_class']} -> {item['pred_class_task5']}"
        confusion_patterns[pattern] += 1
    
    print("\n   TOP 15 CONFUSION PATTERNS (True -> Predicted):")
    for pattern, count in confusion_patterns.most_common(15):
        print(f"   {pattern:<40} {count:>5} lần")
    
    # 5. Confusion Matrix cho Task5
    print("\n5. CONFUSION MATRIX ANALYSIS:")
    cm_task5 = confusion_matrix(gold_task5, pred_task5)
    
    # Tìm các cặp class bị nhầm lẫn nhiều nhất
    confusion_pairs = []
    cm_size = cm_task5.shape[0]  # Get actual confusion matrix size
    for i in range(cm_size):
        for j in range(cm_size):
            if i != j and cm_task5[i][j] > 0:
                # Only add if indices are within classes list bounds
                if i < len(classes) and j < len(classes):
                    confusion_pairs.append({
                        'true_class': classes[i],
                        'pred_class': classes[j],
                        'count': cm_task5[i][j],
                        'true_idx': i,
                        'pred_idx': j
                    })
    
    confusion_pairs_sorted = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)
    
    print("   TOP 15 CẶP CLASS BỊ NHẦM LẪN NHIỀU NHẤT (Task5):")
    print(f"   {'True Class':<15} {'Predicted As':<15} {'Count':<8}")
    print("   " + "-"*45)
    for pair in confusion_pairs_sorted[:15]:
        print(f"   {pair['true_class']:<15} {pair['pred_class']:<15} {pair['count']:<8}")
    
    # 6. Phân tích theo distribution của classes
    print("\n6. PHÂN TÍCH THEO DISTRIBUTION:")
    gold_dist = Counter(gold_task1)
    print("   Distribution của các classes trong test set:")
    for class_idx, count in sorted(gold_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"   {classes[class_idx]:<15} {count:>5} samples")
    
    # 7. Tổng kết và lý do
    print("\n" + "="*80)
    print("7. PHÂN TÍCH LÝ DO ACC GIẢM:")
    print("="*80)
    
    # Phân loại classes theo mức độ giảm
    severe_decrease = [s for s in class_stats if s['decrease'] > 0.3]
    moderate_decrease = [s for s in class_stats if 0.1 < s['decrease'] <= 0.3]
    mild_decrease = [s for s in class_stats if 0 < s['decrease'] <= 0.1]
    
    print(f"\n   Classes bị giảm NGHIÊM TRỌNG (>30%): {len(severe_decrease)} classes")
    for s in severe_decrease:
        print(f"   - {s['class']}: {s['acc_t1']:.2%} -> {s['acc_t5']:.2%} (giảm {s['decrease']:.2%})")
    
    print(f"\n   Classes bị giảm VỪA PHẢI (10-30%): {len(moderate_decrease)} classes")
    for s in moderate_decrease[:5]:  # Chỉ hiển thị 5 đầu
        print(f"   - {s['class']}: {s['acc_t1']:.2%} -> {s['acc_t5']:.2%} (giảm {s['decrease']:.2%})")
    
    print(f"\n   Classes bị giảm NHẸ (<10%): {len(mild_decrease)} classes")
    
    # Phân tích tail classes
    tail_threshold = np.percentile([s['count'] for s in class_stats], 25)
    tail_classes = [s for s in class_stats if s['count'] <= tail_threshold]
    head_classes = [s for s in class_stats if s['count'] > tail_threshold]
    
    avg_decrease_tail = np.mean([s['decrease'] for s in tail_classes])
    avg_decrease_head = np.mean([s['decrease'] for s in head_classes])
    
    print(f"\n   TAIL CLASSES (ít samples, <={tail_threshold:.0f} samples):")
    print(f"   - Số lượng: {len(tail_classes)} classes")
    print(f"   - Độ giảm trung bình: {avg_decrease_tail:.4f}")
    
    print(f"\n   HEAD CLASSES (nhiều samples, >{tail_threshold:.0f} samples):")
    print(f"   - Số lượng: {len(head_classes)} classes")
    print(f"   - Độ giảm trung bình: {avg_decrease_head:.4f}")
    
    print("\n" + "="*80)
    print("KẾT LUẬN:")
    print("="*80)
    print(f"1. Model bị CATASTROPHIC FORGETTING sau khi train thêm 4 tasks")
    print(f"2. Accuracy giảm từ {acc_task1:.2%} xuống {acc_task5:.2%} (giảm {(acc_task1-acc_task5):.2%})")
    print(f"3. Tail classes bị ảnh hưởng {'NHIỀU HƠN' if avg_decrease_tail > avg_decrease_head else 'ÍT HƠN'} head classes")
    print(f"4. Có {len(severe_decrease)} classes bị giảm nghiêm trọng (>30%)")
    print(f"5. Top confusion patterns cho thấy model nhầm lẫn giữa các classes tương tự nhau")
    
    return class_stats, confusion_pairs_sorted, wrong_in_task5

# Main execution
if __name__ == "__main__":
    # Đường dẫn files
    pred_task1_file = "main/pred_result/task1_test_task_1.pred.csv"
    gold_task1_file = "main/pred_result/task1_test_task_1.gold.csv"
    pred_task5_file = "main/pred_result/task5_test_task_1.pred.csv"
    gold_task5_file = "main/pred_result/task5_test_task_1.gold.csv"
    test_csv_file = "incremental_tasks_csv/task1_test.csv"
    
    # Load data
    print("Đang load dữ liệu...")
    pred_task1, gold_task1 = load_predictions(pred_task1_file, gold_task1_file)
    pred_task5, gold_task5 = load_predictions(pred_task5_file, gold_task5_file)
    test_df = load_test_data(test_csv_file)
    
    print(f"Đã load {len(pred_task1)} predictions\n")
    
    # Analyze
    class_stats, confusion_pairs, wrong_samples = analyze_performance(
        pred_task1, gold_task1, pred_task5, gold_task5, test_df
    )
    
    # Save detailed results
    print("\n\nĐang lưu kết quả chi tiết...")
    
    # Save class statistics
    class_stats_df = pd.DataFrame(class_stats)
    class_stats_df.to_csv("task1_class_performance_analysis.csv", index=False)
    print("✓ Đã lưu: task1_class_performance_analysis.csv")
    
    # Save confusion pairs
    confusion_df = pd.DataFrame(confusion_pairs)
    confusion_df.to_csv("task1_confusion_analysis.csv", index=False)
    print("✓ Đã lưu: task1_confusion_analysis.csv")
    
    # Save wrong samples
    wrong_df = pd.DataFrame(wrong_samples)
    wrong_df.to_csv("task1_wrong_predictions_task5.csv", index=False)
    print("✓ Đã lưu: task1_wrong_predictions_task5.csv")
    
    print("\nHoàn thành!")
