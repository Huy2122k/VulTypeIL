"""Quick analysis of task performance"""
import pandas as pd
from collections import Counter

# Classes
classes = [
    'CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20', 'CWE-416',
    'CWE-190', 'CWE-200', 'CWE-120', 'CWE-399', 'CWE-401', 'CWE-264', 'CWE-772',
    'CWE-189', 'CWE-362', 'CWE-835', 'CWE-369', 'CWE-617', 'CWE-400', 'CWE-415',
    'CWE-122', 'CWE-770', 'CWE-22'
]

# Load predictions
with open("main/pred_result/task1_test_task_1.pred.csv", 'r') as f:
    pred_t1 = [int(line.strip()) for line in f]

with open("main/pred_result/task1_test_task_1.gold.csv", 'r') as f:
    gold_t1 = [int(line.strip()) for line in f]

with open("main/pred_result/task5_test_task_1.pred.csv", 'r') as f:
    pred_t5 = [int(line.strip()) for line in f]

with open("main/pred_result/task5_test_task_1.gold.csv", 'r') as f:
    gold_t5 = [int(line.strip()) for line in f]

# Basic stats
acc_t1 = sum([1 for p, g in zip(pred_t1, gold_t1) if p == g]) / len(gold_t1)
acc_t5 = sum([1 for p, g in zip(pred_t5, gold_t5) if p == g]) / len(gold_t5)

print("="*80)
print("PHÂN TÍCH KẾT QUẢ")
print("="*80)
print(f"Accuracy Task1: {acc_t1:.4f} ({acc_t1*100:.2f}%)")
print(f"Accuracy Task5: {acc_t5:.4f} ({acc_t5*100:.2f}%)")
print(f"Giảm: {(acc_t1-acc_t5):.4f} ({(acc_t1-acc_t5)*100:.2f}%)")

# Per-class analysis
print("\n" + "="*80)
print("PHÂN TÍCH THEO CLASS")
print("="*80)
print(f"{'Class':<12} {'Count':<8} {'Acc T1':<10} {'Acc T5':<10} {'Giảm':<10}")
print("-"*70)

results = []
for idx in range(len(classes)):
    indices = [i for i, g in enumerate(gold_t1) if g == idx]
    if len(indices) == 0:
        continue
    
    correct_t1 = sum([1 for i in indices if pred_t1[i] == idx])
    correct_t5 = sum([1 for i in indices if pred_t5[i] == idx])
    
    acc1 = correct_t1 / len(indices)
    acc5 = correct_t5 / len(indices)
    decrease = acc1 - acc5
    
    results.append({
        'class': classes[idx],
        'idx': idx,
        'count': len(indices),
        'acc_t1': acc1,
        'acc_t5': acc5,
        'decrease': decrease
    })
    
    print(f"{classes[idx]:<12} {len(indices):<8} {acc1:<10.4f} {acc5:<10.4f} {decrease:<10.4f}")

# Top decreases
results_sorted = sorted(results, key=lambda x: x['decrease'], reverse=True)
print("\n" + "="*80)
print("TOP 10 CLASSES BỊ GIẢM NHIỀU NHẤT")
print("="*80)
for r in results_sorted[:10]:
    print(f"{r['class']:<12} giảm {r['decrease']:.4f} ({r['acc_t1']:.2%} -> {r['acc_t5']:.2%})")

# Confusion patterns
print("\n" + "="*80)
print("CONFUSION PATTERNS (Task5)")
print("="*80)
confusion = Counter()
for i in range(len(gold_t5)):
    if pred_t5[i] != gold_t5[i]:
        pattern = f"{classes[gold_t5[i]]} -> {classes[pred_t5[i]]}"
        confusion[pattern] += 1

print("TOP 15 nhầm lẫn:")
for pattern, count in confusion.most_common(15):
    print(f"{pattern:<40} {count:>4} lần")

# Save results
df = pd.DataFrame(results)
df.to_csv("class_analysis_results.csv", index=False)
print("\n✓ Đã lưu: class_analysis_results.csv")
