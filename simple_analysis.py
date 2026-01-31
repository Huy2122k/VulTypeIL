"""Simple analysis without external dependencies"""
from collections import Counter

# Classes
classes = [
    'CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20', 'CWE-416',
    'CWE-190', 'CWE-200', 'CWE-120', 'CWE-399', 'CWE-401', 'CWE-264', 'CWE-772',
    'CWE-189', 'CWE-362', 'CWE-835', 'CWE-369', 'CWE-617', 'CWE-400', 'CWE-415',
    'CWE-122', 'CWE-770', 'CWE-22'
]

# Load data
with open("main/pred_result/task1_test_task_1.pred.csv") as f:
    pred_t1 = [int(x.strip()) for x in f if x.strip()]
with open("main/pred_result/task1_test_task_1.gold.csv") as f:
    gold_t1 = [int(x.strip()) for x in f if x.strip()]
with open("main/pred_result/task5_test_task_1.pred.csv") as f:
    pred_t5 = [int(x.strip()) for x in f if x.strip()]
with open("main/pred_result/task5_test_task_1.gold.csv") as f:
    gold_t5 = [int(x.strip()) for x in f if x.strip()]

# Basic accuracy
acc_t1 = sum(p == g for p, g in zip(pred_t1, gold_t1)) / len(gold_t1)
acc_t5 = sum(p == g for p, g in zip(pred_t5, gold_t5)) / len(gold_t5)

print("="*80)
print("PHÂN TÍCH KẾT QUẢ TEST TASK1")
print("="*80)
print(f"\nTổng số samples: {len(gold_t1)}")
print(f"Accuracy Task1 (ngay sau train): {acc_t1:.4f} ({acc_t1*100:.2f}%)")
print(f"Accuracy Task5 (sau train 4 tasks): {acc_t5:.4f} ({acc_t5*100:.2f}%)")
print(f"Độ giảm: {(acc_t1-acc_t5):.4f} ({(acc_t1-acc_t5)*100:.2f}%)")
print(f"Tỷ lệ giảm: {((acc_t1-acc_t5)/acc_t1)*100:.2f}%")

# Per-class analysis
print("\n" + "="*80)
print("PHÂN TÍCH THEO TỪNG CLASS")
print("="*80)
print(f"{'Class':<12} {'Count':<8} {'Acc T1':<10} {'Acc T5':<10} {'Giảm':<10}")
print("-"*70)

class_results = []
for idx in range(len(classes)):
    indices = [i for i, g in enumerate(gold_t1) if g == idx]
    if not indices:
        continue
    
    correct_t1 = sum(pred_t1[i] == idx for i in indices)
    correct_t5 = sum(pred_t5[i] == idx for i in indices)
    
    acc1 = correct_t1 / len(indices)
    acc5 = correct_t5 / len(indices)
    decrease = acc1 - acc5
    
    class_results.append({
        'class': classes[idx],
        'idx': idx,
        'count': len(indices),
        'acc_t1': acc1,
        'acc_t5': acc5,
        'decrease': decrease
    })
    
    print(f"{classes[idx]:<12} {len(indices):<8} {acc1:<10.4f} {acc5:<10.4f} {decrease:<10.4f}")

# Top decreases
class_results.sort(key=lambda x: x['decrease'], reverse=True)
print("\n" + "="*80)
print("TOP 10 CLASSES BỊ GIẢM NHIỀU NHẤT")
print("="*80)
for r in class_results[:10]:
    print(f"{r['class']:<12} giảm {r['decrease']:.4f} ({r['acc_t1']:.2%} -> {r['acc_t5']:.2%}), count={r['count']}")

# Confusion analysis
print("\n" + "="*80)
print("CONFUSION PATTERNS (Task5)")
print("="*80)
confusion = Counter()
for i in range(len(gold_t5)):
    if pred_t5[i] != gold_t5[i]:
        pattern = f"{classes[gold_t5[i]]:>10} -> {classes[pred_t5[i]]:<10}"
        confusion[pattern] += 1

print("TOP 20 nhầm lẫn:")
for pattern, count in confusion.most_common(20):
    print(f"{pattern} {count:>4} lần")

# Samples correct in T1 but wrong in T5
print("\n" + "="*80)
print("SAMPLES ĐÚNG Ở TASK1 NHƯNG SAI Ở TASK5")
print("="*80)
forgot = [i for i in range(len(gold_t1)) 
          if pred_t1[i] == gold_t1[i] and pred_t5[i] != gold_t5[i]]
print(f"Số lượng: {len(forgot)} samples ({len(forgot)/len(gold_t1)*100:.2f}%)")

forgot_patterns = Counter()
for i in forgot:
    pattern = f"{classes[gold_t5[i]]:>10} -> {classes[pred_t5[i]]:<10}"
    forgot_patterns[pattern] += 1

print("\nTop 15 patterns bị quên:")
for pattern, count in forgot_patterns.most_common(15):
    print(f"{pattern} {count:>4} lần")

# Class distribution
print("\n" + "="*80)
print("PHÂN TÍCH DISTRIBUTION")
print("="*80)
gold_dist = Counter(gold_t1)
print("Classes trong test set (sorted by count):")
for idx, count in sorted(gold_dist.items(), key=lambda x: x[1], reverse=True):
    print(f"  {classes[idx]:<15} {count:>4} samples ({count/len(gold_t1)*100:.1f}%)")

# Tail vs Head analysis
counts = [r['count'] for r in class_results]
median_count = sorted(counts)[len(counts)//2]
tail = [r for r in class_results if r['count'] <= median_count]
head = [r for r in class_results if r['count'] > median_count]

avg_dec_tail = sum(r['decrease'] for r in tail) / len(tail)
avg_dec_head = sum(r['decrease'] for r in head) / len(head)

print("\n" + "="*80)
print("TAIL vs HEAD CLASSES")
print("="*80)
print(f"Median count: {median_count}")
print(f"\nTAIL classes (<= {median_count} samples):")
print(f"  Số lượng: {len(tail)} classes")
print(f"  Độ giảm TB: {avg_dec_tail:.4f}")
print(f"\nHEAD classes (> {median_count} samples):")
print(f"  Số lượng: {len(head)} classes")
print(f"  Độ giảm TB: {avg_dec_head:.4f}")

print("\n" + "="*80)
print("KẾT LUẬN")
print("="*80)
print(f"""
1. CATASTROPHIC FORGETTING nghiêm trọng:
   - Accuracy giảm từ {acc_t1:.2%} xuống {acc_t5:.2%}
   - Giảm tuyệt đối: {(acc_t1-acc_t5):.2%}
   - Giảm tương đối: {((acc_t1-acc_t5)/acc_t1)*100:.1f}%

2. Tail classes bị ảnh hưởng {'NHIỀU HƠN' if avg_dec_tail > avg_dec_head else 'ÍT HƠN'}:
   - Tail giảm TB: {avg_dec_tail:.4f}
   - Head giảm TB: {avg_dec_head:.4f}

3. Classes bị giảm nghiêm trọng nhất (top 5):
""")
for i, r in enumerate(class_results[:5], 1):
    print(f"   {i}. {r['class']}: {r['acc_t1']:.2%} -> {r['acc_t5']:.2%} (giảm {r['decrease']:.2%})")

print(f"""
4. Model quên {len(forgot)} samples ({len(forgot)/len(gold_t1)*100:.1f}%)
   - Đúng ở Task1 nhưng sai ở Task5

5. Confusion patterns cho thấy:
   - Model nhầm lẫn giữa các CWE tương tự
   - Cần cải thiện replay strategy
   - EWC chưa đủ mạnh để giữ knowledge
""")

print("="*80)
