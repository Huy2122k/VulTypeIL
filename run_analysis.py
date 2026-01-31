#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from collections import Counter

classes = [
    'CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20', 'CWE-416',
    'CWE-190', 'CWE-200', 'CWE-120', 'CWE-399', 'CWE-401', 'CWE-264', 'CWE-772',
    'CWE-189', 'CWE-362', 'CWE-835', 'CWE-369', 'CWE-617', 'CWE-400', 'CWE-415',
    'CWE-122', 'CWE-770', 'CWE-22'
]

print("Loading data...")
with open("main/pred_result/task1_test_task_1.pred.csv") as f:
    pred_t1 = [int(x.strip()) for x in f if x.strip()]
with open("main/pred_result/task1_test_task_1.gold.csv") as f:
    gold_t1 = [int(x.strip()) for x in f if x.strip()]
with open("main/pred_result/task5_test_task_1.pred.csv") as f:
    pred_t5 = [int(x.strip()) for x in f if x.strip()]
with open("main/pred_result/task5_test_task_1.gold.csv") as f:
    gold_t5 = [int(x.strip()) for x in f if x.strip()]

acc_t1 = sum(p == g for p, g in zip(pred_t1, gold_t1)) / len(gold_t1)
acc_t5 = sum(p == g for p, g in zip(pred_t5, gold_t5)) / len(gold_t5)

print("\n" + "="*80)
print("PHAN TICH KET QUA TEST TASK1")
print("="*80)
print(f"Tong so samples: {len(gold_t1)}")
print(f"Accuracy Task1: {acc_t1:.4f} ({acc_t1*100:.2f}%)")
print(f"Accuracy Task5: {acc_t5:.4f} ({acc_t5*100:.2f}%)")
print(f"Do giam: {(acc_t1-acc_t5):.4f} ({(acc_t1-acc_t5)*100:.2f}%)")

print("\n" + "="*80)
print("PHAN TICH THEO CLASS")
print("="*80)

results = []
for idx in range(len(classes)):
    indices = [i for i, g in enumerate(gold_t1) if g == idx]
    if not indices:
        continue
    
    correct_t1 = sum(pred_t1[i] == idx for i in indices)
    correct_t5 = sum(pred_t5[i] == idx for i in indices)
    
    acc1 = correct_t1 / len(indices)
    acc5 = correct_t5 / len(indices)
    
    results.append({
        'class': classes[idx],
        'count': len(indices),
        'acc_t1': acc1,
        'acc_t5': acc5,
        'decrease': acc1 - acc5
    })

results.sort(key=lambda x: x['decrease'], reverse=True)

print(f"{'Class':<12} {'Count':<8} {'Acc T1':<10} {'Acc T5':<10} {'Giam':<10}")
print("-"*70)
for r in results:
    print(f"{r['class']:<12} {r['count']:<8} {r['acc_t1']:<10.4f} {r['acc_t5']:<10.4f} {r['decrease']:<10.4f}")

print("\n" + "="*80)
print("TOP 10 CLASSES BI GIAM NHIEU NHAT")
print("="*80)
for r in results[:10]:
    print(f"{r['class']:<12} giam {r['decrease']:.4f} ({r['acc_t1']:.2%} -> {r['acc_t5']:.2%})")

print("\n" + "="*80)
print("CONFUSION PATTERNS")
print("="*80)
confusion = Counter()
for i in range(len(gold_t5)):
    if pred_t5[i] != gold_t5[i]:
        confusion[(gold_t5[i], pred_t5[i])] += 1

print("TOP 20 nham lan:")
for (true_idx, pred_idx), count in confusion.most_common(20):
    print(f"{classes[true_idx]:>10} -> {classes[pred_idx]:<10} {count:>4} lan")

print("\nDone!")
