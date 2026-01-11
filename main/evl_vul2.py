#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Script for Continual Learning Performance
Đánh giá hiệu quả của continual learning với các metrics:
- Performance trên từng task qua các phase
- Catastrophic forgetting
- Hiệu quả của replay strategy
- Forward/Backward transfer
"""

import argparse
import ast
import json
import os
import warnings
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils import InputExample
from openprompt.plms import add_special_tokens
from openprompt.plms.seq2seq import T5TokenizerWrapper
from openprompt.prompts import ManualVerbalizer, MixedTemplate
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             matthews_corrcoef,
                             precision_recall_fscore_support)
from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration

warnings.filterwarnings("ignore")

# Sử dụng cùng config với vul2.py
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 16
num_class = 23
max_seq_l = 512
use_cuda = True
model_name = "t5"
pretrainedmodel_path = "Salesforce/codet5-base"

# Define classes (giống vul2.py)
classes = [
    'CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20', 'CWE-416',
    'CWE-190', 'CWE-200', 'CWE-120', 'CWE-399', 'CWE-401', 'CWE-264', 'CWE-772',
    'CWE-189', 'CWE-362', 'CWE-835', 'CWE-369', 'CWE-617', 'CWE-400', 'CWE-415',
    'CWE-122', 'CWE-770', 'CWE-22'
]

# Data paths (default values, can be overridden by command line arguments)
default_data_dir = "incremental_tasks_csv"
default_checkpoint_dir = "best/best"

ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model','wrapper'))

def load_plm(model_name, model_path):
    """Load pre-trained language model (giống vul2.py)"""
    codet5_model = ModelClass(**{
        "config": T5Config, 
        "tokenizer": RobertaTokenizer, 
        "model": T5ForConditionalGeneration,
        "wrapper": T5TokenizerWrapper
    })

    model_class = codet5_model
    model_config = model_class.config.from_pretrained("Salesforce/codet5-base")
    model = model_class.model.from_pretrained("Salesforce/codet5-base", config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained("Salesforce/codet5-base")
    wrapper = model_class.wrapper

    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=None)
    return model, tokenizer, model_config, wrapper

def read_prompt_examples(filename):
    """Read examples from CSV file (giống vul2.py)"""
    examples = []
    data = pd.read_csv(filename).astype(str)
    desc = data['description'].tolist()
    code = data['abstract_func_before'].tolist()
    type = data['cwe_ids'].tolist()
    
    for idx in range(len(data)):
        cwe_list = ast.literal_eval(type[idx])
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
    return examples

def evaluate_model(prompt_model, test_dataloader, task_name=""):
    """Evaluate model performance"""
    prompt_model.eval()
    allpreds = []
    alllabels = []
    
    with torch.no_grad():
        for inputs in test_dataloader:
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text']
            
            if torch.is_tensor(labels):
                alllabels.extend(labels.cpu().tolist())
            else:
                alllabels.extend(labels)
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    
    # Calculate metrics
    acc = accuracy_score(alllabels, allpreds)
    precision_wei, recall_wei, f1_wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted')
    precision_ma, recall_ma, f1_ma, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro')
    mcc = matthews_corrcoef(alllabels, allpreds)
    
    return {
        'accuracy': acc,
        'precision_weighted': precision_wei,
        'recall_weighted': recall_wei,
        'f1_weighted': f1_wei,
        'precision_macro': precision_ma,
        'recall_macro': recall_ma,
        'f1_macro': f1_ma,
        'mcc': mcc,
        'predictions': allpreds,
        'labels': alllabels
    }

def load_checkpoint(prompt_model, checkpoint_path):
    """Load model checkpoint"""
    if os.path.exists(checkpoint_path):
        prompt_model.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device('cuda:0'))
        )
        print(f"Loaded checkpoint: {checkpoint_path}")
        return True
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return False

def calculate_forgetting_metrics(results_matrix):
    """
    Calculate catastrophic forgetting metrics
    results_matrix[i][j] = performance of model after task i on task j
    """
    num_tasks = len(results_matrix)
    forgetting_matrix = np.zeros((num_tasks, num_tasks))
    
    for i in range(num_tasks):
        for j in range(i):  # Only previous tasks
            if i > 0:
                # Forgetting = max performance on task j - current performance on task j
                max_perf = max([results_matrix[k][j] for k in range(j, i+1)])
                current_perf = results_matrix[i][j]
                forgetting_matrix[i][j] = max_perf - current_perf
    
    return forgetting_matrix

def calculate_transfer_metrics(results_matrix):
    """Calculate forward and backward transfer"""
    num_tasks = len(results_matrix)
    
    # Forward transfer: performance on task i when first learned vs random baseline
    forward_transfer = []
    
    # Backward transfer: average improvement on previous tasks
    backward_transfer = []
    
    for i in range(num_tasks):
        if i > 0:
            # Backward transfer: average change in performance on previous tasks
            bt = 0
            for j in range(i):
                if i > j:
                    bt += results_matrix[i][j] - results_matrix[j][j]
            backward_transfer.append(bt / i)
    
    return forward_transfer, backward_transfer

class ContinualLearningEvaluator:
    def __init__(self, data_dir, checkpoint_dir):
        self.results = defaultdict(dict)
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.test_paths = [
            os.path.join(data_dir, f"task{i}_test.csv") for i in range(1, 6)
        ]
        self.valid_paths = [
            os.path.join(data_dir, f"task{i}_valid.csv") for i in range(1, 6)
        ]
        self.setup_model()
        
    def setup_model(self):
        """Setup model and components"""
        # Load model components
        self.plm, self.tokenizer, self.model_config, self.WrapperClass = load_plm(model_name, pretrainedmodel_path)
        
        # Setup template and verbalizer (giống vul2.py)
        template_text = ('Given the following vulnerable code snippet: {"placeholder":"text_a"} '
                        'and its vulnerability description: {"placeholder":"text_b"}, '
                        'classify the vulnerability type as: {"mask"}.')
        
        self.mytemplate = MixedTemplate(tokenizer=self.tokenizer, text=template_text, model=self.plm)
        
        # Setup verbalizer
        self.myverbalizer = ManualVerbalizer(self.tokenizer, classes=classes, label_words={
            "CWE-119": ["Improper Memory Operations", "Buffer Overflow", "Memory Violation"],
            "CWE-125": ["Out-of-bounds Read", "Memory Access Violation", "Read Beyond Boundaries"],
            "CWE-787": ["Out-of-bounds Write", "Buffer Overflow", "Memory Corruption"],
            "CWE-476": ["NULL Pointer Dereference", "Access to Null Pointer", "Dereferencing Null"],
            "CWE-20": ["Improper Input Validation", "Input Sanitization Flaw", "Invalid Input Handling"],
            "CWE-416": ["Use After Free", "Dangling Pointer", "Memory Use After Deallocation"],
            "CWE-190": ["Integer Overflow", "Integer Wraparound", "Overflow in Numeric Calculations"],
            "CWE-200": ["Exposure of Sensitive Data", "Unauthorized Information Access", "Sensitive Information Leak"],
            "CWE-120": ["Classic Buffer Overflow", "Buffer Copy Error", "Unchecked Buffer Size"],
            "CWE-399": ["Resource Management Error", "Improper Resource Handling", "Insufficient Resource Control"],
            "CWE-401": ["Memory Leak", "Unreleased Memory", "Memory Management Flaw"],
            "CWE-264": ["Access Control Flaw", "Privilege Escalation", "Permission Violation"],
            "CWE-772": ["Resource Management Failure", "Resource Leak", "Missing Resource Cleanup"],
            "CWE-189": ["Numeric Error", "Numerical Miscalculation", "Mathematical Error"],
            "CWE-362": ["Race Condition", "Shared Resource Access", "Improper Synchronization"],
            "CWE-835": ["Infinite Loop", "Unreachable Loop", "Loop Without Exit Condition"],
            "CWE-369": ["Divide By Zero", "Division Error", "Mathematical Error in Calculation"],
            "CWE-617": ["Reachable Assertion", "Assertion Failure", "Accessing Unreachable Code"],
            "CWE-400": ["Uncontrolled Resource Consumption", "Excessive Resource Allocation", "Denial of Service"],
            "CWE-415": ["Double Free", "Double Memory Deallocation", "Memory Deallocation Error"],
            "CWE-122": ["Heap Overflow", "Buffer Overflow in Heap", "Heap-based Memory Corruption"],
            "CWE-770": ["Unrestricted Resource Allocation", "Resource Overconsumption", "Resource Mismanagement"],
            "CWE-22": ["Path Traversal", "Directory Traversal", "Improper Path Limitation"]
        })
        
        # Create prompt model
        self.prompt_model = PromptForClassification(
            plm=self.plm, 
            template=self.mytemplate, 
            verbalizer=self.myverbalizer, 
            freeze_plm=False
        )
        
        if use_cuda:
            self.prompt_model = self.prompt_model.cuda()
            
        # Setup test dataloaders
        self.test_dataloaders = []
        for i, test_path in enumerate(self.test_paths):
            dataloader = PromptDataLoader(
                dataset=read_prompt_examples(test_path),
                template=self.mytemplate,
                tokenizer=self.tokenizer,
                tokenizer_wrapper_class=self.WrapperClass,
                max_seq_length=max_seq_l,
                batch_size=batch_size,
                shuffle=False,  # Don't shuffle for consistent evaluation
                teacher_forcing=False,
                predict_eos_token=False,
                truncate_method="head",
                decoder_max_length=3
            )
            self.test_dataloaders.append(dataloader)
    
    def evaluate_all_checkpoints(self):
        """Evaluate all available phase-based checkpoints"""
        print("🔍 Đang tìm kiếm các checkpoint theo phase...")
        
        # Check for checkpoints in the specified directory
        found_checkpoints = []
        checkpoint_base_dir = self.checkpoint_dir
        
        if os.path.exists(checkpoint_base_dir):
            for task_id in range(1, 6):  # Tasks 1-4
                for phase in [1, 2]:
                    checkpoint_name = f'task_{task_id}_phase{phase}_best.ckpt'
                    checkpoint_path = os.path.join(checkpoint_base_dir, checkpoint_name)
                    if os.path.exists(checkpoint_path):
                        found_checkpoints.append((checkpoint_name, checkpoint_path))
        
        if not found_checkpoints:
            print(f"❌ Không tìm thấy checkpoint phase nào trong {checkpoint_base_dir}!")
            return
            
        print(f"✅ Tìm thấy {len(found_checkpoints)} checkpoint(s) trong {checkpoint_base_dir}")
        
        # Evaluate each checkpoint on all tasks
        for checkpoint_name, checkpoint_path in sorted(found_checkpoints):
            print(f"\n📊 Đánh giá checkpoint: {checkpoint_name}")
            
            if load_checkpoint(self.prompt_model, checkpoint_path):
                checkpoint_results = {}
                
                for task_id, test_dataloader in enumerate(self.test_dataloaders, 1):
                    print(f"  - Task {task_id}...", end=" ")
                    results = evaluate_model(self.prompt_model, test_dataloader, f"task_{task_id}")
                    checkpoint_results[f'task_{task_id}'] = results
                    print(f"Acc: {results['accuracy']:.4f}")
                
                self.results[checkpoint_name] = checkpoint_results
    
    def analyze_phase_comparison(self):
        """So sánh hiệu quả giữa Phase 1 và Phase 2"""
        print("\n" + "="*60)
        print("📈 PHÂN TÍCH SO SÁNH PHASE 1 vs PHASE 2")
        print("="*60)
        
        phase_comparison = defaultdict(dict)
        
        # Parse checkpoint names to extract task and phase info
        for checkpoint_name, results in self.results.items():
            # Parse checkpoint name: task_X_phaseY_best.ckpt
            if 'phase1' in checkpoint_name:
                # Extract task number
                parts = checkpoint_name.split('_')
                if len(parts) >= 3:
                    task_id = parts[1]  # task_X_phase1_best.ckpt -> X
                    phase_comparison[task_id]['phase1'] = results
            elif 'phase2' in checkpoint_name:
                parts = checkpoint_name.split('_')
                if len(parts) >= 3:
                    task_id = parts[1]  # task_X_phase2_best.ckpt -> X
                    phase_comparison[task_id]['phase2'] = results
        
        # Create detailed comparison table
        comparison_data = []
        improvement_summary = []
        
        for task_id in sorted(phase_comparison.keys()):
            if 'phase1' in phase_comparison[task_id] and 'phase2' in phase_comparison[task_id]:
                phase1_results = phase_comparison[task_id]['phase1']
                phase2_results = phase_comparison[task_id]['phase2']
                
                print(f"\n📊 Task {task_id} - So sánh Phase 1 vs Phase 2:")
                print("-" * 50)
                
                task_improvements = []
                
                for eval_task in sorted(phase1_results.keys()):
                    p1_metrics = phase1_results[eval_task]
                    p2_metrics = phase2_results[eval_task]
                    
                    p1_acc = p1_metrics['accuracy']
                    p2_acc = p2_metrics['accuracy']
                    p1_f1 = p1_metrics['f1_weighted']
                    p2_f1 = p2_metrics['f1_weighted']
                    
                    acc_improvement = p2_acc - p1_acc
                    f1_improvement = p2_f1 - p1_f1
                    
                    print(f"  {eval_task}:")
                    print(f"    Accuracy: {p1_acc:.4f} → {p2_acc:.4f} ({acc_improvement:+.4f})")
                    print(f"    F1-Score: {p1_f1:.4f} → {p2_f1:.4f} ({f1_improvement:+.4f})")
                    
                    comparison_data.append({
                        'Training_Task': f'Task {task_id}',
                        'Eval_Task': eval_task,
                        'Phase1_Acc': p1_acc,
                        'Phase2_Acc': p2_acc,
                        'Acc_Improvement': acc_improvement,
                        'Acc_Improvement_Pct': (acc_improvement / p1_acc * 100) if p1_acc > 0 else 0,
                        'Phase1_F1': p1_f1,
                        'Phase2_F1': p2_f1,
                        'F1_Improvement': f1_improvement,
                        'F1_Improvement_Pct': (f1_improvement / p1_f1 * 100) if p1_f1 > 0 else 0
                    })
                    
                    task_improvements.append(acc_improvement)
                
                # Calculate average improvement for this task
                avg_improvement = np.mean(task_improvements)
                improvement_summary.append({
                    'Task': f'Task {task_id}',
                    'Avg_Accuracy_Improvement': avg_improvement,
                    'Num_Eval_Tasks': len(task_improvements),
                    'Best_Improvement': max(task_improvements),
                    'Worst_Improvement': min(task_improvements)
                })
                
                print(f"  📈 Cải thiện trung bình: {avg_improvement:+.4f}")
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print(f"\n📋 Bảng so sánh chi tiết Phase 1 vs Phase 2:")
            print(df_comparison.to_string(index=False, float_format='%.4f'))
            
            # Summary table
            if improvement_summary:
                df_summary = pd.DataFrame(improvement_summary)
                print(f"\n📊 Tổng kết cải thiện theo task:")
                print(df_summary.to_string(index=False, float_format='%.4f'))
            
            # Save to CSV
            os.makedirs('evaluation_results', exist_ok=True)
            df_comparison.to_csv('evaluation_results/phase_comparison_detailed.csv', index=False)
            if improvement_summary:
                df_summary.to_csv('evaluation_results/phase_improvement_summary.csv', index=False)
            
            print("\n💾 Đã lưu kết quả vào:")
            print("  - evaluation_results/phase_comparison_detailed.csv")
            print("  - evaluation_results/phase_improvement_summary.csv")
        
        return comparison_data, improvement_summary
    
    def analyze_catastrophic_forgetting(self):
        """Phân tích catastrophic forgetting cho cả Phase 1 và Phase 2"""
        print("\n" + "="*60)
        print("🧠 PHÂN TÍCH CATASTROPHIC FORGETTING")
        print("="*60)
        
        # Analyze forgetting for both phases
        for phase in [1, 2]:
            print(f"\n📊 Phase {phase} Analysis:")
            print("-" * 40)
            
            # Build results matrix for this phase
            phase_results = {}
            for checkpoint_name, results in self.results.items():
                if f'phase{phase}' in checkpoint_name:
                    # Extract task number
                    parts = checkpoint_name.split('_')
                    if len(parts) >= 3:
                        task_id = int(parts[1])
                        phase_results[task_id] = results
            
            if not phase_results:
                print(f"❌ Không tìm thấy checkpoint Phase {phase}!")
                continue
            
            # Create results matrix
            max_task = max(phase_results.keys())
            results_matrix = np.zeros((max_task, max_task))
            
            for i in range(1, max_task + 1):
                if i in phase_results:
                    for j in range(1, max_task + 1):
                        task_key = f'task_{j}'
                        if task_key in phase_results[i]:
                            results_matrix[i-1][j-1] = phase_results[i][task_key]['accuracy']
            
            # Calculate forgetting
            forgetting_matrix = calculate_forgetting_metrics(results_matrix)
            
            print(f"\n� ậMa trận kết quả Phase {phase} (Accuracy):")
            print("Hàng: Model sau khi học task i")
            print("Cột: Đánh giá trên task j")
            df_results = pd.DataFrame(results_matrix, 
                                     index=[f'After Task {i+1}' for i in range(max_task)],
                                     columns=[f'Task {j+1}' for j in range(max_task)])
            print(df_results.to_string(float_format='%.4f'))
            
            print(f"\n🔥 Ma trận Catastrophic Forgetting Phase {phase}:")
            df_forgetting = pd.DataFrame(forgetting_matrix,
                                       index=[f'After Task {i+1}' for i in range(max_task)],
                                       columns=[f'Task {j+1}' for j in range(max_task)])
            print(df_forgetting.to_string(float_format='%.4f'))
            
            # Calculate average forgetting
            avg_forgetting = []
            for i in range(1, max_task):
                task_forgetting = [forgetting_matrix[i][j] for j in range(i) if forgetting_matrix[i][j] > 0]
                if task_forgetting:
                    avg_forgetting.append(np.mean(task_forgetting))
                else:
                    avg_forgetting.append(0)
            
            print(f"\n📈 Trung bình Catastrophic Forgetting Phase {phase}:")
            for i, avg_f in enumerate(avg_forgetting, 2):
                print(f"  Task {i}: {avg_f:.4f}")
            
            # Save results
            os.makedirs('evaluation_results', exist_ok=True)
            df_results.to_csv(f'evaluation_results/results_matrix_phase{phase}.csv')
            df_forgetting.to_csv(f'evaluation_results/forgetting_matrix_phase{phase}.csv')
        
        # Compare forgetting between phases
        self.compare_forgetting_between_phases()
        
        return results_matrix, forgetting_matrix
    
    def compare_forgetting_between_phases(self):
        """So sánh catastrophic forgetting giữa Phase 1 và Phase 2"""
        print(f"\n🔄 So sánh Catastrophic Forgetting giữa các Phase:")
        print("-" * 50)
        
        phase1_forgetting = {}
        phase2_forgetting = {}
        
        # Calculate forgetting for each phase
        for phase in [1, 2]:
            phase_results = {}
            for checkpoint_name, results in self.results.items():
                if f'phase{phase}' in checkpoint_name:
                    parts = checkpoint_name.split('_')
                    if len(parts) >= 3:
                        task_id = int(parts[1])
                        phase_results[task_id] = results
            
            if phase_results:
                max_task = max(phase_results.keys())
                results_matrix = np.zeros((max_task, max_task))
                
                for i in range(1, max_task + 1):
                    if i in phase_results:
                        for j in range(1, max_task + 1):
                            task_key = f'task_{j}'
                            if task_key in phase_results[i]:
                                results_matrix[i-1][j-1] = phase_results[i][task_key]['accuracy']
                
                forgetting_matrix = calculate_forgetting_metrics(results_matrix)
                
                # Calculate average forgetting per task
                for i in range(1, max_task):
                    task_forgetting = [forgetting_matrix[i][j] for j in range(i) if forgetting_matrix[i][j] > 0]
                    if task_forgetting:
                        avg_f = np.mean(task_forgetting)
                        if phase == 1:
                            phase1_forgetting[i+1] = avg_f
                        else:
                            phase2_forgetting[i+1] = avg_f
        
        # Compare and display results
        comparison_data = []
        for task_id in sorted(set(phase1_forgetting.keys()) | set(phase2_forgetting.keys())):
            p1_f = phase1_forgetting.get(task_id, 0)
            p2_f = phase2_forgetting.get(task_id, 0)
            improvement = p1_f - p2_f  # Positive means less forgetting in phase 2
            
            comparison_data.append({
                'Task': task_id,
                'Phase1_Forgetting': p1_f,
                'Phase2_Forgetting': p2_f,
                'Improvement': improvement,
                'Improvement_Pct': (improvement / p1_f * 100) if p1_f > 0 else 0
            })
            
            print(f"Task {task_id}: Phase1={p1_f:.4f}, Phase2={p2_f:.4f}, Cải thiện={improvement:+.4f}")
        
        if comparison_data:
            df_forgetting_comparison = pd.DataFrame(comparison_data)
            df_forgetting_comparison.to_csv('evaluation_results/forgetting_comparison_phases.csv', index=False)
            print(f"\n💾 Đã lưu so sánh forgetting: evaluation_results/forgetting_comparison_phases.csv")
    
    def analyze_replay_effectiveness(self):
        """Phân tích hiệu quả của replay strategy thông qua so sánh Phase 1 vs Phase 2"""
        print("\n" + "="*60)
        print("🔄 PHÂN TÍCH HIỆU QUẢ REPLAY STRATEGY")
        print("="*60)
        
        # Phase 2 bao gồm replay strategy, Phase 1 không có
        print("📝 Giả định: Phase 2 sử dụng replay strategy, Phase 1 không sử dụng")
        
        replay_effectiveness = []
        
        # Analyze improvement from Phase 1 to Phase 2 on previous tasks
        for task_id in range(2, 5):  # Tasks 2-4 (có previous tasks để replay)
            phase1_checkpoint = f'task_{task_id}_phase1_best.ckpt'
            phase2_checkpoint = f'task_{task_id}_phase2_best.ckpt'
            
            if phase1_checkpoint in self.results and phase2_checkpoint in self.results:
                phase1_results = self.results[phase1_checkpoint]
                phase2_results = self.results[phase2_checkpoint]
                
                print(f"\n📊 Task {task_id} - Hiệu quả Replay:")
                print("-" * 40)
                
                # Analyze performance on previous tasks (replay targets)
                prev_task_improvements = []
                current_task_change = None
                
                for eval_task_id in range(1, task_id + 1):
                    eval_task_key = f'task_{eval_task_id}'
                    
                    if eval_task_key in phase1_results and eval_task_key in phase2_results:
                        p1_acc = phase1_results[eval_task_key]['accuracy']
                        p2_acc = phase2_results[eval_task_key]['accuracy']
                        improvement = p2_acc - p1_acc
                        
                        if eval_task_id < task_id:  # Previous task (replay target)
                            prev_task_improvements.append(improvement)
                            print(f"  {eval_task_key} (replay): {p1_acc:.4f} → {p2_acc:.4f} ({improvement:+.4f})")
                        else:  # Current task
                            current_task_change = improvement
                            print(f"  {eval_task_key} (current): {p1_acc:.4f} → {p2_acc:.4f} ({improvement:+.4f})")
                
                if prev_task_improvements:
                    avg_replay_improvement = np.mean(prev_task_improvements)
                    max_replay_improvement = max(prev_task_improvements)
                    min_replay_improvement = min(prev_task_improvements)
                    
                    replay_effectiveness.append({
                        'Task': task_id,
                        'Num_Replay_Tasks': len(prev_task_improvements),
                        'Avg_Replay_Improvement': avg_replay_improvement,
                        'Max_Replay_Improvement': max_replay_improvement,
                        'Min_Replay_Improvement': min_replay_improvement,
                        'Current_Task_Change': current_task_change,
                        'Replay_Success_Rate': sum(1 for x in prev_task_improvements if x > 0) / len(prev_task_improvements)
                    })
                    
                    print(f"  📈 Cải thiện trung bình trên previous tasks: {avg_replay_improvement:+.4f}")
                    print(f"  🎯 Tỷ lệ thành công replay: {replay_effectiveness[-1]['Replay_Success_Rate']:.2%}")
        
        if replay_effectiveness:
            df_replay = pd.DataFrame(replay_effectiveness)
            print(f"\n📊 Tổng kết hiệu quả Replay Strategy:")
            print(df_replay.to_string(index=False, float_format='%.4f'))
            
            # Calculate overall replay effectiveness
            overall_avg_improvement = np.mean([x['Avg_Replay_Improvement'] for x in replay_effectiveness])
            overall_success_rate = np.mean([x['Replay_Success_Rate'] for x in replay_effectiveness])
            
            print(f"\n🎯 Hiệu quả tổng thể của Replay Strategy:")
            print(f"  - Cải thiện trung bình: {overall_avg_improvement:+.4f}")
            print(f"  - Tỷ lệ thành công: {overall_success_rate:.2%}")
            
            # Save results
            os.makedirs('evaluation_results', exist_ok=True)
            df_replay.to_csv('evaluation_results/replay_effectiveness.csv', index=False)
            print(f"\n💾 Đã lưu phân tích replay: evaluation_results/replay_effectiveness.csv")
        else:
            print("❌ Không đủ dữ liệu để phân tích replay effectiveness!")
        
        return replay_effectiveness
    
    def create_visualizations(self):
        """Tạo các biểu đồ trực quan cho phase-based evaluation"""
        print("\n" + "="*60)
        print("📊 TẠO BIỂU ĐỒ TRỰC QUAN")
        print("="*60)
        
        os.makedirs('evaluation_results/plots', exist_ok=True)
        
        # 1. Phase comparison heatmaps
        for phase in [1, 2]:
            phase_results = {}
            for checkpoint_name, results in self.results.items():
                if f'phase{phase}' in checkpoint_name:
                    parts = checkpoint_name.split('_')
                    if len(parts) >= 3:
                        task_id = int(parts[1])
                        phase_results[task_id] = results
            
            if phase_results:
                max_task = max(phase_results.keys())
                results_matrix = np.zeros((max_task, max_task))
                
                for i in range(1, max_task + 1):
                    if i in phase_results:
                        for j in range(1, max_task + 1):
                            task_key = f'task_{j}'
                            if task_key in phase_results[i]:
                                results_matrix[i-1][j-1] = phase_results[i][task_key]['accuracy']
                
                # Heatmap for this phase
                plt.figure(figsize=(10, 8))
                sns.heatmap(results_matrix, 
                           annot=True, 
                           fmt='.3f', 
                           cmap='YlOrRd',
                           xticklabels=[f'Task {j+1}' for j in range(max_task)],
                           yticklabels=[f'After Task {i+1}' for i in range(max_task)])
                plt.title(f'Phase {phase} Performance Matrix\n(Accuracy scores)')
                plt.xlabel('Evaluation Task')
                plt.ylabel('Model Training Stage')
                plt.tight_layout()
                plt.savefig(f'evaluation_results/plots/performance_heatmap_phase{phase}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. Phase comparison side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        for phase_idx, phase in enumerate([1, 2]):
            phase_results = {}
            for checkpoint_name, results in self.results.items():
                if f'phase{phase}' in checkpoint_name:
                    parts = checkpoint_name.split('_')
                    if len(parts) >= 3:
                        task_id = int(parts[1])
                        phase_results[task_id] = results
            
            if phase_results:
                max_task = max(phase_results.keys())
                results_matrix = np.zeros((max_task, max_task))
                
                for i in range(1, max_task + 1):
                    if i in phase_results:
                        for j in range(1, max_task + 1):
                            task_key = f'task_{j}'
                            if task_key in phase_results[i]:
                                results_matrix[i-1][j-1] = phase_results[i][task_key]['accuracy']
                
                ax = ax1 if phase_idx == 0 else ax2
                sns.heatmap(results_matrix, 
                           annot=True, 
                           fmt='.3f', 
                           cmap='YlOrRd',
                           xticklabels=[f'Task {j+1}' for j in range(max_task)],
                           yticklabels=[f'After Task {i+1}' for i in range(max_task)],
                           ax=ax)
                ax.set_title(f'Phase {phase} Performance Matrix')
                ax.set_xlabel('Evaluation Task')
                ax.set_ylabel('Model Training Stage')
        
        plt.tight_layout()
        plt.savefig('evaluation_results/plots/phase_comparison_heatmaps.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Learning curves comparison
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange']
        markers = ['o', 's', '^', 'D']
        
        for task_id in range(1, 5):  # Tasks 1-4
            for phase in [1, 2]:
                task_performance = []
                training_stages = []
                
                # Collect performance for this task across different training stages
                for stage_task in range(task_id, 5):  # From when task was introduced
                    checkpoint_name = f'task_{stage_task}_phase{phase}_best.ckpt'
                    if checkpoint_name in self.results:
                        task_key = f'task_{task_id}'
                        if task_key in self.results[checkpoint_name]:
                            task_performance.append(self.results[checkpoint_name][task_key]['accuracy'])
                            training_stages.append(stage_task + (phase - 1) * 0.5)  # Offset phases
                
                if task_performance:
                    linestyle = '-' if phase == 1 else '--'
                    alpha = 0.7 if phase == 1 else 1.0
                    label = f'Task {task_id} Phase {phase}'
                    
                    plt.plot(training_stages, task_performance, 
                            marker=markers[task_id-1], 
                            color=colors[task_id-1],
                            linestyle=linestyle,
                            alpha=alpha,
                            label=label, 
                            linewidth=2,
                            markersize=8)
        
        plt.xlabel('Training Stage (Task + Phase offset)')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves: Performance Comparison Between Phases')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_results/plots/learning_curves_phase_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Phase improvement bar chart
        phase_improvements = []
        task_labels = []
        
        for task_id in range(1, 5):
            phase1_checkpoint = f'task_{task_id}_phase1_best.ckpt'
            phase2_checkpoint = f'task_{task_id}_phase2_best.ckpt'
            
            if phase1_checkpoint in self.results and phase2_checkpoint in self.results:
                # Calculate average improvement across all evaluation tasks
                improvements = []
                for eval_task_id in range(1, task_id + 1):
                    eval_task_key = f'task_{eval_task_id}'
                    if (eval_task_key in self.results[phase1_checkpoint] and 
                        eval_task_key in self.results[phase2_checkpoint]):
                        p1_acc = self.results[phase1_checkpoint][eval_task_key]['accuracy']
                        p2_acc = self.results[phase2_checkpoint][eval_task_key]['accuracy']
                        improvements.append(p2_acc - p1_acc)
                
                if improvements:
                    avg_improvement = np.mean(improvements)
                    phase_improvements.append(avg_improvement)
                    task_labels.append(f'Task {task_id}')
        
        if phase_improvements:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(task_labels, phase_improvements, 
                          color=['green' if x > 0 else 'red' for x in phase_improvements],
                          alpha=0.7)
            
            # Add value labels on bars
            for bar, improvement in zip(bars, phase_improvements):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{improvement:+.3f}',
                        ha='center', va='bottom')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Training Task')
            plt.ylabel('Average Accuracy Improvement (Phase 2 - Phase 1)')
            plt.title('Phase 2 Improvement Over Phase 1')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('evaluation_results/plots/phase_improvement_bars.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Đã tạo biểu đồ:")
        print("  - evaluation_results/plots/performance_heatmap_phase1.png")
        print("  - evaluation_results/plots/performance_heatmap_phase2.png")
        print("  - evaluation_results/plots/phase_comparison_heatmaps.png")
        print("  - evaluation_results/plots/learning_curves_phase_comparison.png")
        print("  - evaluation_results/plots/phase_improvement_bars.png")
    
    def generate_summary_report(self):
        """Tạo báo cáo tổng kết"""
        print("\n" + "="*60)
        print("📋 TẠO BÁO CÁO TỔNG KẾT")
        print("="*60)
        
        # Collect all analysis results
        summary = {
            'total_checkpoints': len(self.results),
            'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tasks_evaluated': 5,
            'metrics_used': ['accuracy', 'precision', 'recall', 'f1_score', 'mcc']
        }
        
        # Calculate overall statistics
        all_accuracies = []
        for checkpoint_results in self.results.values():
            for task_results in checkpoint_results.values():
                all_accuracies.append(task_results['accuracy'])
        
        if all_accuracies:
            summary.update({
                'mean_accuracy': np.mean(all_accuracies),
                'std_accuracy': np.std(all_accuracies),
                'min_accuracy': np.min(all_accuracies),
                'max_accuracy': np.max(all_accuracies)
            })
        
        # Save summary
        with open('evaluation_results/summary_report.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("📊 Thống kê tổng quan:")
        print(f"  - Tổng số checkpoint: {summary['total_checkpoints']}")
        print(f"  - Số task đánh giá: {summary['tasks_evaluated']}")
        if all_accuracies:
            print(f"  - Accuracy trung bình: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
            print(f"  - Accuracy cao nhất: {summary['max_accuracy']:.4f}")
            print(f"  - Accuracy thấp nhất: {summary['min_accuracy']:.4f}")
        
        print(f"\n💾 Đã lưu báo cáo tổng kết: evaluation_results/summary_report.json")
        
        return summary

def check_checkpoint_structure(checkpoint_dir):
    """Check and display checkpoint file structure"""
    print("📋 KIỂM TRA CẤU TRÚC CHECKPOINT FILES")
    print("="*50)
    
    location = checkpoint_dir
    found_files = []
    
    if os.path.exists(location):
        print(f"📁 Kiểm tra thư mục: {location}")
        try:
            files = os.listdir(location)
            checkpoint_files = [f for f in files if f.endswith('.ckpt') and 'phase' in f]
            if checkpoint_files:
                print(f"  ✅ Tìm thấy {len(checkpoint_files)} checkpoint files:")
                for f in sorted(checkpoint_files):
                    print(f"    - {f}")
                    found_files.append(os.path.join(location, f))
            else:
                print(f"  ❌ Không tìm thấy checkpoint files")
        except Exception as e:
            print(f"  ❌ Lỗi đọc thư mục: {e}")
    else:
        print(f"📁 Thư mục không tồn tại: {location}")
    
    if found_files:
        print(f"\n✅ Tổng cộng tìm thấy {len(found_files)} checkpoint files")
        return True
    else:
        print(f"\n❌ Không tìm thấy checkpoint files nào!")
        print("💡 Hãy đảm bảo các files đã được giải nén với tên:")
        print("   - task_1_phase1_best.ckpt, task_1_phase2_best.ckpt")
        print("   - task_2_phase1_best.ckpt, task_2_phase2_best.ckpt")
        print("   - task_3_phase1_best.ckpt, task_3_phase2_best.ckpt")
        print("   - task_4_phase1_best.ckpt, task_4_phase2_best.ckpt")
        return False

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Continual Learning Performance')
    parser.add_argument('--data_dir', default=default_data_dir, help='Directory containing CSV data files')
    parser.add_argument('--checkpoint_dir', default=default_checkpoint_dir, help='Directory containing checkpoint files')
    
    args = parser.parse_args()
    
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ PHASE-BASED CONTINUAL LEARNING")
    print("="*60)
    
    # Check checkpoint structure first
    if not check_checkpoint_structure(args.checkpoint_dir):
        return
    
    # Initialize evaluator
    evaluator = ContinualLearningEvaluator(args.data_dir, args.checkpoint_dir)
    
    # Run all evaluations
    evaluator.evaluate_all_checkpoints()
    
    if not evaluator.results:
        print("❌ Không có kết quả đánh giá nào!")
        print("💡 Hãy đảm bảo các checkpoint files đã được giải nén đúng vị trí:")
        print("   - best/best/task_X_phaseY_best.ckpt")
        print("   - hoặc trong thư mục gốc: task_X_phaseY_best.ckpt")
        return
    
    # Analyze results
    print(f"\n📊 Đã tải {len(evaluator.results)} checkpoint(s), bắt đầu phân tích...")
    
    # Phase comparison analysis
    comparison_data, improvement_summary = evaluator.analyze_phase_comparison()
    
    # Catastrophic forgetting analysis
    evaluator.analyze_catastrophic_forgetting()
    
    # Replay effectiveness analysis
    evaluator.analyze_replay_effectiveness()
    
    # Create visualizations
    evaluator.create_visualizations()
    
    # Generate summary report
    evaluator.generate_summary_report()
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH ĐÁNH GIÁ PHASE-BASED EVALUATION!")
    print("📁 Kết quả được lưu trong thư mục: evaluation_results/")
    print("\n📋 Các file kết quả:")
    print("  📊 Dữ liệu:")
    print("    - phase_comparison_detailed.csv")
    print("    - phase_improvement_summary.csv")
    print("    - results_matrix_phase1.csv & results_matrix_phase2.csv")
    print("    - forgetting_matrix_phase1.csv & forgetting_matrix_phase2.csv")
    print("    - forgetting_comparison_phases.csv")
    print("    - replay_effectiveness.csv")
    print("    - summary_report.json")
    print("  📈 Biểu đồ:")
    print("    - performance_heatmap_phase1.png & performance_heatmap_phase2.png")
    print("    - phase_comparison_heatmaps.png")
    print("    - learning_curves_phase_comparison.png")
    print("    - phase_improvement_bars.png")
    print("="*60)

if __name__ == "__main__":
    main()