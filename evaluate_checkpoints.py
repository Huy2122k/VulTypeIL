import ast
import os
import warnings
# ==================================================SPECIFIC LIB==============================
from collections import Counter, namedtuple

# import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
# from datasets import Dataset, load_dataset
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils import InputExample
from openprompt.plms import add_special_tokens
from openprompt.plms.seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper
# from openprompt.plms import load_plm
# from code_t5 import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer, MixedTemplate
from scipy.spatial import distance
from sklearn.metrics import (accuracy_score, matthews_corrcoef,
                             precision_recall_fscore_support)
from tqdm.auto import tqdm
from transformers import (AdamW, RobertaTokenizer, T5Config,
                          T5ForConditionalGeneration,
                          get_linear_schedule_with_warmup)
from vulcom import (classes, list_available_checkpoints, load_plm,
                    load_task_checkpoint, read_prompt_examples, test,
                    test_paths)

# Thêm thư viện cho visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import json

# Cấu hình
use_cuda = True
batch_size = 4
max_seq_l = 512
model_name = "t5"
pretrainedmodel_path = "Salesforce/codet5-base"


def setup_model():
    """Khởi tạo model và các thành phần cần thiết."""
    # Load model và tokenizer
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)
    
    # Định nghĩa template
    template_text = ('Given the following vulnerable code snippet: {"placeholder":"text_a"} '
                     'and its vulnerability description: {"placeholder":"text_b"}, '
                     'classify the vulnerability type as: {"mask"}.')
    
    mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text, model=plm)
    
    # Định nghĩa verbalizer
    myverbalizer = ManualVerbalizer(tokenizer, classes=classes, label_words={
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
    
    # Tạo prompt model
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    if use_cuda:
        prompt_model = prompt_model.cuda()
    
    return prompt_model, mytemplate, tokenizer, WrapperClass

def create_test_dataloaders(mytemplate, tokenizer, WrapperClass):
    """Tạo các dataloader cho test."""
    test_dataloaders = []
    for i, test_path in enumerate(test_paths):
        dataloader = PromptDataLoader(
            dataset=read_prompt_examples(test_path),
            template=mytemplate,
            tokenizer=tokenizer, 
            tokenizer_wrapper_class=WrapperClass, 
            max_seq_length=max_seq_l,
            batch_size=batch_size, 
            shuffle=False,  # Không shuffle cho evaluation
            teacher_forcing=False, 
            predict_eos_token=False, 
            truncate_method="head",
            decoder_max_length=3
        )
        test_dataloaders.append(dataloader)
    return test_dataloaders

def evaluate_checkpoint_on_all_tasks(prompt_model, checkpoint_path, test_dataloaders):
    """Đánh giá một checkpoint trên tất cả các task."""
    print(f"\n{'='*60}")
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        prompt_model.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device('cuda:0'))
        )
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    # Test trên tất cả các task
    results = {}
    for task_id, dataloader in enumerate(test_dataloaders, 1):
        print(f"\nTesting on Task {task_id}...")
        acc, precisionma, recallma, f1wei, f1ma = test(
            prompt_model, dataloader, 
            f'checkpoint_eval_task{task_id}_{os.path.basename(checkpoint_path).replace(".ckpt", "")}'
        )
        results[f'task_{task_id}'] = {
            'accuracy': acc,
            'precision_macro': precisionma,
            'recall_macro': recallma,
            'f1_weighted': f1wei,
            'f1_macro': f1ma
        }
    
    return results

def print_continual_learning_explanation():
    """In giải thích về các metrics continual learning."""
    print("\n" + "="*80)
    print("CONTINUAL LEARNING METRICS EXPLANATION")
    print("="*80)
    
    print("\n🔍 FORGETTING MEASURE (F):")
    print("   • Đo mức độ mô hình 'quên' kiến thức cũ khi học task mới")
    print("   • Công thức: F_i = max_k(Acc_i,k) - Acc_i,final")
    print("   • F_i > 0: Mô hình bị quên kiến thức task i")
    print("   • F_i = 0: Không có forgetting")
    print("   • F_i < 0: Hiệu năng task i được cải thiện (hiếm gặp)")
    
    print("\n🔄 BACKWARD TRANSFER (BWT):")
    print("   • Đo xem việc học task mới có ảnh hưởng đến task cũ không")
    print("   • BWT > 0: Học task mới giúp cải thiện task cũ (positive transfer)")
    print("   • BWT = 0: Không có ảnh hưởng")
    print("   • BWT < 0: Học task mới làm giảm hiệu năng task cũ (negative transfer)")
    
    print("\n⚡ FORWARD TRANSFER (FWT):")
    print("   • Đo xem kiến thức từ task trước có giúp học task mới nhanh hơn không")
    print("   • FWT > 0: Kiến thức cũ giúp ích cho task mới")
    print("   • FWT = 0: Không có transfer")
    print("   • FWT < 0: Kiến thức cũ cản trở việc học task mới")
    
    print("\n📊 ACCURACY MATRIX:")
    print("   • Hàng i, cột j: Accuracy của task i sau khi học xong task j")
    print("   • Đường chéo: Hiệu năng ngay sau khi học xong task đó")
    print("   • Dưới đường chéo: Hiệu năng task cũ sau khi học task mới")
    print("   • Trên đường chéo: Hiệu năng task chưa học (thường = 0)")

def main():
    """Hàm chính để chạy evaluation."""
    print("Checkpoint Evaluation Tool with Continual Learning Analysis")
    print("="*70)
    
    # Hiển thị các checkpoint có sẵn
    print("\nAvailable checkpoints:")
    checkpoints = list_available_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found. Please run training first.")
        return
    
    # Setup model
    print("\nSetting up model...")
    prompt_model, mytemplate, tokenizer, WrapperClass = setup_model()
    
    # Tạo test dataloaders
    print("Creating test dataloaders...")
    test_dataloaders = create_test_dataloaders(mytemplate, tokenizer, WrapperClass)
    
    # Tùy chọn evaluation
    print("\nEvaluation options:")
    print("1. Evaluate all checkpoints (with Continual Learning analysis)")
    print("2. Evaluate specific checkpoint")
    print("3. Evaluate final checkpoints only (with Continual Learning analysis)")
    print("4. Show Continual Learning metrics explanation")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # Evaluate tất cả checkpoints
        all_results = {}
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join('model/checkpoints', checkpoint)
            results = evaluate_checkpoint_on_all_tasks(prompt_model, checkpoint_path, test_dataloaders)
            all_results[checkpoint] = results
        
        # Lưu kết quả tổng hợp và tính continual learning metrics
        cl_metrics = save_comprehensive_results(all_results)
        
    elif choice == "2":
        # Evaluate checkpoint cụ thể
        print("\nAvailable checkpoints:")
        for i, checkpoint in enumerate(checkpoints):
            print(f"{i+1}. {checkpoint}")
        
        try:
            idx = int(input("Select checkpoint number: ")) - 1
            if 0 <= idx < len(checkpoints):
                checkpoint_path = os.path.join('model/checkpoints', checkpoints[idx])
                evaluate_checkpoint_on_all_tasks(prompt_model, checkpoint_path, test_dataloaders)
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")
            
    elif choice == "3":
        # Evaluate chỉ final checkpoints
        final_checkpoints = [cp for cp in checkpoints if 'final' in cp]
        if final_checkpoints:
            all_results = {}
            for checkpoint in final_checkpoints:
                checkpoint_path = os.path.join('model/checkpoints', checkpoint)
                results = evaluate_checkpoint_on_all_tasks(prompt_model, checkpoint_path, test_dataloaders)
                all_results[checkpoint] = results
            cl_metrics = save_comprehensive_results(all_results, "final_checkpoints")
        else:
            print("No final checkpoints found.")
    
    elif choice == "4":
        # Hiển thị giải thích về continual learning metrics
        print_continual_learning_explanation()
        return
    
    else:
        print("Invalid choice.")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)

def calculate_continual_learning_metrics(all_results):
    """
    Tính toán các metrics cho Continual Learning:
    - Forgetting Measure (F)
    - Backward Transfer (BWT) 
    - Forward Transfer (FWT)
    """
    # Sắp xếp checkpoints theo thứ tự task
    checkpoints = sorted(all_results.keys())
    num_tasks = len(test_paths)
    
    # Tạo ma trận accuracy: [task_id][checkpoint_id] = accuracy
    acc_matrix = np.zeros((num_tasks, len(checkpoints)))
    
    for checkpoint_idx, checkpoint in enumerate(checkpoints):
        for task_id in range(1, num_tasks + 1):
            task_key = f'task_{task_id}'
            if task_key in all_results[checkpoint]:
                acc_matrix[task_id - 1][checkpoint_idx] = all_results[checkpoint][task_key]['accuracy']
    
    # 1. Forgetting Measure (F)
    forgetting_measures = []
    for task_id in range(num_tasks):
        # Tìm accuracy tốt nhất của task này qua các checkpoint
        max_acc = np.max(acc_matrix[task_id, :])
        # Accuracy cuối cùng của task này
        final_acc = acc_matrix[task_id, -1]
        # Forgetting = max_acc - final_acc
        forgetting = max_acc - final_acc
        forgetting_measures.append(forgetting)
    
    avg_forgetting = np.mean(forgetting_measures)
    
    # 2. Backward Transfer (BWT)
    # BWT đo xem học task mới có làm giảm performance task cũ không
    bwt_values = []
    for task_id in range(num_tasks - 1):  # Không tính task cuối
        # Performance của task này sau khi học xong tất cả task
        final_perf = acc_matrix[task_id, -1]
        # Performance của task này ngay sau khi học xong task này
        after_task_perf = acc_matrix[task_id, task_id]
        bwt = final_perf - after_task_perf
        bwt_values.append(bwt)
    
    avg_bwt = np.mean(bwt_values) if bwt_values else 0
    
    # 3. Forward Transfer (FWT)
    # FWT đo xem kiến thức từ task trước có giúp task mới không
    fwt_values = []
    for task_id in range(1, num_tasks):  # Bắt đầu từ task 2
        # Performance của task này khi mới bắt đầu học (có kiến thức từ task trước)
        initial_perf = acc_matrix[task_id, task_id - 1] if task_id > 0 else 0
        # Performance baseline (giả sử là 0 hoặc random performance)
        baseline_perf = 1.0 / len(classes)  # Random performance
        fwt = initial_perf - baseline_perf
        fwt_values.append(fwt)
    
    avg_fwt = np.mean(fwt_values) if fwt_values else 0
    
    return {
        'forgetting_measures': forgetting_measures,
        'avg_forgetting': avg_forgetting,
        'bwt_values': bwt_values,
        'avg_bwt': avg_bwt,
        'fwt_values': fwt_values,
        'avg_fwt': avg_fwt,
        'acc_matrix': acc_matrix,
        'checkpoints': checkpoints
    }

def create_visualization_curves(all_results, cl_metrics, results_dir):
    """Tạo các biểu đồ visualization cho continual learning."""
    
    # Thiết lập style
    plt.style.use('seaborn-v0_8')
    fig_dir = os.path.join(results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    num_tasks = len(test_paths)
    checkpoints = cl_metrics['checkpoints']
    acc_matrix = cl_metrics['acc_matrix']
    
    # 1. F1 vs Number of tasks
    plt.figure(figsize=(12, 8))
    
    # Tính F1 macro cho mỗi checkpoint
    f1_scores = []
    mcc_scores = []
    
    for checkpoint in checkpoints:
        f1_values = []
        mcc_values = []
        for task_id in range(1, num_tasks + 1):
            task_key = f'task_{task_id}'
            if task_key in all_results[checkpoint]:
                f1_values.append(all_results[checkpoint][task_key]['f1_macro'])
                # Tính MCC từ accuracy (approximation)
                acc = all_results[checkpoint][task_key]['accuracy']
                mcc_approx = 2 * acc - 1  # Rough approximation
                mcc_values.append(mcc_approx)
        
        f1_scores.append(np.mean(f1_values) if f1_values else 0)
        mcc_scores.append(np.mean(mcc_values) if mcc_values else 0)
    
    # Plot F1 vs Number of tasks
    plt.subplot(2, 2, 1)
    task_numbers = list(range(1, len(checkpoints) + 1))
    plt.plot(task_numbers, f1_scores, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Number of Tasks')
    plt.ylabel('Average F1 Score')
    plt.title('F1 Score vs Number of Tasks')
    plt.grid(True, alpha=0.3)
    
    # Plot MCC vs Number of tasks
    plt.subplot(2, 2, 2)
    plt.plot(task_numbers, mcc_scores, 'r-s', linewidth=2, markersize=8)
    plt.xlabel('Number of Tasks')
    plt.ylabel('Average MCC Score')
    plt.title('MCC Score vs Number of Tasks')
    plt.grid(True, alpha=0.3)
    
    # 3. Per-task accuracy matrix (heatmap)
    plt.subplot(2, 2, 3)
    sns.heatmap(acc_matrix, 
                xticklabels=[f'After Task {i+1}' for i in range(len(checkpoints))],
                yticklabels=[f'Task {i+1}' for i in range(num_tasks)],
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Accuracy'})
    plt.title('Per-Task Accuracy Matrix')
    plt.xlabel('Training Progress')
    plt.ylabel('Task ID')
    
    # 4. Continual Learning Metrics Summary
    plt.subplot(2, 2, 4)
    metrics_names = ['Avg Forgetting', 'Backward Transfer', 'Forward Transfer']
    metrics_values = [cl_metrics['avg_forgetting'], cl_metrics['avg_bwt'], cl_metrics['avg_fwt']]
    colors = ['red', 'orange', 'green']
    
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    plt.ylabel('Score')
    plt.title('Continual Learning Metrics')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Thêm giá trị lên các bar
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Lưu figure
    plt.savefig(os.path.join(fig_dir, 'continual_learning_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Tạo biểu đồ riêng cho accuracy matrix với kích thước lớn hơn
    plt.figure(figsize=(12, 8))
    sns.heatmap(acc_matrix, 
                xticklabels=[f'After Task {i+1}' for i in range(len(checkpoints))],
                yticklabels=[f'Task {i+1}' for i in range(num_tasks)],
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Accuracy'},
                square=True)
    plt.title('Per-Task Accuracy Matrix - Detailed View', fontsize=16)
    plt.xlabel('Training Progress (After Learning Each Task)', fontsize=12)
    plt.ylabel('Task ID', fontsize=12)
    
    # Thêm đường chéo để highlight diagonal
    for i in range(min(num_tasks, len(checkpoints))):
        rect = Rectangle((i, i), 1, 1, linewidth=3, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'accuracy_matrix_detailed.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization curves saved to: {fig_dir}")

def save_comprehensive_results(all_results, suffix="comprehensive"):
    """Lưu kết quả tổng hợp vào file CSV và tính toán continual learning metrics."""
    results_dir = "results/checkpoint_evaluation"
    os.makedirs(results_dir, exist_ok=True)
    
    # Tạo DataFrame cho kết quả
    data = []
    for checkpoint, tasks_results in all_results.items():
        for task, metrics in tasks_results.items():
            row = {
                'checkpoint': checkpoint,
                'task': task,
                **metrics
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    output_file = os.path.join(results_dir, f"evaluation_{suffix}.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✓ Comprehensive results saved to: {output_file}")
    
    # Tạo summary
    summary_data = []
    for checkpoint in all_results.keys():
        avg_metrics = {}
        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_weighted', 'f1_macro']:
            values = [all_results[checkpoint][task][metric] for task in all_results[checkpoint].keys()]
            avg_metrics[f'avg_{metric}'] = sum(values) / len(values)
        
        summary_row = {
            'checkpoint': checkpoint,
            **avg_metrics
        }
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(results_dir, f"summary_{suffix}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Summary results saved to: {summary_file}")
    
    # Tính toán Continual Learning metrics
    print("\n" + "="*60)
    print("CONTINUAL LEARNING ANALYSIS")
    print("="*60)
    
    cl_metrics = calculate_continual_learning_metrics(all_results)
    
    # In kết quả
    print(f"\n📊 CONTINUAL LEARNING METRICS:")
    print(f"   • Average Forgetting Measure (F): {cl_metrics['avg_forgetting']:.4f}")
    print(f"   • Backward Transfer (BWT): {cl_metrics['avg_bwt']:.4f}")
    print(f"   • Forward Transfer (FWT): {cl_metrics['avg_fwt']:.4f}")
    
    print(f"\n📈 PER-TASK FORGETTING:")
    for i, forgetting in enumerate(cl_metrics['forgetting_measures']):
        print(f"   • Task {i+1}: {forgetting:.4f}")
    
    # Lưu continual learning metrics
    cl_results = {
        'avg_forgetting': cl_metrics['avg_forgetting'],
        'avg_bwt': cl_metrics['avg_bwt'],
        'avg_fwt': cl_metrics['avg_fwt'],
        'per_task_forgetting': cl_metrics['forgetting_measures'],
        'bwt_values': cl_metrics['bwt_values'],
        'fwt_values': cl_metrics['fwt_values']
    }
    
    cl_file = os.path.join(results_dir, f"continual_learning_metrics_{suffix}.json")
    with open(cl_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        cl_results_json = {}
        for key, value in cl_results.items():
            if isinstance(value, np.ndarray):
                cl_results_json[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.float64):
                cl_results_json[key] = [float(v) for v in value]
            else:
                cl_results_json[key] = value
        json.dump(cl_results_json, f, indent=2)
    
    print(f"✓ Continual learning metrics saved to: {cl_file}")
    
    # Tạo visualization curves
    create_visualization_curves(all_results, cl_metrics, results_dir)
    
    # Lưu accuracy matrix
    acc_matrix_df = pd.DataFrame(
        cl_metrics['acc_matrix'],
        columns=[f'after_task_{i+1}' for i in range(len(cl_metrics['checkpoints']))],
        index=[f'task_{i+1}' for i in range(len(test_paths))]
    )
    matrix_file = os.path.join(results_dir, f"accuracy_matrix_{suffix}.csv")
    acc_matrix_df.to_csv(matrix_file)
    print(f"✓ Accuracy matrix saved to: {matrix_file}")
    
    return cl_metrics

if __name__ == "__main__":
    main()