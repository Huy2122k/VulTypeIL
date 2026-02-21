import argparse
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

# ===== IMPORT CÁC MODULE REPLAY NÂNG CAO VỚI TASK-AWARE STRATEGY =====
from replay_integration import upgrade_existing_replay_function, log_replay_improvements
from replay_config import create_config
from enhanced_replay_strategy import create_task_aware_replay_selector, analyze_forgetting_pattern

ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model','wrapper'))

def load_plm(model_name, model_path):
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

# ==================================================END LIB==============================

warnings.filterwarnings("ignore")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train VulTypeIL model with Task-Aware Enhanced Replay")
parser.add_argument('--data_dir', type=str, default='incremental_tasks_csv', help='Directory containing the data files')
parser.add_argument('--checkpoint_dir', type=str, default='model', help='Directory to save/load checkpoints')
parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
parser.add_argument('--pretrained_model_path', type=str, default='Salesforce/codet5-base', help='Path to pre-trained model')
parser.add_argument('--model_name', type=str, default='t5', help='Model name')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--ewc_lambda', type=float, default=0.4, help='EWC regularization term weight')
parser.add_argument('--num_tasks', type=int, default=5, help='Number of tasks')
parser.add_argument('--max_seq_l', type=int, default=512, help='Maximum sequence length')
parser.add_argument('--num_class', type=int, default=23, help='Number of classes')
parser.add_argument('--use_cuda', action='store_true', default=True, help='Use CUDA')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--patience', type=int, default=5, help='patience')
parser.add_argument('--replay_ratio', type=float, default=0.2, help='Replay buffer ratio of total previous dataset (default=0.2 = 20%)')
parser.add_argument('--min_samples_per_class', type=int, default=2, help='Minimum samples per class in replay buffer')

# ===== THAM SỐ MỚI CHO TASK-AWARE REPLAY =====
parser.add_argument('--replay_config_type', type=str, default='balanced', 
                   choices=['balanced', 'memory_efficient', 'quality_focused', 'fast'],
                   help='Loại cấu hình replay: balanced, memory_efficient, quality_focused, fast')
parser.add_argument('--task_decay_factor', type=float, default=0.8, 
                   help='Hệ số ưu tiên task cũ (0.5-0.9, cao hơn = ưu tiên task cũ nhiều hơn)')
parser.add_argument('--min_task_ratio', type=float, default=0.18, 
                   help='Tỷ lệ tối thiểu cho mỗi task (0.1-0.25)')
parser.add_argument('--similarity_threshold', type=float, default=0.85, 
                   help='Ngưỡng tương tự để lọc dư thừa ngữ nghĩa (0.0-1.0)')
parser.add_argument('--max_code_lines', type=int, default=10, 
                   help='Số dòng code tối đa để giữ lại sau tóm tắt')
parser.add_argument('--n_clusters', type=int, default=10, 
                   help='Số clusters cho ưu tiên replay')
parser.add_argument('--enable_gradient_importance', action='store_true', 
                   help='Bật gradient-based sample importance')

args = parser.parse_args()

# Set parameters from args
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = args.seed
batch_size = args.batch_size
num_class = args.num_class
max_seq_l = args.max_seq_l
lr = args.lr
num_epochs = args.num_epochs
use_cuda = args.use_cuda
model_name = args.model_name
pretrainedmodel_path = args.pretrained_model_path
early_stop_threshold = 10
ewc_lambda = args.ewc_lambda

# Define classes
classes = [
    'CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20', 'CWE-416',
    'CWE-190', 'CWE-200', 'CWE-120', 'CWE-399', 'CWE-401', 'CWE-264', 'CWE-772',
    'CWE-189', 'CWE-362', 'CWE-835', 'CWE-369', 'CWE-617', 'CWE-400', 'CWE-415',
    'CWE-122', 'CWE-770', 'CWE-22'
]

# Construct data paths
data_paths = [os.path.join(args.data_dir, f"task{i}_train.csv") for i in range(1, args.num_tasks + 1)]
test_paths = [os.path.join(args.data_dir, f"task{i}_test.csv") for i in range(1, args.num_tasks + 1)]
valid_paths = [os.path.join(args.data_dir, f"task{i}_valid.csv") for i in range(1, args.num_tasks + 1)]


# Define function to read examples
def read_prompt_examples(filename):
    examples = []
    data = pd.read_csv(filename).astype(str)
    desc = data['description'].tolist()
    code = data['abstract_func_before'].tolist()
    type = data['cwe_ids'].tolist()
    for idx in range(len(data)):
        # Convert CWE IDs to class index for classification
        cwe_list = ast.literal_eval(type[idx])
        # Take the first CWE ID and map it to class index
        if cwe_list and cwe_list[0] in classes:
            class_idx = classes.index(cwe_list[0])
        else:
            class_idx = 0  # Default to first class if not found
        
        examples.append(
            InputExample(
                guid=idx,
                text_a=' '.join(code[idx].split(' ')[:384]),
                text_b=' '.join(desc[idx].split(' ')[:64]),
                tgt_text=class_idx,  # Use integer label instead of string
            )
        )
    return examples


def read_and_merge_previous_datasets(current_index, data_paths, return_task_info=False):
    merged_data = pd.DataFrame()
    examples = []
    task_origins = []  # Track which task each sample comes from
    
    for i in range(current_index - 1):
        data = pd.read_csv(data_paths[i]).astype(str)
        task_id = i + 1  # Task IDs start from 1
        # Add task_id column to track origin
        data['task_origin'] = task_id
        merged_data = pd.concat([merged_data, data], ignore_index=True)
    
    desc = merged_data['description'].tolist()
    code = merged_data['abstract_func_before'].tolist()
    type = merged_data['cwe_ids'].tolist()
    task_origin_list = merged_data['task_origin'].tolist()
    
    for idx in range(len(merged_data)):
        # Convert CWE IDs to class index for classification
        cwe_list = ast.literal_eval(type[idx])
        # Take the first CWE ID and map it to class index
        if cwe_list and cwe_list[0] in classes:
            class_idx = classes.index(cwe_list[0])
        else:
            class_idx = 0  # Default to first class if not found
            
        examples.append(
            InputExample(
                guid=idx,
                text_a=' '.join(code[idx].split(' ')[:384]),
                text_b=' '.join(desc[idx].split(' ')[:64]),
                tgt_text=class_idx,  # Use integer label instead of string
            )
        )
        task_origins.append(int(task_origin_list[idx]))
    
    if return_task_info:
        return examples, task_origins
    return examples


class OnlineEWCWithFocalLabelSmoothLoss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.1, focal_alpha=1.0, focal_gamma=2.0, ewc_lambda=0.4, decay_factor=0.9):
        super(OnlineEWCWithFocalLabelSmoothLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ewc_lambda = ewc_lambda
        self.decay_factor = decay_factor
        self.fisher_dict = {}
        self.optpar_dict = {}

    def focal_label_smooth_ce_loss(self, logits, target, w=0.5):
        # Label Smoothing Cross Entropy Loss
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        pred = F.softmax(logits, dim=-1)
        log_probs = torch.log(pred)
        ce_loss = -torch.sum(target_smooth * log_probs, dim=-1).mean()

        # Focal Loss
        true_class_pred = pred.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = ((1 - true_class_pred) ** self.focal_gamma) * self.focal_alpha
        focal_loss = (focal_weight * (-torch.log(true_class_pred))).mean()

        # Combine Focal Loss and Label Smoothing Cross Entropy Loss
        combined_loss = w * focal_loss + (1 - w) * ce_loss
        return combined_loss

    def ewc_loss_online(self, prompt_model):
        # Calculate Online EWC loss using accumulated Fisher information and optimal parameters
        ewc_loss = 0
        for name, param in prompt_model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                optpar = self.optpar_dict[name]
                ewc_loss += (fisher * (param - optpar) ** 2).sum()
        return self.ewc_lambda * ewc_loss

    def update_fisher(self, prompt_model, dataloader):
        # Update Fisher information with current task's data and parameters
        current_fisher_dict = {}
        prompt_model.eval()

        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text']
            # Convert labels to tensor if needed
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels)
            labels = labels.cuda()
            loss = self.focal_label_smooth_ce_loss(logits, labels)
            prompt_model.zero_grad()
            loss.backward()

            for name, param in prompt_model.named_parameters():
                if param.grad is not None:
                    if name not in current_fisher_dict:
                        current_fisher_dict[name] = param.grad.data.clone().detach() ** 2
                    else:
                        current_fisher_dict[name] += param.grad.data.clone().detach() ** 2

        # Average Fisher information for the current task
        for name in current_fisher_dict:
            current_fisher_dict[name] /= len(dataloader)

        # Update global Fisher information with decay
        for name, fisher_value in current_fisher_dict.items():
            if name in self.fisher_dict:
                self.fisher_dict[name] = self.decay_factor * self.fisher_dict[name] + fisher_value
            else:
                self.fisher_dict[name] = fisher_value
            self.optpar_dict[name] = prompt_model.state_dict()[name].clone().detach()

    def forward(self, prompt_model, logits, target):
        # Combine Focal + Label Smoothing CE Loss and Online EWC Loss
        focal_label_smooth_loss = self.focal_label_smooth_ce_loss(logits, target)
        ewc_loss = self.ewc_loss_online(prompt_model)
        total_loss = focal_label_smooth_loss + ewc_loss
        return total_loss


# Define function to test the model
def test(prompt_model, test_dataloader, name):
    num_test_steps = len(test_dataloader)
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text']
            # Handle both tensor and list labels
            if torch.is_tensor(labels):
                alllabels.extend(labels.cpu().tolist())
            else:
                alllabels.extend(labels)
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        acc = accuracy_score(alllabels, allpreds)
        precisionwei, recallwei, f1wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted')
        precisionma, recallma, f1ma, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro')
        mcc = matthews_corrcoef(alllabels, allpreds)
        # Create results directory if it doesn't exist
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, "{}.pred.csv".format(name)), 'w', encoding='utf-8') as f, \
                open(os.path.join(args.results_dir, "{}.gold.csv".format(name)), 'w', encoding='utf-8') as f1:
            for ref, gold in zip(allpreds, alllabels):
                f.write(str(ref) + '\n')
                f1.write(str(gold) + '\n')
        print("acc: {}   precisionma: {}  recallma: {} recallwei: {} weighted-f1: {}  macro-f1: {} mcc: {}".format(acc,
                                                                                                                   precisionma,
                                                                                                                   recallma,
                                                                                                                   recallwei,
                                                                                                                   f1wei,
                                                                                                                   f1ma,
                                                                                                                   mcc))
    return acc, precisionma, recallma, f1wei, f1ma

# Two-phase training functions (giữ nguyên từ vul_main4.py)
def train_phase_one(prompt_model, train_dataloader, val_dataloader, optimizer1, optimizer2, scheduler1, scheduler2,
                    num_epochs, loss_func_no_ewc, task_id, patience=5):
    """Phase 1: Train with Focal Loss + Label Smoothing only (no EWC) to learn task-specific features."""
    best_val_loss = float('inf')
    patience_counter = 0
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))

    for epoch in range(num_epochs):
        prompt_model.train()
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text']
            # Convert labels to tensor if needed
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels)
            labels = labels.cuda()
            # Phase 1 loss (without EWC)
            loss = loss_func_no_ewc.focal_label_smooth_ce_loss(logits, labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()
            prompt_model.zero_grad()
            progress_bar.update(1)

        # Validation phase to check performance on validation data
        prompt_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_dataloader:
                if use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['tgt_text']
                # Convert labels to tensor if needed
                if not torch.is_tensor(labels):
                    labels = torch.tensor(labels)
                labels = labels.cuda()
                loss = loss_func_no_ewc.focal_label_smooth_ce_loss(logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the patience counter
            # Create directory if it doesn't exist
            os.makedirs(os.path.join(args.checkpoint_dir, 'best'), exist_ok=True)
            # Save checkpoint for specific task
            torch.save(prompt_model.state_dict(), os.path.join(args.checkpoint_dir, 'best', f'task_{task_id}_phase1_best.ckpt'))
            # Also save as general best for backward compatibility
            torch.save(prompt_model.state_dict(), os.path.join(args.checkpoint_dir, 'best', 'best.ckpt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


def train_phase_two(prompt_model, train_dataloader, val_dataloader, optimizer1, optimizer2, scheduler1, scheduler2,
                    num_epochs, loss_func_with_ewc, task_id, patience=5):
    """Phase 2: Train with Focal Loss + Label Smoothing + EWC to prevent forgetting of previous tasks."""
    best_val_loss = float('inf')
    patience_counter = 0
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))

    for epoch in range(num_epochs):
        prompt_model.train()
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['tgt_text']
            # Convert labels to tensor if needed
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels)
            labels = labels.cuda()
            # Phase 2 loss (with EWC)
            loss = loss_func_with_ewc(prompt_model, logits, labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()
            prompt_model.zero_grad()
            progress_bar.update(1)

        # Validation phase to check performance on validation data
        prompt_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_dataloader:
                if use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['tgt_text']
                # Convert labels to tensor if needed
                if not torch.is_tensor(labels):
                    labels = torch.tensor(labels)
                labels = labels.cuda()
                loss = loss_func_with_ewc(prompt_model, logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the patience counter
            # Create directory if it doesn't exist
            os.makedirs(os.path.join(args.checkpoint_dir, 'best'), exist_ok=True)
            # Save checkpoint for specific task
            torch.save(prompt_model.state_dict(), os.path.join(args.checkpoint_dir, 'best', f'task_{task_id}_phase2_best.ckpt'))
            # Also save as general best for backward compatibility
            torch.save(prompt_model.state_dict(), os.path.join(args.checkpoint_dir, 'best', 'best.ckpt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


# Utility functions for checkpoint management (giữ nguyên)
def save_task_checkpoint(prompt_model, task_id, phase="final"):
    """Save checkpoint for a specific task and phase."""
    os.makedirs(os.path.join(args.checkpoint_dir, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoints', f'task_{task_id}_{phase}.ckpt')
    torch.save(prompt_model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path

def load_task_checkpoint(prompt_model, task_id, phase="final"):
    """Load checkpoint for a specific task and phase."""
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoints', f'task_{task_id}_{phase}.ckpt')
    if os.path.exists(checkpoint_path):
        prompt_model.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device('cuda:0'))
        )
        print(f"Loaded checkpoint: {checkpoint_path}")
        return True
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return False

def list_available_checkpoints():
    """List all available checkpoints."""
    checkpoint_dir_path = os.path.join(args.checkpoint_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir_path):
        checkpoints = [f for f in os.listdir(checkpoint_dir_path) if f.endswith('.ckpt')]
        print("Available checkpoints:")
        for checkpoint in sorted(checkpoints):
            print(f"  - {checkpoint}")
        return checkpoints
    else:
        print("No checkpoints directory found.")
        return []

# Initialize EWC and non-EWC loss functions
loss_func_no_ewc = OnlineEWCWithFocalLabelSmoothLoss(num_classes=num_class, smoothing=0.1, focal_alpha=1.0, focal_gamma=2.0, ewc_lambda=0.0)
loss_func_with_ewc = OnlineEWCWithFocalLabelSmoothLoss(num_classes=num_class, smoothing=0.1, focal_alpha=1.0, focal_gamma=2.0, ewc_lambda=args.ewc_lambda)

# Load model and tokenizer
plm, tokenizer, model_config, WrapperClass = load_plm(model_name, args.pretrained_model_path)

# Define template - Improved prompt template based on the paper's methodology
template_text = ('Given the following vulnerable code snippet: {"placeholder":"text_a"} '
                 'and its vulnerability description: {"placeholder":"text_b"}, '
                 'classify the vulnerability type as: {"mask"}.')

mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text, model=plm)
# Define the verbalizer - Updated to match exactly with classes list
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

# Define the prompt model
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()

# Initialize loss function
loss_func = OnlineEWCWithFocalLabelSmoothLoss(num_classes=num_class, smoothing=0.1, focal_alpha=1.0, focal_gamma=2.0, ewc_lambda=args.ewc_lambda, decay_factor=0.9)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.01}
]
optimizer_grouped_parameters2 = [
    {'params': [p for n, p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]
optimizer1 = AdamW(optimizer_grouped_parameters1, lr=lr)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=lr)

# ===== KHỞI TẠO TASK-AWARE REPLAY SELECTOR =====
print(f"\n🎯 KHỞI TẠO TASK-AWARE REPLAY SYSTEM")
print(f"{'='*70}")
print(f"Loại cấu hình: {args.replay_config_type}")
print(f"Task decay factor: {args.task_decay_factor} (cao hơn = ưu tiên task cũ nhiều hơn)")
print(f"Min task ratio: {args.min_task_ratio} (tỷ lệ tối thiểu mỗi task)")
print(f"Similarity threshold: {args.similarity_threshold}")
print(f"Max code lines: {args.max_code_lines}")
print(f"Number of clusters: {args.n_clusters}")
print(f"Gradient importance: {args.enable_gradient_importance}")
print(f"{'='*70}\n")

# Tạo task-aware replay selector
task_aware_selector = create_task_aware_replay_selector(
    config_type=args.replay_config_type,
    task_decay_factor=args.task_decay_factor,
    min_task_ratio=args.min_task_ratio
)

# Bật gradient importance nếu được yêu cầu
if args.enable_gradient_importance:
    task_aware_selector.enable_gradient_importance(prompt_model, loss_func_no_ewc)

# Load test dataloaders for all tasks
test_dataloaders = []
for j in range(args.num_tasks):
    test_dataloader = PromptDataLoader(
        dataset=read_prompt_examples(test_paths[j]),
        template=mytemplate,
        tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
        batch_size=batch_size, shuffle=True,
        teacher_forcing=False, predict_eos_token=False, truncate_method="head",
        decoder_max_length=3)
    test_dataloaders.append(test_dataloader)

# Training process with Task-Aware Enhanced Replay
global_step = 0
prev_dev_loss = float('inf')
best_dev_loss = float('inf')

# Theo dõi performance của từng task để phân tích forgetting
task_performance_history = {}

print(f"\n🚀 BẮT ĐẦU TRAINING VỚI TASK-AWARE ENHANCED REPLAY")
print(f"{'='*70}")

# Main loop for each dataset
for i in range(1, args.num_tasks + 1):
    print(f"----------------------- Task {i} ---------------------------")
    if i == 1:
        train_dataloader = PromptDataLoader(
            dataset=read_prompt_examples(data_paths[i - 1]),
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            batch_size=batch_size,
            shuffle=True,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="head",
            decoder_max_length=3
        )
    else:
        # Create dataloader with merged previous datasets
        prev_examples, prev_task_origins = read_and_merge_previous_datasets(i, data_paths, return_task_info=True)
        
        # Compute replay budget based on ratio of total previous dataset
        total_prev_samples = len(prev_examples)
        replay_budget = int(total_prev_samples * args.replay_ratio)
        
        print(f"\n{'='*80}")
        print(f"CẤU HÌNH TASK-AWARE REPLAY BUFFER CHO TASK {i}")
        print(f"{'='*80}")
        print(f"Tổng số mẫu trước đó: {total_prev_samples}")
        print(f"Tỷ lệ replay: {args.replay_ratio} ({args.replay_ratio*100}%)")
        print(f"Ngân sách replay: {replay_budget} mẫu")
        print(f"Task decay factor: {args.task_decay_factor}")
        print(f"Min task ratio: {args.min_task_ratio}")
        print(f"{'='*80}\n")
        
        # Create dataloader for task-aware replay computation
        train_dataloader_prev = PromptDataLoader(
            dataset=prev_examples,
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for consistent indexing
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="head",
            decoder_max_length=3
        )
        
        # ===== SỬ DỤNG TASK-AWARE ENHANCED REPLAY SELECTION =====
        # Lấy examples task hiện tại để phân tích vulnerability
        current_examples = read_prompt_examples(data_paths[i - 1])
        
        # Lấy performance history của các task trước đó
        previous_performance = task_performance_history.get(i-1, None)
        
        # Task-aware replay selection với ưu tiên task cũ
        indices_to_replay, selection_info = task_aware_selector.select_enhanced_replay_samples_with_task_awareness(
            prompt_model=prompt_model,
            dataloader=train_dataloader_prev,
            examples=prev_examples,
            task_origins=prev_task_origins,
            num_samples=replay_budget,
            current_task_id=i,
            min_samples_per_class=args.min_samples_per_class,
            current_task_examples=current_examples,
            previous_performance=previous_performance
        )
        
        # Ghi log cải tiến để phân tích
        log_replay_improvements(selection_info, i, log_file="task_aware_replay_improvements.log")
        
        # Lấy ngữ cảnh lịch sử để prompting nâng cao (tùy chọn)
        historical_context = task_aware_selector.get_historical_context(i)
        if historical_context:
            print(f"📚 NGỮ CẢNH LỊCH SỬ CHO TASK {i}:")
            print(f"{'='*50}")
            print(historical_context)
            print(f"{'='*50}\n")
        
        # Build training dataset with current task + replay samples
        examples = read_prompt_examples(data_paths[i - 1])
        current_task_count = len(examples)
        
        for idx in indices_to_replay:
            examples.append(prev_examples[idx])
        
        print(f"Thành phần dataset training cho Task {i}:")
        print(f"  Task hiện tại (Task {i}): {current_task_count} mẫu")
        print(f"  Task-aware replay buffer: {len(indices_to_replay)} mẫu")
        print(f"  Tổng cộng: {len(examples)} mẫu\n")

        train_dataloader = PromptDataLoader(
            dataset=examples,
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            batch_size=batch_size,
            shuffle=True,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="head",
            decoder_max_length=3
        )

    validation_dataloader = PromptDataLoader(
        dataset=read_prompt_examples(valid_paths[i - 1]),
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_l,
        batch_size=batch_size,
        shuffle=True,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="head",
        decoder_max_length=3
    )

    num_training_steps = num_epochs * len(train_dataloader)
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=0, num_training_steps=num_training_steps)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0, num_training_steps=num_training_steps)

    if i >= 2:
        prompt_model.load_state_dict(
            torch.load(os.path.join(args.checkpoint_dir, 'best', 'best.ckpt'),
                       map_location=torch.device('cuda:0')))

    print(f"Bắt đầu Phase 1 cho Task {i}: Focal Loss + Label Smoothing")
    train_phase_one(
        prompt_model,
        train_dataloader,
        validation_dataloader,
        optimizer1,
        optimizer2,
        scheduler1,
        scheduler2,
        num_epochs,
        loss_func_no_ewc,
        i,
        patience=args.patience
    )

    eval_results_phase1 = test(prompt_model, validation_dataloader, f'task_{i}_val_phase1')
    print(f"Đánh giá Phase 1 cho task {i}: ", eval_results_phase1)

    prompt_model.load_state_dict(
        torch.load(os.path.join(args.checkpoint_dir, 'best', 'best.ckpt'),
                   map_location=torch.device('cuda:0')))
    print(f"Bắt đầu Phase 2 cho Task {i}: Focal Loss + Label Smoothing + EWC")
    train_phase_two(
        prompt_model,
        train_dataloader,
        validation_dataloader,
        optimizer1,
        optimizer2,
        scheduler1,
        scheduler2,
        num_epochs,
        loss_func_with_ewc,
        i,
        patience=args.patience
    )

    eval_results_phase2 = test(prompt_model, validation_dataloader, f'task_{i}_val_phase2')
    print(f"Đánh giá Phase 2 cho task {i}: ", eval_results_phase2)
    
    # Save final checkpoint for this task
    save_task_checkpoint(prompt_model, i, "final")
    
    # Update Fisher Information for EWC after each task
    loss_func_with_ewc.update_fisher(prompt_model, train_dataloader)
    print(f"Kiểm tra Task {i} model trên các datasets trước đó sau Phase 2")

    print("----------------------Tải model tốt nhất và kiểm tra-----------------------------")
    prompt_model.load_state_dict(
        torch.load(os.path.join(args.checkpoint_dir, "best", "best.ckpt"),
                   map_location=torch.device('cuda:0')))
    
    # Test trên tất cả tasks và lưu performance
    current_task_performance = {}
    for j, (task_dataloader, task_name) in enumerate(zip(test_dataloaders, [f'task{k}' for k in range(1, args.num_tasks + 1)]), 1):
        if j <= i:  # Chỉ test các task đã học
            acc, precisionma, recallma, f1wei, f1ma = test(prompt_model, task_dataloader, f'{task_name}_test_task_{i}')
            current_task_performance[j] = {
                'accuracy': acc,
                'precision_macro': precisionma,
                'recall_macro': recallma,
                'f1_weighted': f1wei,
                'f1_macro': f1ma
            }
    
    # Lưu performance history
    task_performance_history[i] = current_task_performance
    
    # Phân tích forgetting pattern sau mỗi task
    if i > 1:
        task_accuracies = {task_id: perf['accuracy'] for task_id, perf in current_task_performance.items()}
        forgetting_analysis = analyze_forgetting_pattern(task_accuracies)
        
        print(f"\n🔍 PHÂN TÍCH FORGETTING SAU TASK {i}:")
        print(f"{'='*60}")
        print(f"Task bị forgetting nhiều nhất: Task {forgetting_analysis.get('worst_task', 'N/A')} "
              f"({forgetting_analysis.get('worst_accuracy', 0):.2%})")
        print(f"Task tốt nhất: Task {forgetting_analysis.get('best_task', 'N/A')} "
              f"({forgetting_analysis.get('best_accuracy', 0):.2%})")
        print(f"Accuracy trung bình: {forgetting_analysis.get('average_accuracy', 0):.2%}")
        print(f"Độ lệch chuẩn: {forgetting_analysis.get('accuracy_std', 0):.3f}")
        
        if forgetting_analysis.get('recommendations'):
            print(f"\n💡 KHUYẾN NGHỊ CHO TASK TIẾP THEO:")
            for rec in forgetting_analysis['recommendations']:
                print(f"  - {rec}")
        print(f"{'='*60}\n")

# Display all available checkpoints at the end
print("\n" + "="*80)
print("HOÀN THÀNH TRAINING - TÓM TẮT CHECKPOINT")
print("="*80)
list_available_checkpoints()
print("\nQuy ước đặt tên checkpoint:")
print("  - task_X_phase1_best.ckpt: Model tốt nhất từ Phase 1 của Task X")
print("  - task_X_phase2_best.ckpt: Model tốt nhất từ Phase 2 của Task X") 
print("  - task_X_final.ckpt: Model cuối cùng sau khi hoàn thành Task X")
print("="*80)

print(f"\n🎉 HOÀN THÀNH TRAINING VỚI TASK-AWARE ENHANCED REPLAY!")
print(f"📊 Kiểm tra file 'task_aware_replay_improvements.log' để xem thống kê cải tiến")
print(f"📁 Kiểm tra thư mục long-term memory để xem bộ nhớ dài hạn")
print(f"🔍 Performance history đã được lưu để phân tích forgetting pattern")
print(f"{'='*70}")