"""
Main runner for VulTypeIL++ experiments
Supports all 6 ablation configurations
"""
import argparse
import os
import random
import yaml
import torch
import numpy as np
from collections import namedtuple, Counter
from transformers import (RobertaTokenizer, T5Config, T5ForConditionalGeneration,
                          AdamW, get_linear_schedule_with_warmup)
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.plms import add_special_tokens
from openprompt.plms.seq2seq import T5TokenizerWrapper
from openprompt.prompts import MixedTemplate, ManualVerbalizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import VulTypeIL++ modules
from data_utils import read_prompt_examples, read_and_merge_previous_datasets
from replay_buffer import ReplayBuffer
from mixed_dataloader import MixedBatchIterator
from selectors import mahalanobis_select, mcss_select, gcr_approx_select
from trainer import (OnlineEWCWithFocalLabelSmoothLoss, train_phase_one, 
                     train_phase_two, train_consolidation)
from metrics import ContinualMetrics

# Define classes
CLASSES = [
    'CWE-119', 'CWE-125', 'CWE-787', 'CWE-476', 'CWE-20', 'CWE-416',
    'CWE-190', 'CWE-200', 'CWE-120', 'CWE-399', 'CWE-401', 'CWE-264', 'CWE-772',
    'CWE-189', 'CWE-362', 'CWE-835', 'CWE-369', 'CWE-617', 'CWE-400', 'CWE-415',
    'CWE-122', 'CWE-770', 'CWE-22'
]

ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model', 'wrapper'))


def load_plm(model_name, model_path):
    """Load pre-trained language model (CodeT5)."""
    codet5_model = ModelClass(**{
        "config": T5Config,
        "tokenizer": RobertaTokenizer,
        "model": T5ForConditionalGeneration,
        "wrapper": T5TokenizerWrapper
    })
    
    model_config = codet5_model.config.from_pretrained("Salesforce/codet5-base")
    model = codet5_model.model.from_pretrained("Salesforce/codet5-base", config=model_config)
    tokenizer = codet5_model.tokenizer.from_pretrained("Salesforce/codet5-base")
    wrapper = codet5_model.wrapper
    
    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=None)
    
    return model, tokenizer, model_config, wrapper


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test(prompt_model, test_dataloader, name, results_dir, use_cuda=True):
    """Test the model and save predictions."""
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
    
    acc = accuracy_score(alllabels, allpreds)
    precisionwei, recallwei, f1wei, _ = precision_recall_fscore_support(
        alllabels, allpreds, average='weighted', zero_division=0)
    precisionma, recallma, f1ma, _ = precision_recall_fscore_support(
        alllabels, allpreds, average='macro', zero_division=0)
    
    # Save predictions
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{name}.pred.csv"), 'w') as f, \
         open(os.path.join(results_dir, f"{name}.gold.csv"), 'w') as f1:
        for ref, gold in zip(allpreds, alllabels):
            f.write(str(ref) + '\n')
            f1.write(str(gold) + '\n')
    
    print(f"Test {name}: Acc={acc:.4f}, Macro-F1={f1ma:.4f}, Weighted-F1={f1wei:.4f}")
    
    return acc, f1ma


def run_experiment(config_path, batch_size=None, num_epochs=None, patience=None):
    """Run experiment based on configuration file."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command-line arguments if provided
    if batch_size is not None:
        config['training']['batch_size'] = batch_size
        print(f"Overriding batch_size: {batch_size}")
    
    if num_epochs is not None:
        config['training']['num_epochs'] = num_epochs
        print(f"Overriding num_epochs: {num_epochs}")
    
    if patience is not None:
        config['training']['patience'] = patience
        print(f"Overriding patience: {patience}")
    
    print(f"\n{'='*80}")
    print(f"Running: {config['method']}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}\n")
    
    # Set seed
    set_seed(config['training']['seed'])
    
    # Setup paths
    data_dir = config['data']['data_dir']
    # If data_dir is relative and doesn't exist, try parent directory
    if not os.path.exists(data_dir):
        parent_data_dir = os.path.join('..', data_dir)
        if os.path.exists(parent_data_dir):
            data_dir = parent_data_dir
            print(f"Using data directory: {data_dir}")
    
    num_tasks = config['data']['num_tasks']
    num_classes = config['data']['num_classes']
    
    data_paths = [os.path.join(data_dir, f"task{i}_train.csv") for i in range(1, num_tasks + 1)]
    test_paths = [os.path.join(data_dir, f"task{i}_test.csv") for i in range(1, num_tasks + 1)]
    valid_paths = [os.path.join(data_dir, f"task{i}_valid.csv") for i in range(1, num_tasks + 1)]
    
    # Create output directories
    checkpoint_dir = config['output']['checkpoint_dir']
    results_dir = config['output']['results_dir']
    metrics_dir = config['output']['metrics_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Load model
    use_cuda = config['training']['use_cuda'] and torch.cuda.is_available()
    plm, tokenizer, model_config, WrapperClass = load_plm(
        config['model']['name'], config['model']['pretrained_path'])
    
    # Define template and verbalizer
    template_text = ('Given the following vulnerable code snippet: {"placeholder":"text_a"} '
                    'and its vulnerability description: {"placeholder":"text_b"}, '
                    'classify the vulnerability type as: {"mask"}.')
    mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text, model=plm)
    
    myverbalizer = ManualVerbalizer(tokenizer, classes=CLASSES, label_words={
        "CWE-119": ["Improper Memory Operations", "Buffer Overflow"],
        "CWE-125": ["Out-of-bounds Read", "Memory Access Violation"],
        "CWE-787": ["Out-of-bounds Write", "Buffer Overflow"],
        "CWE-476": ["NULL Pointer Dereference", "Access to Null Pointer"],
        "CWE-20": ["Improper Input Validation", "Input Sanitization Flaw"],
        "CWE-416": ["Use After Free", "Dangling Pointer"],
        "CWE-190": ["Integer Overflow", "Integer Wraparound"],
        "CWE-200": ["Exposure of Sensitive Data", "Unauthorized Information Access"],
        "CWE-120": ["Classic Buffer Overflow", "Buffer Copy Error"],
        "CWE-399": ["Resource Management Error", "Improper Resource Handling"],
        "CWE-401": ["Memory Leak", "Unreleased Memory"],
        "CWE-264": ["Access Control Flaw", "Privilege Escalation"],
        "CWE-772": ["Resource Management Failure", "Resource Leak"],
        "CWE-189": ["Numeric Error", "Numerical Miscalculation"],
        "CWE-362": ["Race Condition", "Shared Resource Access"],
        "CWE-835": ["Infinite Loop", "Unreachable Loop"],
        "CWE-369": ["Divide By Zero", "Division Error"],
        "CWE-617": ["Reachable Assertion", "Assertion Failure"],
        "CWE-400": ["Uncontrolled Resource Consumption", "Excessive Resource Allocation"],
        "CWE-415": ["Double Free", "Double Memory Deallocation"],
        "CWE-122": ["Heap Overflow", "Buffer Overflow in Heap"],
        "CWE-770": ["Unrestricted Resource Allocation", "Resource Overconsumption"],
        "CWE-22": ["Path Traversal", "Directory Traversal"]
    })
    
    prompt_model = PromptForClassification(
        plm=plm, template=mytemplate, verbalizer=myverbalizer,
        freeze_plm=config['model']['freeze_plm'])
    
    if use_cuda:
        prompt_model = prompt_model.cuda()
    
    # Initialize loss functions
    loss_func_no_ewc = OnlineEWCWithFocalLabelSmoothLoss(
        num_classes=num_classes,
        smoothing=config['loss']['smoothing'],
        focal_alpha=config['loss']['focal_alpha'],
        focal_gamma=config['loss']['focal_gamma'],
        ewc_lambda=0.0
    )
    
    loss_func_with_ewc = OnlineEWCWithFocalLabelSmoothLoss(
        num_classes=num_classes,
        smoothing=config['loss']['smoothing'],
        focal_alpha=config['loss']['focal_alpha'],
        focal_gamma=config['loss']['focal_gamma'],
        ewc_lambda=config['loss']['ewc_lambda'],
        decay_factor=config['loss']['decay_factor']
    )
    
    # Initialize replay buffer if needed
    buffer = None
    if config['buffer']['enabled']:
        buffer = ReplayBuffer(
            max_size=config['buffer']['size'],
            tail_threshold=config['buffer'].get('tail_threshold', 0.05)
        )
    
    # Initialize metrics tracker
    metrics = ContinualMetrics(num_tasks=num_tasks, num_classes=num_classes)
    
    # Load test dataloaders
    test_dataloaders = []
    for j in range(num_tasks):
        test_dataloader = PromptDataLoader(
            dataset=read_prompt_examples(test_paths[j], CLASSES),
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=config['model']['max_seq_length'],
            batch_size=config['training']['batch_size'],
            shuffle=False,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="head",
            decoder_max_length=3
        )
        test_dataloaders.append(test_dataloader)
    
    # Main training loop
    for task_id in range(1, num_tasks + 1):
        print(f"\n{'='*80}")
        print(f"TASK {task_id}")
        print(f"{'='*80}\n")
        
        # Prepare training data based on replay mode
        if config['replay']['mode'] == 'append_dataset':
            # Ablation 1 & 2: Materialize full dataset
            new_examples = read_prompt_examples(data_paths[task_id - 1], CLASSES)
            
            if task_id > 1:
                old_examples = read_and_merge_previous_datasets(task_id, data_paths, CLASSES)
                
                if config['replay']['selector'] == 'mahalanobis':
                    # Ablation 1: Fixed 200 with Mahalanobis
                    old_loader = PromptDataLoader(
                        dataset=old_examples, template=mytemplate, tokenizer=tokenizer,
                        tokenizer_wrapper_class=WrapperClass,
                        max_seq_length=config['model']['max_seq_length'],
                        batch_size=config['training']['batch_size'], shuffle=False,
                        teacher_forcing=False, predict_eos_token=False,
                        truncate_method="head", decoder_max_length=3
                    )
                    indices, _ = mahalanobis_select(
                        prompt_model, old_loader, old_examples,
                        num_samples=config['replay']['num_samples'],
                        use_cuda=use_cuda
                    )
                    replay_examples = [old_examples[i] for i in indices]
                    print(f"Selected {len(replay_examples)} samples via Mahalanobis")
                
                elif config['replay']['selector'] == 'random':
                    # Ablation 2: Random ratio of all-old
                    k = int(config['replay']['ratio_allold'] * len(old_examples))
                    replay_examples = random.sample(old_examples, k)
                    print(f"Selected {len(replay_examples)} samples randomly ({config['replay']['ratio_allold']*100}%)")
                
                new_examples.extend(replay_examples)
            
            train_dataloader = PromptDataLoader(
                dataset=new_examples, template=mytemplate, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass,
                max_seq_length=config['model']['max_seq_length'],
                batch_size=config['training']['batch_size'], shuffle=True,
                teacher_forcing=False, predict_eos_token=False,
                truncate_method="head", decoder_max_length=3
            )
        
        elif config['replay']['mode'] == 'batch_mix':
            # Ablation 3-6: Mixed batch iterator (scalable)
            new_examples = read_prompt_examples(data_paths[task_id - 1], CLASSES)
            new_loader = PromptDataLoader(
                dataset=new_examples, template=mytemplate, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass,
                max_seq_length=config['model']['max_seq_length'],
                batch_size=config['training']['batch_size'], shuffle=True,
                teacher_forcing=False, predict_eos_token=False,
                truncate_method="head", decoder_max_length=3
            )
            
            if task_id == 1 or len(buffer) == 0:
                train_dataloader = new_loader
            else:
                mem_loader = PromptDataLoader(
                    dataset=buffer.sample(len(buffer.examples), 
                                        mode=config['buffer']['sample']),
                    template=mytemplate, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass,
                    max_seq_length=config['model']['max_seq_length'],
                    batch_size=config['training']['batch_size'], shuffle=True,
                    teacher_forcing=False, predict_eos_token=False,
                    truncate_method="head", decoder_max_length=3
                )
                train_dataloader = MixedBatchIterator(
                    new_loader, mem_loader,
                    replay_ratio=config['replay']['replay_ratio']
                )
                print(f"Using mixed batch iterator with replay_ratio={config['replay']['replay_ratio']}")
        
        # Validation dataloader
        validation_dataloader = PromptDataLoader(
            dataset=read_prompt_examples(valid_paths[task_id - 1], CLASSES),
            template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=config['model']['max_seq_length'],
            batch_size=config['training']['batch_size'], shuffle=False,
            teacher_forcing=False, predict_eos_token=False,
            truncate_method="head", decoder_max_length=3
        )
        
        # Setup optimizers and schedulers
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in prompt_model.named_parameters() 
                       if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() 
                       if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer_grouped_parameters2 = [
            {'params': [p for n, p in prompt_model.template.named_parameters() 
                       if "raw_embedding" not in n]}
        ]
        
        optimizer1 = AdamW(optimizer_grouped_parameters1, lr=config['training']['lr'])
        optimizer2 = AdamW(optimizer_grouped_parameters2, lr=5e-5)
        
        num_training_steps = config['training']['num_epochs'] * len(train_dataloader)
        scheduler1 = get_linear_schedule_with_warmup(
            optimizer1, num_warmup_steps=0, num_training_steps=num_training_steps)
        scheduler2 = get_linear_schedule_with_warmup(
            optimizer2, num_warmup_steps=0, num_training_steps=num_training_steps)
        
        # Load previous checkpoint if task > 1
        if task_id > 1:
            prev_checkpoint = os.path.join(checkpoint_dir, f'task_{task_id-1}_phase2_best.ckpt')
            if os.path.exists(prev_checkpoint):
                prompt_model.load_state_dict(torch.load(prev_checkpoint))
                print(f"Loaded checkpoint from Task {task_id-1}")
        
        # Phase 1: Train without EWC
        print(f"\n--- Phase 1: Focal + Label Smoothing ---")
        train_phase_one(
            prompt_model, train_dataloader, validation_dataloader,
            optimizer1, optimizer2, scheduler1, scheduler2,
            config['training']['num_epochs'], loss_func_no_ewc,
            task_id, checkpoint_dir, config['training']['patience'], use_cuda
        )
        
        # Load best Phase 1 model
        prompt_model.load_state_dict(torch.load(
            os.path.join(checkpoint_dir, f'task_{task_id}_phase1_best.ckpt')))
        
        # Phase 2: Train with EWC
        print(f"\n--- Phase 2: Focal + Label Smoothing + EWC ---")
        train_phase_two(
            prompt_model, train_dataloader, validation_dataloader,
            optimizer1, optimizer2, scheduler1, scheduler2,
            config['training']['num_epochs'], loss_func_with_ewc,
            task_id, checkpoint_dir, config['training']['patience'], use_cuda
        )
        
        # Load best Phase 2 model
        prompt_model.load_state_dict(torch.load(
            os.path.join(checkpoint_dir, f'task_{task_id}_phase2_best.ckpt')))
        
        # Consolidation phase (if enabled)
        if config['consolidation']['enabled'] and buffer is not None and len(buffer) > 0:
            print(f"\n--- Consolidation Phase ---")
            cons_loader = PromptDataLoader(
                dataset=buffer.examples, template=mytemplate, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass,
                max_seq_length=config['model']['max_seq_length'],
                batch_size=config['training']['batch_size'], shuffle=True,
                teacher_forcing=False, predict_eos_token=False,
                truncate_method="head", decoder_max_length=3
            )
            train_consolidation(
                prompt_model, cons_loader, optimizer1, optimizer2,
                scheduler1, scheduler2, config['consolidation']['steps'],
                loss_func_with_ewc, task_id, checkpoint_dir, use_cuda
            )
        
        # Update Fisher information
        if task_id > 1 or config['replay']['mode'] == 'batch_mix':
            print("\nUpdating Fisher information...")
            loss_func_with_ewc.update_fisher(prompt_model, train_dataloader, use_cuda)
        
        # Update replay buffer (for scalable methods)
        if buffer is not None:
            print("\nUpdating replay buffer...")
            if config['buffer']['update'] == 'reservoir':
                # Ablation 3: Reservoir sampling
                buffer.add_stream(new_examples)
            
            elif config['buffer']['update'] in ['mcss', 'gcr_approx']:
                # Ablation 4-6: MCSS or GCR
                candidates = buffer.examples + new_examples
                cand_loader = PromptDataLoader(
                    dataset=candidates, template=mytemplate, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass,
                    max_seq_length=config['model']['max_seq_length'],
                    batch_size=config['training']['batch_size'], shuffle=False,
                    teacher_forcing=False, predict_eos_token=False,
                    truncate_method="head", decoder_max_length=3
                )
                
                if config['buffer']['update'] == 'mcss':
                    selected = mcss_select(
                        candidates, buffer.max_size, prompt_model, cand_loader,
                        use_cuda=use_cuda,
                        tail_threshold=config['mcss']['tail_threshold'],
                        alpha_loss=config['mcss']['alpha_loss'],
                        overselect_k=config['mcss']['overselect_k']
                    )
                else:  # gcr_approx
                    selected = gcr_approx_select(
                        candidates, buffer.max_size, prompt_model, cand_loader,
                        use_cuda=use_cuda,
                        per_class=config['gcr']['per_class']
                    )
                
                buffer.examples = selected
            
            # Print buffer statistics
            stats = buffer.get_statistics()
            print(f"Buffer size: {stats['size']}, Classes: {stats['num_classes']}, "
                  f"Tail classes: {stats['tail_classes']}")
        
        # Evaluate on all tasks
        print(f"\n--- Evaluation after Task {task_id} ---")
        for eval_task_id in range(1, task_id + 1):
            acc, f1 = test(
                prompt_model, test_dataloaders[eval_task_id - 1],
                f"task{eval_task_id}_after_task{task_id}",
                results_dir, use_cuda
            )
            metrics.update(task_id, eval_task_id, acc, f1)
    
    # Save final metrics
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*80}\n")
    metrics.save(metrics_dir, config['method'])
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VulTypeIL++ experiments")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Override number of epochs from config')
    parser.add_argument('--patience', type=int, default=None,
                       help='Override early stopping patience from config')
    args = parser.parse_args()
    
    run_experiment(args.config, args.batch_size, args.num_epochs, args.patience)
