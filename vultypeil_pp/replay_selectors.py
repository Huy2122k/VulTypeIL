"""
Selection strategies for VulTypeIL++
Implements: Mahalanobis, MCSS, GCR-approx
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import distance
from collections import defaultdict, Counter
from typing import List, Tuple
from openprompt.data_utils import InputExample
from tqdm import tqdm


def prepare_labels(labels, use_cuda=True):
    """
    Prepare labels for training, ensuring they are proper tensors.
    
    Args:
        labels: Labels from dataloader (can be tensor, list, or mixed types)
        use_cuda: Whether to move to CUDA
    
    Returns:
        Tensor of labels with dtype=torch.long
    """
    if not torch.is_tensor(labels):
        # Ensure all labels are integers
        if isinstance(labels, list):
            labels = [int(l) if not isinstance(l, int) else l for l in labels]
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        # Ensure tensor has correct dtype
        if labels.dtype != torch.long:
            labels = labels.long()
    
    if use_cuda:
        labels = labels.cuda()
    
    return labels


def mahalanobis_select(prompt_model, dataloader, examples: List[InputExample], 
                       num_samples: int, use_cuda: bool = True) -> Tuple[List[int], np.ndarray]:
    """
    Select samples using Mahalanobis distance (VulTypeIL original method).
    
    Args:
        prompt_model: The prompt model
        dataloader: DataLoader for candidates
        examples: List of InputExample objects
        num_samples: Number of samples to select
        use_cuda: Whether to use CUDA
    
    Returns:
        Tuple of (selected_indices, all_features)
    """
    prompt_model.eval()
    all_features = []
    
    with torch.no_grad():
        for inputs in dataloader:
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            all_features.append(logits.cpu().numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    
    # Compute Mahalanobis distance
    mean_features = np.mean(all_features, axis=0)
    cov_matrix = np.cov(all_features, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)
    
    distances = [distance.mahalanobis(f, mean_features, cov_inv) for f in all_features]
    
    # Select top num_samples by distance (most uncertain)
    indices = np.argsort(distances)[::-1][:num_samples].tolist()
    
    return indices, all_features


def mcss_select(candidates: List[InputExample], num_samples: int, 
                prompt_model, dataloader, use_cuda: bool = True,
                tail_threshold: float = 0.05, alpha_loss: float = 0.5,
                overselect_k: int = 5) -> List[InputExample]:
    """
    Multi-Criteria Coreset Selection (MCSS).
    
    Combines:
    - Prototype distance (representativeness)
    - Intra-class variation (diversity)
    - Classifier loss (difficulty)
    
    Args:
        candidates: List of candidate examples
        num_samples: Target buffer size
        prompt_model: Model for feature extraction
        dataloader: DataLoader for candidates
        use_cuda: Whether to use CUDA
        tail_threshold: Threshold for tail class identification
        alpha_loss: Weight for loss-based scoring
        overselect_k: Overselection factor for diversity
    
    Returns:
        Selected examples
    """
    prompt_model.eval()
    
    # Extract features and compute losses
    all_features = []
    all_losses = []
    all_labels = []
    
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="MCSS: Computing features"):
            if use_cuda:
                inputs = inputs.cuda()
            
            logits = prompt_model(inputs)
            labels = prepare_labels(inputs['tgt_text'], use_cuda)
            
            # Compute loss per sample
            loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
            
            all_features.append(logits.cpu().numpy())
            all_losses.append(loss_per_sample.cpu().numpy())
            all_labels.extend(labels.cpu().tolist())
    
    all_features = np.concatenate(all_features, axis=0)
    all_losses = np.concatenate(all_losses, axis=0)
    
    # Normalize losses
    all_losses = (all_losses - all_losses.min()) / (all_losses.max() - all_losses.min() + 1e-8)
    
    # Group by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(all_labels):
        class_to_indices[label].append(idx)
    
    # Identify tail classes
    class_counts = Counter(all_labels)
    total = len(all_labels)
    tail_classes = [c for c, cnt in class_counts.items() 
                   if cnt / total < tail_threshold]
    
    # Allocate per-class budget (favor tail classes)
    per_class_budget = {}
    remaining_budget = num_samples
    
    # First, ensure minimum for tail classes
    min_tail = max(2, num_samples // (len(class_counts) * 2))
    for c in tail_classes:
        budget = min(min_tail, len(class_to_indices[c]))
        per_class_budget[c] = budget
        remaining_budget -= budget
    
    # Distribute remaining proportionally
    for c in class_to_indices.keys():
        if c not in tail_classes:
            prop = class_counts[c] / total
            budget = int(remaining_budget * prop)
            per_class_budget[c] = budget
    
    # Adjust to match exactly num_samples
    total_allocated = sum(per_class_budget.values())
    if total_allocated < num_samples:
        # Add to largest classes
        for c in sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True):
            if total_allocated >= num_samples:
                break
            if per_class_budget[c] < len(class_to_indices[c]):
                per_class_budget[c] += 1
                total_allocated += 1
    
    # Select per class
    selected_indices = []
    
    for c, budget in per_class_budget.items():
        if budget == 0:
            continue
        
        indices = class_to_indices[c]
        if len(indices) <= budget:
            selected_indices.extend(indices)
            continue
        
        # Compute class prototype
        class_features = all_features[indices]
        prototype = class_features.mean(axis=0)
        
        # Score: combine distance to prototype + loss
        scores = []
        for idx in indices:
            feat = all_features[idx]
            dist = np.linalg.norm(feat - prototype)
            loss = all_losses[idx]
            score = (1 - alpha_loss) * dist + alpha_loss * loss
            scores.append((idx, score))
        
        # Select top k*budget by score
        scores.sort(key=lambda x: x[1], reverse=True)
        candidates_pool = [idx for idx, _ in scores[:budget * overselect_k]]
        
        # Farthest-first for diversity
        selected = [candidates_pool[0]]
        candidates_pool = candidates_pool[1:]
        
        while len(selected) < budget and candidates_pool:
            # Find farthest from selected
            max_min_dist = -1
            best_idx = None
            
            for idx in candidates_pool:
                feat = all_features[idx]
                min_dist = min([np.linalg.norm(feat - all_features[s]) 
                               for s in selected])
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                candidates_pool.remove(best_idx)
            else:
                break
        
        selected_indices.extend(selected)
    
    # Return selected examples
    return [candidates[i] for i in selected_indices]


def gcr_approx_select(candidates: List[InputExample], num_samples: int,
                      prompt_model, dataloader, use_cuda: bool = True,
                      per_class: bool = True) -> List[InputExample]:
    """
    Gradient Coreset Reduction (GCR) - Approximation.
    
    Uses gradient in output space (logits) as a cheap approximation.
    
    Args:
        candidates: List of candidate examples
        num_samples: Target buffer size
        prompt_model: Model for gradient computation
        dataloader: DataLoader for candidates
        use_cuda: Whether to use CUDA
        per_class: Whether to do per-class selection
    
    Returns:
        Selected examples
    """
    prompt_model.eval()
    
    # Compute gradient embeddings
    all_gradients = []
    all_labels = []
    
    for inputs in tqdm(dataloader, desc="GCR: Computing gradients"):
        if use_cuda:
            inputs = inputs.cuda()
        
        # Forward pass
        logits = prompt_model(inputs)
        labels = prepare_labels(inputs['tgt_text'], use_cuda)
        
        # Compute gradient in output space
        logits_detached = logits.detach().requires_grad_(True)
        loss = F.cross_entropy(logits_detached, labels, reduction='mean')
        
        # Backward to get gradients
        grad = torch.autograd.grad(loss, logits_detached)[0]
        
        all_gradients.append(grad.cpu().numpy())
        all_labels.extend(labels.cpu().tolist())
    
    all_gradients = np.concatenate(all_gradients, axis=0)
    
    if not per_class:
        # Global selection
        selected_indices = _greedy_gradient_matching(all_gradients, num_samples)
        return [candidates[i] for i in selected_indices]
    
    # Per-class selection
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(all_labels):
        class_to_indices[label].append(idx)
    
    # Allocate budget proportionally
    class_counts = Counter(all_labels)
    total = len(all_labels)
    per_class_budget = {}
    
    for c, cnt in class_counts.items():
        budget = max(1, int(num_samples * cnt / total))
        per_class_budget[c] = budget
    
    # Adjust to match exactly
    total_allocated = sum(per_class_budget.values())
    if total_allocated < num_samples:
        for c in sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True):
            if total_allocated >= num_samples:
                break
            per_class_budget[c] += 1
            total_allocated += 1
    elif total_allocated > num_samples:
        for c in sorted(class_counts.keys(), key=lambda x: class_counts[x]):
            if total_allocated <= num_samples:
                break
            if per_class_budget[c] > 1:
                per_class_budget[c] -= 1
                total_allocated -= 1
    
    # Select per class
    selected_indices = []
    for c, budget in per_class_budget.items():
        indices = class_to_indices[c]
        if len(indices) <= budget:
            selected_indices.extend(indices)
        else:
            class_gradients = all_gradients[indices]
            local_selected = _greedy_gradient_matching(class_gradients, budget)
            selected_indices.extend([indices[i] for i in local_selected])
    
    return [candidates[i] for i in selected_indices]


def _greedy_gradient_matching(gradients: np.ndarray, k: int) -> List[int]:
    """
    Greedy algorithm to select k samples that best approximate mean gradient.
    
    Args:
        gradients: Array of shape [N, D]
        k: Number of samples to select
    
    Returns:
        List of selected indices
    """
    N = len(gradients)
    if k >= N:
        return list(range(N))
    
    target_mean = gradients.mean(axis=0)
    selected = []
    remaining = list(range(N))
    
    # Greedy selection
    for _ in range(k):
        best_idx = None
        best_error = float('inf')
        
        for idx in remaining:
            # Compute mean if we add this sample
            test_selected = selected + [idx]
            test_mean = gradients[test_selected].mean(axis=0)
            error = np.linalg.norm(test_mean - target_mean)
            
            if error < best_error:
                best_error = error
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
    
    return selected
