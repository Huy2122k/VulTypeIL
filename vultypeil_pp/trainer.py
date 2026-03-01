"""
Trainer for VulTypeIL++
Handles 2-phase training + optional consolidation
"""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import os


def move_to_cuda(inputs, use_cuda=True):
    """
    Move inputs to CUDA, handling both OpenPrompt batch objects and dicts.
    
    Args:
        inputs: Batch from dataloader (OpenPrompt object or dict)
        use_cuda: Whether to use CUDA
    
    Returns:
        Inputs moved to CUDA
    """
    if not use_cuda:
        return inputs
    
    # If inputs has .cuda() method (OpenPrompt batch object), use it
    if hasattr(inputs, 'cuda') and callable(getattr(inputs, 'cuda')):
        return inputs.cuda()
    
    # Otherwise, treat as dict and move tensors manually
    if isinstance(inputs, dict):
        return {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    # Fallback: return as is
    return inputs


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


class OnlineEWCWithFocalLabelSmoothLoss(torch.nn.Module):
    """Loss function combining Focal Loss, Label Smoothing, and Online EWC."""
    
    def __init__(self, num_classes, smoothing=0.1, focal_alpha=1.0, 
                 focal_gamma=2.0, ewc_lambda=0.4, decay_factor=0.9):
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
        """Combine Focal Loss and Label Smoothing Cross Entropy."""
        # Label Smoothing
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        pred = F.softmax(logits, dim=-1)
        log_probs = torch.log(pred + 1e-8)
        ce_loss = -torch.sum(target_smooth * log_probs, dim=-1).mean()

        # Focal Loss
        true_class_pred = pred.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = ((1 - true_class_pred) ** self.focal_gamma) * self.focal_alpha
        focal_loss = (focal_weight * (-torch.log(true_class_pred + 1e-8))).mean()

        return w * focal_loss + (1 - w) * ce_loss

    def ewc_loss_online(self, prompt_model):
        """Calculate Online EWC loss."""
        ewc_loss = 0
        for name, param in prompt_model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                optpar = self.optpar_dict[name]
                ewc_loss += (fisher * (param - optpar) ** 2).sum()
        return self.ewc_lambda * ewc_loss

    def update_fisher(self, prompt_model, dataloader, use_cuda=True):
        """Update Fisher information with current task's data."""
        current_fisher_dict = {}
        prompt_model.eval()

        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = move_to_cuda(inputs, use_cuda)
            logits = prompt_model(inputs)
            labels = prepare_labels(inputs['tgt_text'], use_cuda)
            loss = self.focal_label_smooth_ce_loss(logits, labels)
            prompt_model.zero_grad()
            loss.backward()

            for name, param in prompt_model.named_parameters():
                if param.grad is not None:
                    if name not in current_fisher_dict:
                        current_fisher_dict[name] = param.grad.data.clone().detach() ** 2
                    else:
                        current_fisher_dict[name] += param.grad.data.clone().detach() ** 2

        # Average Fisher information
        for name in current_fisher_dict:
            current_fisher_dict[name] /= len(dataloader)

        # Update global Fisher with decay
        for name, fisher_value in current_fisher_dict.items():
            if name in self.fisher_dict:
                self.fisher_dict[name] = self.decay_factor * self.fisher_dict[name] + fisher_value
            else:
                self.fisher_dict[name] = fisher_value
            self.optpar_dict[name] = prompt_model.state_dict()[name].clone().detach()

    def forward(self, prompt_model, logits, target):
        """Combine all losses."""
        focal_label_smooth_loss = self.focal_label_smooth_ce_loss(logits, target)
        ewc_loss = self.ewc_loss_online(prompt_model)
        return focal_label_smooth_loss + ewc_loss


def train_phase_one(prompt_model, train_dataloader, val_dataloader, 
                   optimizer1, optimizer2, scheduler1, scheduler2,
                   num_epochs, loss_func_no_ewc, task_id, 
                   checkpoint_dir, patience=5, use_cuda=True):
    """Phase 1: Train with Focal Loss + Label Smoothing only (no EWC)."""
    best_val_loss = float('inf')
    patience_counter = 0
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)), 
                       desc=f"Task {task_id} Phase 1")

    for epoch in range(num_epochs):
        prompt_model.train()
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = move_to_cuda(inputs, use_cuda)
            logits = prompt_model(inputs)
            labels = prepare_labels(inputs['tgt_text'], use_cuda)
            
            loss = loss_func_no_ewc.focal_label_smooth_ce_loss(logits, labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()
            prompt_model.zero_grad()
            progress_bar.update(1)

        # Validation
        prompt_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_dataloader:
                if use_cuda:
                    inputs = move_to_cuda(inputs, use_cuda)
                logits = prompt_model(inputs)
                labels = prepare_labels(inputs['tgt_text'], use_cuda)
                loss = loss_func_no_ewc.focal_label_smooth_ce_loss(logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(prompt_model.state_dict(), 
                      os.path.join(checkpoint_dir, f'task_{task_id}_phase1_best.ckpt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    progress_bar.close()


def train_phase_two(prompt_model, train_dataloader, val_dataloader,
                   optimizer1, optimizer2, scheduler1, scheduler2,
                   num_epochs, loss_func_with_ewc, task_id,
                   checkpoint_dir, patience=5, use_cuda=True):
    """Phase 2: Train with Focal Loss + Label Smoothing + EWC."""
    best_val_loss = float('inf')
    patience_counter = 0
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)),
                       desc=f"Task {task_id} Phase 2")

    for epoch in range(num_epochs):
        prompt_model.train()
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = move_to_cuda(inputs, use_cuda)
            logits = prompt_model(inputs)
            labels = prepare_labels(inputs['tgt_text'], use_cuda)
            
            loss = loss_func_with_ewc(prompt_model, logits, labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()
            prompt_model.zero_grad()
            progress_bar.update(1)

        # Validation
        prompt_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_dataloader:
                if use_cuda:
                    inputs = move_to_cuda(inputs, use_cuda)
                logits = prompt_model(inputs)
                labels = prepare_labels(inputs['tgt_text'], use_cuda)
                loss = loss_func_with_ewc(prompt_model, logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(prompt_model.state_dict(),
                      os.path.join(checkpoint_dir, f'task_{task_id}_phase2_best.ckpt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    progress_bar.close()


def train_consolidation(prompt_model, cons_dataloader, optimizer1, optimizer2,
                       scheduler1, scheduler2, steps, loss_func_with_ewc,
                       task_id, checkpoint_dir, use_cuda=True):
    """Consolidation Phase: Train on buffer only to reduce forgetting."""
    print(f"\nConsolidation Phase for Task {task_id}: {steps} steps")
    prompt_model.train()
    progress_bar = tqdm(range(steps), desc=f"Task {task_id} Consolidation")
    
    cons_it = iter(cons_dataloader)
    
    for step in range(steps):
        try:
            inputs = next(cons_it)
        except StopIteration:
            cons_it = iter(cons_dataloader)
            inputs = next(cons_it)
        
        if use_cuda:
            inputs = move_to_cuda(inputs, use_cuda)
        logits = prompt_model(inputs)
        labels = prepare_labels(inputs['tgt_text'], use_cuda)
        
        loss = loss_func_with_ewc(prompt_model, logits, labels)
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        scheduler1.step()
        scheduler2.step()
        prompt_model.zero_grad()
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Save consolidation checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(prompt_model.state_dict(),
              os.path.join(checkpoint_dir, f'task_{task_id}_cons.ckpt'))
