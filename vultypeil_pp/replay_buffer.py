"""
Replay Buffer Implementation for VulTypeIL++
Supports multiple update strategies: reservoir, MCSS, GCR-approx
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from typing import List, Dict, Any
from openprompt.data_utils import InputExample


class ReplayBuffer:
    """Fixed-size replay buffer with multiple update strategies."""
    
    def __init__(self, max_size: int = 2000, tail_threshold: float = 0.05):
        self.max_size = max_size
        self.tail_threshold = tail_threshold
        self.examples = []
        self.stream_count = 0  # For reservoir sampling
        
    def __len__(self):
        return len(self.examples)
    
    def add_stream(self, new_examples: List[InputExample]):
        """
        Reservoir sampling update (for random ER).
        Add new examples using reservoir sampling to maintain fixed size.
        """
        for example in new_examples:
            self.stream_count += 1
            if len(self.examples) < self.max_size:
                self.examples.append(example)
            else:
                # Reservoir sampling: replace with probability k/n
                j = random.randint(0, self.stream_count - 1)
                if j < self.max_size:
                    self.examples[j] = example
    
    def update_by_selector(self, candidates: List[InputExample], 
                          selector_fn, **selector_kwargs):
        """
        Update buffer using a selection function (MCSS or GCR).
        
        Args:
            candidates: List of candidate examples (buffer + new)
            selector_fn: Function that selects M examples from candidates
            selector_kwargs: Additional arguments for selector
        """
        selected_examples = selector_fn(candidates, self.max_size, **selector_kwargs)
        self.examples = selected_examples
    
    def sample(self, k: int, mode: str = 'random') -> List[InputExample]:
        """
        Sample k examples from buffer.
        
        Args:
            k: Number of samples
            mode: 'random', 'balanced_tail_head'
        """
        if len(self.examples) == 0:
            return []
        
        k = min(k, len(self.examples))
        
        if mode == 'random':
            return random.sample(self.examples, k)
        
        elif mode == 'balanced_tail_head':
            # Group by class
            class_groups = defaultdict(list)
            for ex in self.examples:
                class_groups[ex.tgt_text].append(ex)
            
            # Identify tail classes
            class_counts = {c: len(exs) for c, exs in class_groups.items()}
            total = sum(class_counts.values())
            tail_classes = [c for c, cnt in class_counts.items() 
                          if cnt / total < self.tail_threshold]
            
            # Allocate quota
            k_tail = int(k * 0.5)  # 50% for tail
            k_head = k - k_tail
            
            selected = []
            
            # Sample tail
            tail_examples = []
            for c in tail_classes:
                tail_examples.extend(class_groups[c])
            if tail_examples:
                selected.extend(random.sample(tail_examples, 
                                            min(k_tail, len(tail_examples))))
            
            # Sample head
            head_examples = []
            for c, exs in class_groups.items():
                if c not in tail_classes:
                    head_examples.extend(exs)
            if head_examples:
                remaining = k - len(selected)
                selected.extend(random.sample(head_examples, 
                                            min(remaining, len(head_examples))))
            
            # Fill up if needed
            if len(selected) < k:
                remaining_examples = [ex for ex in self.examples 
                                    if ex not in selected]
                selected.extend(random.sample(remaining_examples, 
                                            k - len(selected)))
            
            return selected[:k]
        
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if len(self.examples) == 0:
            return {"size": 0}
        
        class_counts = Counter([ex.tgt_text for ex in self.examples])
        total = len(self.examples)
        
        tail_classes = [c for c, cnt in class_counts.items() 
                       if cnt / total < self.tail_threshold]
        
        return {
            "size": len(self.examples),
            "num_classes": len(class_counts),
            "tail_classes": len(tail_classes),
            "class_distribution": dict(class_counts),
            "min_class_count": min(class_counts.values()) if class_counts else 0,
            "max_class_count": max(class_counts.values()) if class_counts else 0,
        }
