"""
Mixed Batch Iterator for VulTypeIL++
Mixes batches from new task and replay buffer without materializing full dataset.
"""
import torch
from typing import Iterator, Dict, Any


class MixedBatchIterator:
    """
    Iterator that mixes batches from new task loader and memory loader.
    This prevents dataset scaling while maintaining replay ratio.
    """
    
    def __init__(self, new_loader, mem_loader, replay_ratio: float = 0.2, 
                 steps_per_epoch: int = None):
        """
        Args:
            new_loader: DataLoader for current task
            mem_loader: DataLoader for replay buffer
            replay_ratio: Ratio of replay samples in each batch (0-1)
            steps_per_epoch: Number of steps per epoch (default: len(new_loader))
        """
        self.new_loader = new_loader
        self.mem_loader = mem_loader
        self.replay_ratio = replay_ratio
        self.steps = steps_per_epoch or len(new_loader)
    
    def __len__(self):
        return self.steps
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        new_it = iter(self.new_loader)
        mem_it = iter(self.mem_loader)
        
        for _ in range(self.steps):
            # Get batch from new task
            try:
                b_new = next(new_it)
            except StopIteration:
                new_it = iter(self.new_loader)
                b_new = next(new_it)
            
            # Get batch from memory
            try:
                b_mem = next(mem_it)
            except StopIteration:
                mem_it = iter(self.mem_loader)
                b_mem = next(mem_it)
            
            # Mix batches
            yield self._mix_batches(b_new, b_mem, self.replay_ratio)
    
    def _mix_batches(self, b_new: Dict, b_mem: Dict, r: float) -> Dict:
        """
        Mix two batches according to replay ratio.
        
        Args:
            b_new: Batch from new task
            b_mem: Batch from memory
            r: Replay ratio (fraction of batch from memory)
        
        Returns:
            Mixed batch dictionary
        """
        # Determine batch sizes
        # Find a tensor to get batch size
        B = None
        for v in b_new.values():
            if torch.is_tensor(v):
                B = v.size(0)
                break
        
        if B is None:
            # Fallback: no tensors found, return b_new
            return b_new
        
        B_mem = int(round(r * B))
        B_new = B - B_mem
        
        # If no memory samples needed, return new batch
        if B_mem == 0:
            return b_new
        
        # If all memory samples, return memory batch
        if B_new == 0:
            return b_mem
        
        # Mix all fields
        out = {}
        for k in b_new.keys():
            v_new = b_new[k]
            v_mem = b_mem.get(k, None)
            
            if v_mem is None:
                out[k] = v_new
                continue
            
            if torch.is_tensor(v_new) and torch.is_tensor(v_mem):
                # Ensure both tensors are on the same device before concatenating
                if v_new.device != v_mem.device:
                    v_mem = v_mem.to(v_new.device)
                # Concatenate tensors
                out[k] = torch.cat([v_new[:B_new], v_mem[:B_mem]], dim=0)
            elif isinstance(v_new, list) and isinstance(v_mem, list):
                # Concatenate lists
                out[k] = v_new[:B_new] + v_mem[:B_mem]
            else:
                # Keep new if types don't match
                out[k] = v_new
        
        return out


def create_mixed_loader(new_loader, mem_loader, replay_ratio: float = 0.2):
    """
    Convenience function to create a mixed batch iterator.
    
    Args:
        new_loader: DataLoader for current task
        mem_loader: DataLoader for replay buffer
        replay_ratio: Ratio of replay samples in each batch
    
    Returns:
        MixedBatchIterator instance
    """
    return MixedBatchIterator(new_loader, mem_loader, replay_ratio)
