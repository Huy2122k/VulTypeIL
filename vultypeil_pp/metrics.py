"""
Continual Learning Metrics for VulTypeIL++
"""
import numpy as np
import pandas as pd
from typing import Dict, List
import os


class ContinualMetrics:
    """Track and compute continual learning metrics."""
    
    def __init__(self, num_tasks: int, num_classes: int):
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        # Accuracy matrix: acc[i][j] = accuracy on task j after training task i
        self.acc_matrix = np.zeros((num_tasks, num_tasks))
        self.f1_matrix = np.zeros((num_tasks, num_tasks))
        
    def update(self, task_trained: int, task_eval: int, acc: float, f1: float):
        """
        Update metrics after evaluation.
        
        Args:
            task_trained: Task just trained (1-based)
            task_eval: Task being evaluated (1-based)
            acc: Accuracy score
            f1: F1 score
        """
        self.acc_matrix[task_trained - 1, task_eval - 1] = acc
        self.f1_matrix[task_trained - 1, task_eval - 1] = f1
    
    def compute_forgetting(self) -> Dict[str, float]:
        """
        Compute forgetting metrics.
        
        Forgetting[j] = max_i(Acc[i][j]) - Acc[T][j]
        where T is the final task.
        
        Returns:
            Dictionary with forgetting metrics
        """
        forgetting_per_task = []
        
        for j in range(self.num_tasks):
            # Only compute for tasks that have been trained
            max_acc = np.max(self.acc_matrix[:, j])
            final_acc = self.acc_matrix[-1, j]
            forgetting = max_acc - final_acc
            forgetting_per_task.append(forgetting)
        
        avg_forgetting = np.mean(forgetting_per_task)
        
        return {
            "avg_forgetting": avg_forgetting,
            "forgetting_per_task": forgetting_per_task,
            "task1_final_acc": self.acc_matrix[-1, 0],  # Task 1 at the end
        }
    
    def compute_backward_transfer(self) -> float:
        """
        Compute backward transfer (BWT).
        
        BWT = (1/(T-1)) * sum_{i=1}^{T-1} (Acc[T][i] - Acc[i][i])
        """
        if self.num_tasks == 1:
            return 0.0
        
        bwt = 0.0
        for i in range(self.num_tasks - 1):
            bwt += self.acc_matrix[-1, i] - self.acc_matrix[i, i]
        
        return bwt / (self.num_tasks - 1)
    
    def compute_forward_transfer(self) -> float:
        """
        Compute forward transfer (FWT).
        
        FWT = (1/(T-1)) * sum_{i=2}^{T} (Acc[i-1][i] - Acc_random[i])
        
        Note: Acc_random is typically 1/num_classes for random baseline.
        """
        if self.num_tasks == 1:
            return 0.0
        
        random_acc = 1.0 / self.num_classes
        fwt = 0.0
        
        for i in range(1, self.num_tasks):
            # Acc before training task i (if available)
            if i > 0:
                fwt += self.acc_matrix[i - 1, i] - random_acc
        
        return fwt / (self.num_tasks - 1)
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        forgetting_metrics = self.compute_forgetting()
        
        return {
            "avg_forgetting": forgetting_metrics["avg_forgetting"],
            "task1_final_acc": forgetting_metrics["task1_final_acc"],
            "backward_transfer": self.compute_backward_transfer(),
            "forward_transfer": self.compute_forward_transfer(),
            "final_avg_acc": np.mean(self.acc_matrix[-1, :]),
            "forgetting_per_task": forgetting_metrics["forgetting_per_task"],
        }
    
    def save(self, output_dir: str, exp_name: str):
        """Save metrics to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save accuracy matrix
        acc_df = pd.DataFrame(
            self.acc_matrix,
            index=[f"After_Task_{i+1}" for i in range(self.num_tasks)],
            columns=[f"Task_{j+1}" for j in range(self.num_tasks)]
        )
        acc_df.to_csv(os.path.join(output_dir, f"{exp_name}_acc_matrix.csv"))
        
        # Save F1 matrix
        f1_df = pd.DataFrame(
            self.f1_matrix,
            index=[f"After_Task_{i+1}" for i in range(self.num_tasks)],
            columns=[f"Task_{j+1}" for j in range(self.num_tasks)]
        )
        f1_df.to_csv(os.path.join(output_dir, f"{exp_name}_f1_matrix.csv"))
        
        # Save summary
        summary = self.get_summary()
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(output_dir, f"{exp_name}_summary.csv"), index=False)
        
        # Save forgetting details
        forgetting_df = pd.DataFrame({
            "Task": [f"Task_{i+1}" for i in range(self.num_tasks)],
            "Forgetting": summary["forgetting_per_task"]
        })
        forgetting_df.to_csv(os.path.join(output_dir, f"{exp_name}_forgetting.csv"), index=False)
        
        print(f"\nMetrics saved to {output_dir}/{exp_name}_*.csv")
        print(f"Average Forgetting: {summary['avg_forgetting']:.4f}")
        print(f"Task 1 Final Accuracy: {summary['task1_final_acc']:.4f}")
        print(f"Final Average Accuracy: {summary['final_avg_acc']:.4f}")
