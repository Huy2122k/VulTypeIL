"""
Enhanced Replay Strategy để giải quyết Catastrophic Forgetting cho Task đầu tiên
===============================================================================

Chiến lược mới:
1. Task-Aware Replay: Ưu tiên task cũ hơn
2. Exponential Decay Protection: Bảo vệ task đầu tiên
3. Balanced Task Sampling: Đảm bảo mỗi task được đại diện
4. Forgetting-Aware Selection: Theo dõi performance drop

Tác giả: AI Assistant
"""

import numpy as np
import torch
from collections import Counter, defaultdict
from replay_integration import EnhancedReplaySelector
from replay_config import create_config


class TaskAwareReplaySelector(EnhancedReplaySelector):
    """
    Enhanced Replay Selector với Task-Aware Strategy để chống forgetting
    """
    
    def __init__(self, 
                 similarity_threshold=0.85,
                 max_code_lines=10,
                 n_clusters=10,
                 memory_dir="long_term_memory",
                 use_gradient_importance=False,
                 task_decay_factor=0.7,  # Hệ số ưu tiên task cũ
                 min_task_ratio=0.15):   # Tỷ lệ tối thiểu cho mỗi task
        
        super().__init__(similarity_threshold, max_code_lines, n_clusters, 
                        memory_dir, use_gradient_importance)
        
        self.task_decay_factor = task_decay_factor
        self.min_task_ratio = min_task_ratio
        self.task_performance_history = {}  # Theo dõi performance của từng task
        
    def select_enhanced_replay_samples_with_task_awareness(self, 
                                                         prompt_model, 
                                                         dataloader, 
                                                         examples, 
                                                         task_origins,
                                                         num_samples, 
                                                         current_task_id,
                                                         min_samples_per_class=2,
                                                         current_task_examples=None,
                                                         previous_performance=None):
        """
        Chọn mẫu replay với Task-Aware Strategy
        
        Args:
            prompt_model: Model để trích xuất đặc trưng
            dataloader: DataLoader cho các examples trước đó
            examples: Danh sách InputExample objects
            task_origins: List task origin cho mỗi example
            num_samples: Số lượng mẫu cần chọn
            current_task_id: ID task hiện tại
            min_samples_per_class: Số mẫu tối thiểu mỗi class
            current_task_examples: Examples từ task hiện tại
            previous_performance: Dict performance của các task trước đó
            
        Returns:
            selected_indices: Chỉ số của các mẫu được chọn
            selection_info: Thông tin chi tiết về quá trình chọn
        """
        print(f"\n🎯 TASK-AWARE REPLAY SELECTION CHO TASK {current_task_id}")
        print(f"{'='*70}")
        
        # Lưu performance history
        if previous_performance:
            self.task_performance_history[current_task_id - 1] = previous_performance
        
        # Trích xuất features
        mahalanobis_distances, all_features, all_cwe_ids = self._compute_features(
            prompt_model, dataloader
        )
        
        # Tính toán task priorities với exponential decay
        task_priorities = self._compute_task_priorities(task_origins, current_task_id)
        
        # Phân bổ budget theo task với ưu tiên task cũ
        task_budgets = self._allocate_budget_by_task_priority(
            task_origins, num_samples, task_priorities, current_task_id
        )
        
        print(f"📊 PHÂN BỔ BUDGET THEO TASK:")
        for task_id, budget in task_budgets.items():
            priority = task_priorities.get(task_id, 1.0)
            print(f"  Task {task_id}: {budget} mẫu (ưu tiên: {priority:.2f})")
        
        # Chọn samples theo từng task
        selected_indices = []
        selection_details = {}
        
        for task_id, budget in task_budgets.items():
            if budget == 0:
                continue
                
            # Lấy indices của task này
            task_indices = [i for i, origin in enumerate(task_origins) if origin == task_id]
            
            if len(task_indices) == 0:
                continue
            
            # Chọn samples tốt nhất cho task này
            task_selected = self._select_best_samples_for_task(
                task_indices, mahalanobis_distances, all_cwe_ids, 
                budget, min_samples_per_class
            )
            
            selected_indices.extend(task_selected)
            selection_details[task_id] = {
                'available': len(task_indices),
                'selected': len(task_selected),
                'budget': budget
            }
        
        # Sử dụng scalable replay manager để xử lý semantic filtering
        current_task_vulnerabilities = set()
        if current_task_examples:
            current_task_vulnerabilities = set([ex.tgt_text for ex in current_task_examples])
        
        # Áp dụng semantic filtering và summarization cho selected samples
        selected_examples = [examples[i] for i in selected_indices]
        processed_examples, processing_info = self.replay_manager.process_replay_buffer(
            examples=selected_examples,
            features=np.array(all_features)[selected_indices] if len(all_features) > 0 else [],
            labels=[all_cwe_ids[i] for i in selected_indices],
            task_id=current_task_id,
            replay_budget=len(selected_indices),  # Đã chọn rồi, chỉ cần process
            current_task_vulnerabilities=current_task_vulnerabilities,
            min_samples_per_class=1  # Đã đảm bảo ở trên
        )
        
        # Cập nhật selected_indices sau processing
        final_selected_indices = processing_info.get('selection_indices', list(range(len(processed_examples))))
        final_indices = [selected_indices[i] for i in final_selected_indices]
        
        # Thống kê cuối cùng
        final_task_counts = Counter([task_origins[i] for i in final_indices])
        
        print(f"\n📈 KẾT QUẢ TASK-AWARE SELECTION:")
        print(f"  Tổng mẫu được chọn: {len(final_indices)}")
        for task_id in sorted(final_task_counts.keys()):
            count = final_task_counts[task_id]
            percentage = (count / len(final_indices)) * 100 if len(final_indices) > 0 else 0
            print(f"  Task {task_id}: {count} mẫu ({percentage:.1f}%)")
        
        selection_info = {
            **processing_info,
            'original_count': len(examples),
            'after_task_aware_selection': len(selected_indices),
            'after_processing': len(final_indices),
            'final_selected': len(final_indices),
            'selection_indices': final_indices,
            'task_budgets': task_budgets,
            'task_priorities': task_priorities,
            'selection_details': selection_details,
            'final_task_distribution': dict(final_task_counts),
        }
        
        print(f"{'='*70}\n")
        
        return final_indices, selection_info
    
    def _compute_task_priorities(self, task_origins, current_task_id):
        """
        Tính toán ưu tiên cho từng task với exponential decay
        Task cũ hơn có ưu tiên cao hơn để chống forgetting
        """
        task_priorities = {}
        unique_tasks = set(task_origins)
        
        for task_id in unique_tasks:
            # Exponential decay: task cũ hơn có priority cao hơn
            age = current_task_id - task_id  # Tuổi của task
            priority = (self.task_decay_factor ** age) * 2.0  # Boost cho task cũ
            
            # Thêm bonus cho task đầu tiên để chống forgetting
            if task_id == 1:
                priority *= 1.5  # Task 1 được ưu tiên đặc biệt
            
            # Nếu có performance history, điều chỉnh priority
            if task_id in self.task_performance_history:
                perf = self.task_performance_history[task_id].get('accuracy', 0.8)
                if perf < 0.7:  # Nếu performance thấp, tăng priority
                    priority *= 1.3
            
            task_priorities[task_id] = priority
        
        return task_priorities
    
    def _allocate_budget_by_task_priority(self, task_origins, total_budget, task_priorities, current_task_id):
        """
        Phân bổ budget theo ưu tiên task với đảm bảo minimum cho mỗi task
        """
        task_counts = Counter(task_origins)
        task_budgets = {}
        
        # Phase 1: Đảm bảo minimum cho mỗi task
        min_per_task = max(1, int(total_budget * self.min_task_ratio))
        remaining_budget = total_budget
        
        for task_id in task_counts.keys():
            min_budget = min(min_per_task, task_counts[task_id], remaining_budget)
            task_budgets[task_id] = min_budget
            remaining_budget -= min_budget
        
        # Phase 2: Phân bổ remaining budget theo priority
        if remaining_budget > 0:
            total_priority = sum(task_priorities.values())
            
            for task_id in task_counts.keys():
                if total_priority > 0:
                    priority_ratio = task_priorities[task_id] / total_priority
                    additional_budget = int(remaining_budget * priority_ratio)
                    
                    # Không vượt quá số samples có sẵn
                    max_additional = task_counts[task_id] - task_budgets[task_id]
                    additional_budget = min(additional_budget, max_additional)
                    
                    task_budgets[task_id] += additional_budget
        
        return task_budgets
    
    def _select_best_samples_for_task(self, task_indices, mahalanobis_distances, 
                                    all_cwe_ids, budget, min_samples_per_class):
        """
        Chọn samples tốt nhất cho một task cụ thể
        """
        if len(task_indices) <= budget:
            return task_indices
        
        # Group by class trong task này
        task_class_groups = defaultdict(list)
        for idx in task_indices:
            class_label = all_cwe_ids[idx]
            distance = mahalanobis_distances[idx]
            task_class_groups[class_label].append((idx, distance))
        
        selected = []
        remaining_budget = budget
        
        # Phase 1: Đảm bảo min samples per class
        for class_label, samples in task_class_groups.items():
            n_min = min(min_samples_per_class, len(samples), remaining_budget)
            if n_min > 0:
                # Sort by distance (descending) - higher uncertainty
                samples_sorted = sorted(samples, key=lambda x: x[1], reverse=True)
                selected.extend([idx for idx, _ in samples_sorted[:n_min]])
                remaining_budget -= n_min
                
                # Update remaining samples
                task_class_groups[class_label] = samples_sorted[n_min:]
        
        # Phase 2: Phân bổ remaining budget
        if remaining_budget > 0:
            all_remaining = []
            for samples in task_class_groups.values():
                all_remaining.extend(samples)
            
            # Sort by uncertainty và chọn top
            all_remaining.sort(key=lambda x: x[1], reverse=True)
            selected.extend([idx for idx, _ in all_remaining[:remaining_budget]])
        
        return selected[:budget]  # Đảm bảo không vượt budget


def create_task_aware_replay_selector(config_type='balanced', 
                                    task_decay_factor=0.7,
                                    min_task_ratio=0.15):
    """
    Factory function để tạo Task-Aware Replay Selector
    
    Args:
        config_type: Loại cấu hình ('balanced', 'memory_efficient', etc.)
        task_decay_factor: Hệ số decay cho task priority (0.5-0.9)
        min_task_ratio: Tỷ lệ tối thiểu cho mỗi task (0.1-0.2)
    
    Returns:
        TaskAwareReplaySelector: Configured selector
    """
    config = create_config(config_type)
    
    return TaskAwareReplaySelector(
        similarity_threshold=config.semantic_filter.similarity_threshold,
        max_code_lines=config.code_summarizer.max_code_lines,
        n_clusters=config.clustering.n_clusters,
        memory_dir=config.long_term_memory.memory_dir,
        use_gradient_importance=config.gradient_importance.enabled,
        task_decay_factor=task_decay_factor,
        min_task_ratio=min_task_ratio
    )


# Utility functions
def analyze_forgetting_pattern(task_results):
    """
    Phân tích pattern forgetting từ kết quả các task
    
    Args:
        task_results: Dict {task_id: accuracy}
    
    Returns:
        analysis: Dict với thông tin phân tích
    """
    if not task_results:
        return {}
    
    task_ids = sorted(task_results.keys())
    accuracies = [task_results[tid] for tid in task_ids]
    
    # Tính forgetting cho từng task
    forgetting_scores = {}
    for i, task_id in enumerate(task_ids[:-1]):  # Không tính task cuối
        current_acc = accuracies[i]
        if current_acc < 0.65:  # Threshold cho forgetting nghiêm trọng
            forgetting_scores[task_id] = 'severe'
        elif current_acc < 0.75:
            forgetting_scores[task_id] = 'moderate'
        else:
            forgetting_scores[task_id] = 'mild'
    
    # Tìm task bị forgetting nhiều nhất
    worst_task = min(task_ids[:-1], key=lambda x: task_results[x])
    best_task = max(task_ids, key=lambda x: task_results[x])
    
    analysis = {
        'worst_task': worst_task,
        'worst_accuracy': task_results[worst_task],
        'best_task': best_task,
        'best_accuracy': task_results[best_task],
        'forgetting_scores': forgetting_scores,
        'average_accuracy': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'recommendations': []
    }
    
    # Đưa ra khuyến nghị
    if task_results[worst_task] < 0.6:
        analysis['recommendations'].append(f"Tăng task_decay_factor lên 0.8-0.9 để ưu tiên Task {worst_task}")
        analysis['recommendations'].append(f"Tăng min_task_ratio lên 0.2 để đảm bảo Task {worst_task}")
    
    if analysis['accuracy_std'] > 0.1:
        analysis['recommendations'].append("Tăng min_task_ratio để cân bằng performance các task")
    
    return analysis


if __name__ == "__main__":
    # Test với kết quả hiện tại
    current_results = {
        1: 0.5921,  # Task 1 bị forgetting nghiêm trọng
        2: 0.8476,
        3: 0.7936,
        4: 0.6752,
        5: 0.8194
    }
    
    analysis = analyze_forgetting_pattern(current_results)
    print("🔍 PHÂN TÍCH FORGETTING PATTERN:")
    print(f"Task bị forgetting nhiều nhất: Task {analysis['worst_task']} ({analysis['worst_accuracy']:.2%})")
    print(f"Task tốt nhất: Task {analysis['best_task']} ({analysis['best_accuracy']:.2%})")
    print(f"Độ lệch chuẩn accuracy: {analysis['accuracy_std']:.3f}")
    
    print(f"\n💡 KHUYẾN NGHỊ:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")