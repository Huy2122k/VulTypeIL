"""
Module Tích hợp Dễ dàng cho Cải tiến Scalable Replay
===================================================

Module này cung cấp các hàm tích hợp đơn giản để nâng cấp
vul_main2.py hiện có với khả năng replay có thể mở rộng.

Cách sử dụng:
1. Import module này trong vul_main2.py
2. Thay thế việc chọn replay hiện có bằng phiên bản nâng cao
3. Chỉ cần thay đổi tối thiểu code

Tác giả: AI Assistant
"""

import torch
import numpy as np
from collections import Counter
from scalable_replay_improvements import (
    ScalableReplayManager, 
    create_scalable_replay_manager,
    GradientBasedSampleImportance
)


class EnhancedReplaySelector:
    """
    Thay thế trực tiếp cho việc chọn replay hiện có với các cải tiến có thể mở rộng
    """
    def __init__(self, 
                 similarity_threshold=0.85,
                 max_code_lines=10,
                 n_clusters=10,
                 memory_dir="long_term_memory",
                 use_gradient_importance=False):
        
        self.replay_manager = create_scalable_replay_manager({
            'similarity_threshold': similarity_threshold,
            'max_code_lines': max_code_lines,
            'n_clusters': n_clusters,
            'memory_dir': memory_dir
        })
        
        self.use_gradient_importance = use_gradient_importance
        self.gradient_importance = None
        
    def select_enhanced_replay_samples(self, 
                                     prompt_model, 
                                     dataloader, 
                                     examples, 
                                     num_samples, 
                                     task_id,
                                     min_samples_per_class=2,
                                     current_task_examples=None):
        """
        Chọn mẫu replay nâng cao với tất cả các cải tiến
        
        Args:
            prompt_model: Model để trích xuất đặc trưng
            dataloader: DataLoader cho các examples trước đó
            examples: Danh sách InputExample objects
            num_samples: Số lượng mẫu cần chọn
            task_id: ID task hiện tại
            min_samples_per_class: Số mẫu tối thiểu mỗi class
            current_task_examples: Examples từ task hiện tại (để phân tích vulnerability)
            
        Returns:
            selected_indices: Chỉ số của các mẫu được chọn
            selection_info: Thông tin chi tiết về quá trình chọn
        """
        print(f"\n🚀 CHỌN REPLAY NÂNG CAO CHO TASK {task_id}")
        print(f"{'='*70}")
        
        # Trích xuất features sử dụng tính toán Mahalanobis hiện có
        mahalanobis_distances, all_features, all_cwe_ids = self._compute_features(
            prompt_model, dataloader
        )
        
        # Lấy vulnerabilities của task hiện tại để tính toán ưu tiên
        current_task_vulnerabilities = set()
        if current_task_examples:
            current_task_vulnerabilities = set([ex.tgt_text for ex in current_task_examples])
        
        # Sử dụng scalable replay manager để xử lý
        selected_examples, selection_info = self.replay_manager.process_replay_buffer(
            examples=examples,
            features=np.array(all_features),
            labels=all_cwe_ids,
            task_id=task_id,
            replay_budget=num_samples,
            current_task_vulnerabilities=current_task_vulnerabilities,
            min_samples_per_class=min_samples_per_class
        )
        
        # Ánh xạ ngược về chỉ số ban đầu
        selected_indices = selection_info['selection_indices']
        
        # Tùy chọn: Sử dụng gradient-based importance
        if self.use_gradient_importance and hasattr(self, 'gradient_importance'):
            selected_indices = self._refine_with_gradient_importance(
                selected_indices, mahalanobis_distances, num_samples
            )
        
        # In thống kê chọn lựa
        self._print_selection_stats(selection_info, all_cwe_ids, selected_indices)
        
        return selected_indices, selection_info
    
    def _compute_features(self, prompt_model, dataloader):
        """Tính toán features sử dụng phương pháp Mahalanobis hiện có"""
        prompt_model.eval()
        all_features = []
        all_cwe_ids = []

        with torch.no_grad():
            for inputs in dataloader:
                cwe_ids = inputs['tgt_text']
                if torch.is_tensor(cwe_ids):
                    all_cwe_ids.extend(cwe_ids.cpu().tolist())
                else:
                    all_cwe_ids.extend(cwe_ids)
                    
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    
                logits = prompt_model(inputs)
                all_features.append(logits.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)
        
        # Tính toán Mahalanobis distances để tương thích
        mean_features = np.mean(all_features, axis=0)
        cov_matrix = np.cov(all_features, rowvar=False)
        cov_inv = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)
        
        from scipy.spatial import distance
        mahalanobis_distances = [
            distance.mahalanobis(f, mean_features, cov_inv) for f in all_features
        ]
        
        return mahalanobis_distances, all_features, all_cwe_ids
    
    def _refine_with_gradient_importance(self, selected_indices, mahalanobis_distances, num_samples):
        """Tinh chỉnh việc chọn lựa sử dụng gradient-based importance"""
        if not hasattr(self, 'gradient_norms') or len(self.gradient_norms) == 0:
            return selected_indices
            
        # Kết hợp Mahalanobis distance với gradient importance
        combined_scores = []
        for i in selected_indices:
            mahal_score = mahalanobis_distances[i] if i < len(mahalanobis_distances) else 0
            grad_score = self.gradient_norms[i] if i < len(self.gradient_norms) else 0
            combined_score = 0.7 * mahal_score + 0.3 * grad_score  # Kết hợp có trọng số
            combined_scores.append((i, combined_score))
        
        # Sắp xếp theo điểm kết hợp và chọn top samples
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        refined_indices = [i for i, _ in combined_scores[:num_samples]]
        
        return refined_indices
    
    def _print_selection_stats(self, selection_info, all_cwe_ids, selected_indices):
        """In thống kê chọn lựa chi tiết"""
        print(f"\n📊 THỐNG KÊ CHỌN LỰA:")
        print(f"  Mẫu ban đầu: {selection_info['original_count']}")
        print(f"  Sau lọc: {selection_info['after_filtering']}")
        print(f"  Sau tóm tắt: {selection_info['after_summarization']}")
        print(f"  Cuối cùng được chọn: {selection_info['final_selected']}")
        
        # Phân bố class trong các mẫu được chọn
        selected_labels = [all_cwe_ids[i] for i in selected_indices if i < len(all_cwe_ids)]
        class_dist = Counter(selected_labels)
        
        print(f"\n📈 PHÂN BỐ CLASS TRONG REPLAY BUFFER:")
        for class_label, count in class_dist.most_common(10):
            percentage = (count / len(selected_labels)) * 100 if len(selected_labels) > 0 else 0
            print(f"  Class {class_label}: {count} mẫu ({percentage:.1f}%)")
        
        print(f"{'='*70}\n")
    
    def enable_gradient_importance(self, prompt_model, loss_fn):
        """Bật tính toán gradient-based importance"""
        self.use_gradient_importance = True
        self.gradient_importance = GradientBasedSampleImportance(prompt_model)
        print("✅ Đã bật gradient-based importance")
    
    def get_historical_context(self, task_id):
        """Lấy ngữ cảnh lịch sử để prompting"""
        return self.replay_manager.get_historical_context(task_id)


def upgrade_existing_replay_function():
    """
    Trả về phiên bản nâng cấp của hàm chọn replay hiện có
    
    Cách sử dụng trong vul_main2.py:
        # Thay thế lời gọi hàm hiện có
        # CŨ:
        # indices_to_replay, _ = select_uncertain_samples_with_stratified_class(...)
        
        # MỚI:
        enhanced_selector = upgrade_existing_replay_function()
        indices_to_replay, selection_info = enhanced_selector.select_enhanced_replay_samples(...)
    """
    return EnhancedReplaySelector(
        similarity_threshold=0.85,  # Điều chỉnh theo nhu cầu
        max_code_lines=10,         # Giảm độ dài code để tiết kiệm memory
        n_clusters=10,             # Số clusters để tính toán ưu tiên
        memory_dir="long_term_memory",
        use_gradient_importance=False  # Đặt True để tinh chỉnh dựa trên gradient
    )


# Các hàm hỗ trợ tích hợp
def create_enhanced_template_with_history(original_template_text, historical_context=""):
    """
    Tạo template nâng cao bao gồm ngữ cảnh lịch sử
    
    Args:
        original_template_text: Text template ban đầu
        historical_context: Ngữ cảnh lịch sử từ long-term memory
        
    Returns:
        enhanced_template_text: Template với ngữ cảnh lịch sử
    """
    if not historical_context:
        return original_template_text
    
    enhanced_template = f"""
    {historical_context}
    
    Task hiện tại: {original_template_text}
    """
    
    return enhanced_template


def log_replay_improvements(selection_info, task_id, log_file="replay_improvements.log"):
    """
    Ghi log thống kê cải tiến replay để phân tích
    
    Args:
        selection_info: Thông tin từ việc chọn replay nâng cao
        task_id: ID task hiện tại
        log_file: Đường dẫn file log
    """
    import json
    import datetime
    
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'task_id': task_id,
        'selection_info': selection_info,
        'improvements': {
            'redundancy_reduction': selection_info['original_count'] - selection_info['after_filtering'],
            'summarization_applied': True,
            'clustering_priority_used': True,
            'long_term_memory_stored': True
        }
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


# Ví dụ code tích hợp cho vul_main2.py
INTEGRATION_EXAMPLE = """
# Thêm vào đầu file vul_main2.py sau các imports hiện có
from replay_integration import upgrade_existing_replay_function, log_replay_improvements

# Thay thế code chọn replay hiện có (khoảng dòng 700) bằng:
if i > 1:  # Cho các tasks sau task đầu tiên
    # ... code hiện có cho prev_examples và train_dataloader_prev ...
    
    # CHỌN REPLAY NÂNG CAO - Thay thế select_uncertain_samples_with_stratified_class hiện có
    enhanced_selector = upgrade_existing_replay_function()
    
    # Lấy examples task hiện tại để phân tích vulnerability
    current_examples = read_prompt_examples(data_paths[i - 1])
    
    # Chọn lựa nâng cao với tất cả cải tiến
    indices_to_replay, selection_info = enhanced_selector.select_enhanced_replay_samples(
        prompt_model=prompt_model,
        dataloader=train_dataloader_prev,
        examples=prev_examples,
        num_samples=replay_budget,
        task_id=i,
        min_samples_per_class=args.min_samples_per_class,
        current_task_examples=current_examples
    )
    
    # Ghi log cải tiến để phân tích
    log_replay_improvements(selection_info, i)
    
    # Tùy chọn: Lấy ngữ cảnh lịch sử để prompting nâng cao
    historical_context = enhanced_selector.get_historical_context(i)
    if historical_context:
        print(f"Ngữ cảnh lịch sử cho Task {i}:")
        print(historical_context)
    
    # ... phần còn lại của code hiện có giữ nguyên ...
"""

if __name__ == "__main__":
    print("Module Tích hợp Scalable Replay")
    print("===============================")
    print("\nĐể tích hợp với code hiện có:")
    print(INTEGRATION_EXAMPLE)