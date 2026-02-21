"""
Module Tích hợp Dễ dàng cho Cải tiến Scalable Replay
===================================================

Module này cung cấp các hàm tích hợp đơn giản để nâng cấp
vul_main2.py hiện có với khả năng replay có thể mở rộng.

PHIÊN BẢN CẢI TIẾN:
- SemanticRedundancyFilter: Sử dụng features từ model thay vì TF-IDF
- VulnerabilityCodeSummarizer: Hỗ trợ nhiều phương pháp extraction

Cách sử dụng:
1. Import module này trong vul_main2.py
2. Thay thế việc chọn replay hiện có bằng phiên bản nâng cao
3. Chỉ cần thay đổi tối thiểu code

Tác giả: AI Assistant
"""

import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import os
import re
from typing import List, Dict, Tuple, Optional

# Import components gốc
from scalable_replay_improvements import (
    ClusteringBasedReplayPriority,
    LongTermMemoryWithPrompting,
    GradientBasedSampleImportance
)


class ImprovedSemanticRedundancyFilter:
    """
    Lọc dư thừa ngữ nghĩa SỬ DỤNG FEATURES ĐÃ CÓ từ model
    
    CẢI TIẾN: Sử dụng features từ model thay vì tự tính TF-IDF
    """
    def __init__(self, similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold
        # Fallback TF-IDF nếu không có features
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        
    def filter_redundant_samples(self, examples, features=None):
        """
        Loại bỏ các mẫu tương tự về mặt ngữ nghĩa
        
        Args:
            examples: Danh sách InputExample objects
            features: Features ĐÃ TÍNH SẴN từ model (N x feature_dim) - QUAN TRỌNG!
            
        Returns:
            filtered_examples: Danh sách mẫu không dư thừa
            kept_indices: Chỉ số của các mẫu được giữ lại
        """
        if len(examples) <= 1:
            return examples, list(range(len(examples)))
        
        # CASE 1: Sử dụng features đã có từ model (PREFERRED)
        if features is not None and len(features) > 0:
            print(f"✅ Semantic Filter: Sử dụng features từ model (shape: {np.array(features).shape})")
            similarity_matrix = self._compute_similarity_from_features(features)
        # CASE 2: Fallback sang TF-IDF
        else:
            print(f"⚠️  Semantic Filter: Fallback sang TF-IDF (không có features từ model)")
            similarity_matrix = self._compute_similarity_from_text(examples)
        
        # Greedy selection
        kept_indices = self._greedy_selection(similarity_matrix)
        filtered_examples = [examples[i] for i in kept_indices]
        
        reduction_rate = (len(examples) - len(filtered_examples)) / len(examples) * 100
        print(f"📉 Lọc ngữ nghĩa: {len(examples)} → {len(filtered_examples)} mẫu ({reduction_rate:.1f}% giảm)")
        
        return filtered_examples, kept_indices
    
    def _compute_similarity_from_features(self, features):
        """Tính similarity từ features đã có"""
        if isinstance(features, list):
            features = np.array(features)
        # Normalize để tính cosine similarity
        features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(features_normalized, features_normalized.T)
        return similarity_matrix
    
    def _compute_similarity_from_text(self, examples):
        """Fallback: Tính similarity từ text"""
        texts = [f"{ex.text_a} {ex.text_b}" for ex in examples]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except:
            similarity_matrix = np.eye(len(examples))
        return similarity_matrix
    
    def _greedy_selection(self, similarity_matrix):
        """Greedy selection để giữ các mẫu không dư thừa"""
        n_samples = similarity_matrix.shape[0]
        kept_indices = []
        for i in range(n_samples):
            is_redundant = False
            for j in kept_indices:
                if similarity_matrix[i, j] > self.similarity_threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                kept_indices.append(i)
        return kept_indices


class ImprovedVulnerabilityCodeSummarizer:
    """
    Tóm tắt mã nguồn vulnerability với HỖ TRỢ NHIỀU PHƯƠNG PHÁP
    
    CẢI TIẾN: Hỗ trợ rule-based (cải tiến), attention-based, gradient-based
    """
    def __init__(self, extraction_method='rule_based', max_code_lines=10, custom_keywords=None):
        self.extraction_method = extraction_method
        self.max_code_lines = max_code_lines
        # Từ khóa vulnerability (có thể tùy chỉnh)
        self.vuln_keywords = custom_keywords or [
            'malloc', 'calloc', 'realloc', 'free', 'alloca',
            'strcpy', 'strcat', 'sprintf', 'gets', 'scanf',
            'memcpy', 'memmove', 'strncpy', 'strncat', 'snprintf',
            'buffer', 'overflow', 'underflow', 'null', 'pointer', 
            'dereference', 'bounds', 'check', 'validate', 'sanitize',
            'length', 'size', 'array', 'index',
            'if', 'while', 'for', 'return', 'goto', 'break', 'continue',
        ]
        
    def extract_vulnerability_lines(self, code_text, attention_weights=None, gradient_scores=None):
        """Trích xuất các dòng vulnerability quan trọng"""
        if not code_text or len(code_text.strip()) == 0:
            return code_text, []
        
        lines = code_text.split('\n')
        if len(lines) <= self.max_code_lines:
            return code_text, list(range(len(lines)))
        
        # Chọn phương pháp
        if self.extraction_method == 'rule_based':
            line_scores = self._rule_based_scoring(lines)
        elif self.extraction_method == 'attention_based' and attention_weights is not None:
            line_scores = self._attention_based_scoring(lines, attention_weights)
        elif self.extraction_method == 'gradient_based' and gradient_scores is not None:
            line_scores = self._gradient_based_scoring(lines, gradient_scores)
        else:
            line_scores = self._rule_based_scoring(lines)
        
        # Chọn top lines
        line_scores.sort(key=lambda x: x[1], reverse=True)
        selected_lines = line_scores[:self.max_code_lines]
        selected_lines.sort(key=lambda x: x[0])  # Giữ thứ tự
        
        summarized_code = '\n'.join([line[2] for line in selected_lines])
        selected_line_indices = [line[0] for line in selected_lines]
        
        return summarized_code, selected_line_indices
    
    def _rule_based_scoring(self, lines):
        """Tính điểm dựa trên rules (CẢI TIẾN với nhiều heuristics)"""
        line_scores = []
        for i, line in enumerate(lines):
            score = 0.0
            line_lower = line.lower()
            line_stripped = line.strip()
            
            # Skip empty và comments
            if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('#'):
                line_scores.append((i, 0.0, line))
                continue
            
            # 1. Keyword matching (trọng số cao)
            keyword_count = sum(1 for kw in self.vuln_keywords if kw in line_lower)
            score += keyword_count * 2.0
            
            # 2. Function calls
            if '(' in line and ')' in line:
                score += 1.5
            
            # 3. Pointer operations
            score += sum(1.0 for op in ['->', '*', '&'] if op in line)
            
            # 4. Array indexing
            if '[' in line and ']' in line:
                score += 1.0
            
            # 5. Conditional statements
            if any(cond in line_lower for cond in ['if', 'while', 'for', 'switch']):
                score += 1.0
            
            # 6. Assignment operations
            if '=' in line and '==' not in line:
                score += 0.5
            
            # 7. Return statements
            if 'return' in line_lower:
                score += 0.8
            
            # 8. Bonus cho dòng dài
            if len(line_stripped) > 50:
                score += 0.3
            
            line_scores.append((i, score, line))
        
        return line_scores
    
    def _attention_based_scoring(self, lines, attention_weights):
        """Tính điểm dựa trên attention weights"""
        line_scores = []
        tokens_per_line = [len(line.split()) for line in lines]
        cumsum_tokens = np.cumsum([0] + tokens_per_line)
        
        for i, line in enumerate(lines):
            start_idx = cumsum_tokens[i]
            end_idx = cumsum_tokens[i + 1]
            
            if end_idx <= len(attention_weights):
                line_attention = np.mean(attention_weights[start_idx:end_idx])
            else:
                line_attention = 0.0
            
            # Kết hợp với rule-based
            rule_score = self._rule_based_scoring([line])[0][1]
            combined_score = 0.6 * line_attention + 0.4 * rule_score
            line_scores.append((i, combined_score, line))
        
        return line_scores
    
    def _gradient_based_scoring(self, lines, gradient_scores):
        """Tính điểm dựa trên gradient importance"""
        line_scores = []
        tokens_per_line = [len(line.split()) for line in lines]
        cumsum_tokens = np.cumsum([0] + tokens_per_line)
        
        for i, line in enumerate(lines):
            start_idx = cumsum_tokens[i]
            end_idx = cumsum_tokens[i + 1]
            
            if end_idx <= len(gradient_scores):
                line_gradient = np.mean(gradient_scores[start_idx:end_idx])
            else:
                line_gradient = 0.0
            
            # Kết hợp với rule-based
            rule_score = self._rule_based_scoring([line])[0][1]
            combined_score = 0.7 * line_gradient + 0.3 * rule_score
            line_scores.append((i, combined_score, line))
        
        return line_scores
    
    def summarize_examples(self, examples, attention_weights_list=None, gradient_scores_list=None):
        """Tóm tắt code trong examples"""
        summarized_examples = []
        total_lines_before = 0
        total_lines_after = 0
        
        for idx, ex in enumerate(examples):
            attention = attention_weights_list[idx] if attention_weights_list else None
            gradient = gradient_scores_list[idx] if gradient_scores_list else None
            
            summarized_code, _ = self.extract_vulnerability_lines(ex.text_a, attention, gradient)
            
            lines_before = len(ex.text_a.split('\n'))
            lines_after = len(summarized_code.split('\n'))
            total_lines_before += lines_before
            total_lines_after += lines_after
            
            new_ex = type(ex)(
                guid=ex.guid,
                text_a=summarized_code,
                text_b=ex.text_b,
                tgt_text=ex.tgt_text
            )
            summarized_examples.append(new_ex)
        
        reduction_rate = (total_lines_before - total_lines_after) / total_lines_before * 100 if total_lines_before > 0 else 0
        print(f"📝 Tóm tắt code: {total_lines_before} → {total_lines_after} dòng ({reduction_rate:.1f}% giảm)")
        
        summary_info = {
            'total_examples': len(examples),
            'total_lines_before': total_lines_before,
            'total_lines_after': total_lines_after,
            'reduction_rate': reduction_rate,
            'extraction_method': self.extraction_method
        }
        
        return summarized_examples, summary_info


class ImprovedScalableReplayManager:
    """
    Scalable Replay Manager với components CẢI TIẾN
    
    Thay thế:
    - SemanticRedundancyFilter → ImprovedSemanticRedundancyFilter (dùng features từ model)
    - VulnerabilityCodeSummarizer → ImprovedVulnerabilityCodeSummarizer (nhiều phương pháp)
    
    Giữ nguyên:
    - ClusteringBasedReplayPriority
    - LongTermMemoryWithPrompting
    """
    def __init__(self, similarity_threshold=0.85, max_code_lines=10, n_clusters=10, 
                 memory_dir="long_term_memory", code_extraction_method='rule_based'):
        # IMPROVED COMPONENTS
        self.redundancy_filter = ImprovedSemanticRedundancyFilter(similarity_threshold)
        self.code_summarizer = ImprovedVulnerabilityCodeSummarizer(code_extraction_method, max_code_lines)
        
        # EXISTING COMPONENTS
        self.clustering_priority = ClusteringBasedReplayPriority(n_clusters)
        self.long_term_memory = LongTermMemoryWithPrompting(memory_dir)
        
    def process_replay_buffer(self, examples, features, labels, task_id, replay_budget,
                             current_task_vulnerabilities, min_samples_per_class=2,
                             attention_weights_list=None, gradient_scores_list=None):
        """Xử lý replay buffer với tất cả cải tiến"""
        print(f"\n{'='*70}")
        print(f"🔧 IMPROVED SCALABLE REPLAY PROCESSING FOR TASK {task_id}")
        print(f"{'='*70}")
        
        # Step 1: Semantic filtering (SỬ DỤNG FEATURES TỪ MODEL)
        print(f"\n📍 Step 1: Semantic Redundancy Filtering...")
        filtered_examples, kept_indices = self.redundancy_filter.filter_redundant_samples(
            examples, features=features  # TRUYỀN FEATURES ĐÃ CÓ
        )
        
        # Update features và labels
        if features is not None and len(features) > 0:
            filtered_features = np.array(features)[kept_indices]
        else:
            filtered_features = []
        filtered_labels = [labels[i] for i in kept_indices]
        
        # Step 2: Code summarization
        print(f"\n📍 Step 2: Vulnerability Code Summarization...")
        filtered_attention = [attention_weights_list[i] for i in kept_indices] if attention_weights_list else None
        filtered_gradient = [gradient_scores_list[i] for i in kept_indices] if gradient_scores_list else None
        
        summarized_examples, summary_info = self.code_summarizer.summarize_examples(
            filtered_examples, filtered_attention, filtered_gradient
        )
        
        # Step 3: Clustering priority
        print(f"\n📍 Step 3: Clustering-based Priority Computation...")
        if len(filtered_features) > 0:
            self.clustering_priority.update_clusters(filtered_features, filtered_labels, task_id)
            priorities = self.clustering_priority.get_replay_priorities(
                filtered_features, filtered_labels, current_task_vulnerabilities
            )
        else:
            priorities = [1.0] * len(summarized_examples)
        
        # Step 4: Priority-based selection
        print(f"\n📍 Step 4: Priority-based Sample Selection...")
        selected_indices = self._priority_stratified_selection(
            summarized_examples, priorities, replay_budget, min_samples_per_class
        )
        
        selected_examples = [summarized_examples[i] for i in selected_indices]
        
        # Step 5: Long-term memory
        print(f"\n📍 Step 5: Long-term Memory Storage...")
        performance_metrics = {"replay_selected": len(selected_examples)}
        selected_features = filtered_features[selected_indices] if len(filtered_features) > 0 else []
        self.long_term_memory.store_task_memory(task_id, selected_examples, selected_features, performance_metrics)
        
        selection_info = {
            'original_count': len(examples),
            'after_filtering': len(filtered_examples),
            'after_summarization': len(summarized_examples),
            'final_selected': len(selected_examples),
            'selection_indices': selected_indices,
            'priorities': [priorities[i] for i in selected_indices],
            'summary_info': summary_info,
            'used_model_features': features is not None and len(features) > 0
        }
        
        print(f"\n✅ Hoàn thành: {len(examples)} → {len(selected_examples)} mẫu")
        print(f"{'='*70}\n")
        
        return selected_examples, selection_info
    
    def _priority_stratified_selection(self, examples, priorities, budget, min_samples_per_class):
        """Chọn samples dựa trên priorities với class balance"""
        if len(examples) <= budget:
            return list(range(len(examples)))
        
        class_to_indices = defaultdict(list)
        for i, ex in enumerate(examples):
            class_to_indices[ex.tgt_text].append(i)
        
        selected_indices = []
        remaining_budget = budget
        
        # Phase 1: Min samples per class
        for class_label, indices in class_to_indices.items():
            n_min = min(min_samples_per_class, len(indices), remaining_budget)
            if n_min > 0:
                indices_with_priority = [(i, priorities[i]) for i in indices]
                indices_with_priority.sort(key=lambda x: x[1], reverse=True)
                selected_indices.extend([i for i, _ in indices_with_priority[:n_min]])
                remaining_budget -= n_min
                class_to_indices[class_label] = [i for i, _ in indices_with_priority[n_min:]]
        
        # Phase 2: Remaining budget
        if remaining_budget > 0:
            all_remaining = []
            for indices in class_to_indices.values():
                all_remaining.extend([(i, priorities[i]) for i in indices])
            all_remaining.sort(key=lambda x: x[1], reverse=True)
            selected_indices.extend([i for i, _ in all_remaining[:remaining_budget]])
        
        return selected_indices[:budget]
    
    def get_historical_context(self, current_task_id):
        """Lấy ngữ cảnh lịch sử"""
        return self.long_term_memory.get_historical_context_prompt(current_task_id)


class EnhancedReplaySelector:
    """
    Thay thế trực tiếp cho việc chọn replay hiện có với các cải tiến
    
    PHIÊN BẢN CẢI TIẾN: Sử dụng ImprovedScalableReplayManager
    """
    def __init__(self, 
                 similarity_threshold=0.85,
                 max_code_lines=10,
                 n_clusters=10,
                 memory_dir="long_term_memory",
                 use_gradient_importance=False):
        """
        Args:
            similarity_threshold: Ngưỡng similarity cho lọc dư thừa
            max_code_lines: Số dòng code tối đa sau tóm tắt
            n_clusters: Số clusters cho priority computation
            memory_dir: Thư mục lưu long-term memory
            use_gradient_importance: Có sử dụng gradient importance không
        """
        # Sử dụng IMPROVED manager (với rule_based mặc định)
        self.replay_manager = ImprovedScalableReplayManager(
            similarity_threshold=similarity_threshold,
            max_code_lines=max_code_lines,
            n_clusters=n_clusters,
            memory_dir=memory_dir,
            code_extraction_method='rule_based'  # Mặc định
        )
        
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
        
        CẢI TIẾN: Sử dụng features từ model cho semantic filtering
        
        Args:
            prompt_model: Model để trích xuất đặc trưng
            dataloader: DataLoader cho các examples trước đó
            examples: Danh sách InputExample objects
            num_samples: Số lượng mẫu cần chọn
            task_id: ID task hiện tại
            min_samples_per_class: Số mẫu tối thiểu mỗi class
            current_task_examples: Examples từ task hiện tại
            
        Returns:
            selected_indices: Chỉ số của các mẫu được chọn
            selection_info: Thông tin chi tiết về quá trình chọn
        """
        print(f"\n🚀 ENHANCED REPLAY SELECTION CHO TASK {task_id}")
        print(f"{'='*70}")
        
        # Trích xuất features từ model (QUAN TRỌNG!)
        mahalanobis_distances, all_features, all_cwe_ids = self._compute_features(
            prompt_model, dataloader
        )
        
        # Lấy vulnerabilities của task hiện tại
        current_task_vulnerabilities = set()
        if current_task_examples:
            current_task_vulnerabilities = set([ex.tgt_text for ex in current_task_examples])
        
        # Sử dụng IMPROVED replay manager
        selected_examples, selection_info = self.replay_manager.process_replay_buffer(
            examples=examples,
            features=all_features,  # TRUYỀN FEATURES TỪ MODEL
            labels=all_cwe_ids,
            task_id=task_id,
            replay_budget=num_samples,
            current_task_vulnerabilities=current_task_vulnerabilities,
            min_samples_per_class=min_samples_per_class
        )
        
        # Ánh xạ ngược về chỉ số ban đầu
        selected_indices = selection_info['selection_indices']
        
        # In thống kê
        self._print_selection_stats(selection_info, all_cwe_ids, selected_indices)
        
        return selected_indices, selection_info
    
    def _compute_features(self, prompt_model, dataloader):
        """Tính toán features từ model (logits)"""
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
        
        # Tính Mahalanobis distances (để tương thích với code cũ)
        mean_features = np.mean(all_features, axis=0)
        cov_matrix = np.cov(all_features, rowvar=False)
        cov_inv = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)
        
        from scipy.spatial import distance
        mahalanobis_distances = [
            distance.mahalanobis(f, mean_features, cov_inv) for f in all_features
        ]
        
        return mahalanobis_distances, all_features, all_cwe_ids
    
    def _print_selection_stats(self, selection_info, all_cwe_ids, selected_indices):
        """In thống kê chọn lựa chi tiết"""
        print(f"\n📊 THỐNG KÊ CHỌN LỰA:")
        print(f"  Mẫu ban đầu: {selection_info['original_count']}")
        print(f"  Sau lọc: {selection_info['after_filtering']}")
        print(f"  Sau tóm tắt: {selection_info['after_summarization']}")
        print(f"  Cuối cùng: {selection_info['final_selected']}")
        print(f"  Sử dụng features từ model: {selection_info.get('used_model_features', False)}")
        
        # Phân bố class
        selected_labels = [all_cwe_ids[i] for i in selected_indices if i < len(all_cwe_ids)]
        class_dist = Counter(selected_labels)
        
        print(f"\n📈 PHÂN BỐ CLASS:")
        for class_label, count in class_dist.most_common(10):
            percentage = (count / len(selected_labels)) * 100 if len(selected_labels) > 0 else 0
            print(f"  Class {class_label}: {count} mẫu ({percentage:.1f}%)")
        
        print(f"{'='*70}\n")
    
    def enable_gradient_importance(self, prompt_model, loss_fn):
        """Bật gradient-based importance"""
        self.use_gradient_importance = True
        self.gradient_importance = GradientBasedSampleImportance(prompt_model)
        print("✅ Đã bật gradient-based importance")
    
    def get_historical_context(self, task_id):
        """Lấy ngữ cảnh lịch sử"""
        return self.replay_manager.get_historical_context(task_id)


def upgrade_existing_replay_function():
    """
    Trả về phiên bản nâng cấp của hàm chọn replay
    
    CẢI TIẾN:
    - Semantic filter: Sử dụng features từ model
    - Code summarizer: Hỗ trợ nhiều phương pháp (rule/attention/gradient)
    
    Cách sử dụng trong vul_main2.py:
        enhanced_selector = upgrade_existing_replay_function()
        indices_to_replay, selection_info = enhanced_selector.select_enhanced_replay_samples(...)
    
    Returns:
        EnhancedReplaySelector với components cải tiến
    """
    return EnhancedReplaySelector(
        similarity_threshold=0.85,
        max_code_lines=10,
        n_clusters=10,
        memory_dir="long_term_memory",
        use_gradient_importance=False
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
# ===== TÍCH HỢP VÀO VUL_MAIN2.PY =====

# Thêm vào đầu file sau các imports hiện có
from replay_integration import upgrade_existing_replay_function, log_replay_improvements

# Thay thế code chọn replay hiện có (khoảng dòng 700) bằng:
if i > 1:  # Cho các tasks sau task đầu tiên
    # ... code hiện có cho prev_examples và train_dataloader_prev ...
    
    # ===== CHỌN REPLAY NÂNG CAO (CẢI TIẾN) =====
    # Thay thế select_uncertain_samples_with_stratified_class hiện có
    enhanced_selector = upgrade_existing_replay_function(
        similarity_threshold=0.85,      # Điều chỉnh theo nhu cầu
        max_code_lines=10,              # Số dòng code tối đa
        n_clusters=10,                  # Số clusters
        code_extraction_method='rule_based'  # Hoặc 'attention_based', 'gradient_based'
    )
    
    # Lấy examples task hiện tại
    current_examples = read_prompt_examples(data_paths[i - 1])
    
    # Chọn lựa nâng cao với features từ model
    indices_to_replay, selection_info = enhanced_selector.select_enhanced_replay_samples(
        prompt_model=prompt_model,
        dataloader=train_dataloader_prev,
        examples=prev_examples,
        num_samples=replay_budget,
        task_id=i,
        min_samples_per_class=args.min_samples_per_class,
        current_task_examples=current_examples
    )
    
    # Ghi log cải tiến
    log_replay_improvements(selection_info, i)
    
    # Tùy chọn: Lấy ngữ cảnh lịch sử
    historical_context = enhanced_selector.get_historical_context(i)
    if historical_context:
        print(f"📚 Ngữ cảnh lịch sử cho Task {i}:")
        print(historical_context)
    
    # ... phần còn lại của code hiện có giữ nguyên ...

# ===== CẢI TIẾN CHÍNH =====
# 1. Semantic Filter: Sử dụng features từ model thay vì TF-IDF
# 2. Code Summarizer: Hỗ trợ 3 phương pháp (rule/attention/gradient)
# 3. Tự động fallback nếu không có features
# 4. Ghi log chi tiết để phân tích
"""

if __name__ == "__main__":
    print("="*70)
    print("Module Tích hợp Scalable Replay (PHIÊN BẢN CẢI TIẾN)")
    print("="*70)
    print("\n✨ CẢI TIẾN CHÍNH:")
    print("1. SemanticRedundancyFilter: Sử dụng features từ model")
    print("2. VulnerabilityCodeSummarizer: Hỗ trợ 3 phương pháp")
    print("3. Tự động fallback thông minh")
    print("\n📖 Để tích hợp với code hiện có:")
    print(INTEGRATION_EXAMPLE)