"""
Cải tiến Scalable Replay cho VulTypeIL
=====================================

Triển khai các cơ chế replay tiên tiến bao gồm:
1. Lọc thông tin dư thừa về mặt ngữ nghĩa
2. Tóm tắt mã nguồn vulnerability  
3. Ưu tiên replay dựa trên clustering
4. Bộ nhớ dài hạn với prompting
5. Đánh giá tầm quan trọng dựa trên gradient

Tác giả: AI Assistant
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import json
from typing import List, Dict, Tuple, Optional
import re


class SemanticRedundancyFilter:
    """
    Lọc các mẫu dư thừa về mặt ngữ nghĩa sử dụng TF-IDF + cosine similarity
    """
    def __init__(self, similarity_threshold=0.85, max_features=5000):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def filter_redundant_samples(self, examples, features=None):
        """
        Loại bỏ các mẫu tương tự về mặt ngữ nghĩa để giảm dư thừa
        
        Args:
            examples: Danh sách các InputExample objects
            features: Features đã tính sẵn (tùy chọn)
            
        Returns:
            filtered_examples: Danh sách các mẫu không dư thừa
            kept_indices: Chỉ số của các mẫu được giữ lại
        """
        if len(examples) <= 1:
            return examples, list(range(len(examples)))
            
        # Trích xuất text để tính toán độ tương tự
        texts = []
        for ex in examples:
            combined_text = f"{ex.text_a} {ex.text_b}"
            texts.append(combined_text)
        
        # Tính toán TF-IDF features
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except:
            # Fallback nếu TF-IDF thất bại
            return examples, list(range(len(examples)))
        
        # Tìm các mẫu dư thừa
        kept_indices = []
        for i in range(len(examples)):
            is_redundant = False
            for j in kept_indices:
                if similarity_matrix[i, j] > self.similarity_threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                kept_indices.append(i)
        
        filtered_examples = [examples[i] for i in kept_indices]
        
        print(f"Lọc ngữ nghĩa: {len(examples)} → {len(filtered_examples)} mẫu "
              f"({len(examples) - len(filtered_examples)} mẫu dư thừa đã loại bỏ)")
        
        return filtered_examples, kept_indices


class VulnerabilityCodeSummarizer:
    """
    Tóm tắt mã nguồn vulnerability bằng cách trích xuất các dòng quan trọng
    """
    def __init__(self):
        # Từ khóa liên quan đến vulnerability
        self.vuln_keywords = [
            'malloc', 'free', 'strcpy', 'strcat', 'sprintf', 'gets', 'scanf',
            'memcpy', 'memmove', 'strncpy', 'strncat', 'snprintf',
            'buffer', 'overflow', 'underflow', 'null', 'pointer', 'dereference',
            'bounds', 'check', 'validate', 'sanitize', 'length', 'size',
            'if', 'while', 'for', 'return', 'goto', 'break', 'continue'
        ]
        
    def extract_vulnerability_lines(self, code_text, max_lines=10):
        """
        Trích xuất các dòng có khả năng chứa vulnerability patterns cao nhất
        
        Args:
            code_text: Chuỗi mã nguồn
            max_lines: Số dòng tối đa để giữ lại
            
        Returns:
            summarized_code: Các dòng vulnerability quan trọng
        """
        if not code_text or len(code_text.strip()) == 0:
            return code_text
            
        lines = code_text.split('\n')
        if len(lines) <= max_lines:
            return code_text
            
        # Tính điểm cho các dòng dựa trên từ khóa vulnerability
        line_scores = []
        for i, line in enumerate(lines):
            score = 0
            line_lower = line.lower()
            
            # Khớp từ khóa
            for keyword in self.vuln_keywords:
                if keyword in line_lower:
                    score += 1
                    
            # Tăng điểm cho dòng có function calls
            if '(' in line and ')' in line:
                score += 0.5
                
            # Tăng điểm cho dòng có pointer operations
            if '->' in line or '*' in line or '&' in line:
                score += 0.5
                
            # Tăng điểm cho conditional statements
            if any(cond in line_lower for cond in ['if', 'while', 'for']):
                score += 0.3
                
            line_scores.append((i, score, line))
        
        # Sắp xếp theo điểm và chọn top lines
        line_scores.sort(key=lambda x: x[1], reverse=True)
        selected_lines = line_scores[:max_lines]
        
        # Sắp xếp các dòng đã chọn theo thứ tự ban đầu
        selected_lines.sort(key=lambda x: x[0])
        
        summarized_code = '\n'.join([line[2] for line in selected_lines])
        return summarized_code
    
    def summarize_examples(self, examples, max_code_lines=10):
        """
        Tóm tắt code trong examples để giảm memory footprint
        
        Args:
            examples: Danh sách InputExample objects
            max_code_lines: Số dòng code tối đa để giữ lại cho mỗi example
            
        Returns:
            summarized_examples: Examples với code đã được tóm tắt
        """
        summarized_examples = []
        
        for ex in examples:
            summarized_code = self.extract_vulnerability_lines(ex.text_a, max_code_lines)
            
            # Tạo example mới với code đã tóm tắt
            new_ex = type(ex)(
                guid=ex.guid,
                text_a=summarized_code,
                text_b=ex.text_b,  # Giữ nguyên description
                tgt_text=ex.tgt_text
            )
            summarized_examples.append(new_ex)
            
        return summarized_examples


class ClusteringBasedReplayPriority:
    """
    Triển khai ưu tiên replay dựa trên clustering với theo dõi tần suất vulnerability
    """
    def __init__(self, n_clusters=10, feature_dim=768):
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_vulnerability_freq = defaultdict(Counter)
        self.task_cluster_mapping = {}
        
    def update_clusters(self, features, labels, task_id):
        """
        Cập nhật clusters với dữ liệu task mới và theo dõi tần suất vulnerability
        
        Args:
            features: Biểu diễn đặc trưng (N x feature_dim)
            labels: Nhãn vulnerability
            task_id: ID task hiện tại
        """
        if len(features) == 0:
            return
            
        # Fit clustering trên features hiện tại
        cluster_labels = self.kmeans.fit_predict(features)
        
        # Cập nhật tần suất vulnerability cho mỗi cluster
        for cluster_id, vuln_label in zip(cluster_labels, labels):
            self.cluster_vulnerability_freq[cluster_id][vuln_label] += 1
            
        # Lưu trữ mapping task-cluster
        self.task_cluster_mapping[task_id] = {
            'cluster_labels': cluster_labels,
            'vulnerability_labels': labels
        }
        
        print(f"Task {task_id}: Đã cập nhật {self.n_clusters} clusters với {len(features)} mẫu")
        
    def get_replay_priorities(self, features, labels, current_task_vulnerabilities):
        """
        Tính toán ưu tiên replay dựa trên tần suất vulnerability trong cluster
        
        Args:
            features: Features lịch sử
            labels: Nhãn lịch sử  
            current_task_vulnerabilities: Vulnerabilities trong task hiện tại
            
        Returns:
            priorities: Điểm ưu tiên cho mỗi mẫu
        """
        if len(features) == 0:
            return []
            
        # Dự đoán cluster assignments cho dữ liệu lịch sử
        cluster_assignments = self.kmeans.predict(features)
        
        priorities = []
        for cluster_id, vuln_label in zip(cluster_assignments, labels):
            # Ưu tiên cơ bản từ tần suất vulnerability trong cluster
            cluster_freq = self.cluster_vulnerability_freq[cluster_id]
            total_cluster_samples = sum(cluster_freq.values())
            
            if total_cluster_samples == 0:
                priority = 0.5  # Ưu tiên mặc định
            else:
                # Ưu tiên dựa trên độ hiếm trong cluster
                vuln_freq_in_cluster = cluster_freq[vuln_label]
                rarity_score = 1.0 - (vuln_freq_in_cluster / total_cluster_samples)
                
                # Tăng ưu tiên nếu vulnerability xuất hiện trong task hiện tại
                current_task_boost = 1.5 if vuln_label in current_task_vulnerabilities else 1.0
                
                priority = rarity_score * current_task_boost
                
            priorities.append(priority)
            
        return priorities


class GradientBasedSampleImportance:
    """
    Computes sample importance based on gradient norms for replay selection
    """
    def __init__(self, model):
        self.model = model
        
    def compute_gradient_norms(self, dataloader, loss_fn):
        """
        Compute gradient norms for each sample to measure importance
        
        Args:
            dataloader: DataLoader with samples
            loss_fn: Loss function
            
        Returns:
            gradient_norms: List of gradient norms per sample
        """
        self.model.eval()
        gradient_norms = []
        
        for batch_idx, inputs in enumerate(dataloader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                
            # Forward pass
            logits = self.model(inputs)
            labels = inputs['tgt_text']
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels)
            if torch.cuda.is_available():
                labels = labels.cuda()
                
            # Compute loss for each sample individually
            for i in range(len(labels)):
                self.model.zero_grad()
                
                # Single sample loss
                sample_logits = logits[i:i+1]
                sample_labels = labels[i:i+1]
                loss = loss_fn(sample_logits, sample_labels)
                
                # Backward pass
                loss.backward(retain_graph=True)
                
                # Compute gradient norm
                total_norm = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                gradient_norms.append(total_norm)
                
        return gradient_norms


class LongTermMemoryWithPrompting:
    """
    Implements long-term memory storage with prompting for historical context
    """
    def __init__(self, memory_dir="long_term_memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        self.task_summaries = {}
        
    def store_task_memory(self, task_id, examples, features, performance_metrics):
        """
        Store task memory with examples, features, and performance
        
        Args:
            task_id: Task identifier
            examples: List of InputExample objects
            features: Feature representations
            performance_metrics: Task performance metrics
        """
        memory_data = {
            'task_id': task_id,
            'examples': examples,
            'features': features.tolist() if isinstance(features, np.ndarray) else features,
            'performance_metrics': performance_metrics,
            'vulnerability_distribution': Counter([ex.tgt_text for ex in examples])
        }
        
        # Save to disk
        memory_file = os.path.join(self.memory_dir, f"task_{task_id}_memory.pkl")
        with open(memory_file, 'wb') as f:
            pickle.dump(memory_data, f)
            
        # Create task summary for prompting
        self.task_summaries[task_id] = self._create_task_summary(memory_data)
        
        print(f"Stored long-term memory for Task {task_id}")
        
    def _create_task_summary(self, memory_data):
        """Create a textual summary of task for prompting"""
        vuln_dist = memory_data['vulnerability_distribution']
        top_vulnerabilities = vuln_dist.most_common(3)
        
        summary = f"Task {memory_data['task_id']} Summary:\n"
        summary += f"- Total samples: {len(memory_data['examples'])}\n"
        summary += f"- Top vulnerabilities: {', '.join([f'{v}({c})' for v, c in top_vulnerabilities])}\n"
        summary += f"- Performance: {memory_data['performance_metrics']}\n"
        
        return summary
        
    def get_historical_context_prompt(self, current_task_id):
        """
        Generate prompting context from historical tasks
        
        Args:
            current_task_id: Current task ID
            
        Returns:
            context_prompt: Historical context for prompting
        """
        if len(self.task_summaries) == 0:
            return ""
            
        context_prompt = "Historical Context:\n"
        for task_id in sorted(self.task_summaries.keys()):
            if task_id < current_task_id:
                context_prompt += self.task_summaries[task_id] + "\n"
                
        return context_prompt
        
    def load_task_memory(self, task_id):
        """Load task memory from disk"""
        memory_file = os.path.join(self.memory_dir, f"task_{task_id}_memory.pkl")
        if os.path.exists(memory_file):
            with open(memory_file, 'rb') as f:
                return pickle.load(f)
        return None


class ScalableReplayManager:
    """
    Main class that integrates all replay improvements
    """
    def __init__(self, 
                 similarity_threshold=0.85,
                 max_code_lines=10,
                 n_clusters=10,
                 memory_dir="long_term_memory"):
        
        self.redundancy_filter = SemanticRedundancyFilter(similarity_threshold)
        self.code_summarizer = VulnerabilityCodeSummarizer()
        self.clustering_priority = ClusteringBasedReplayPriority(n_clusters)
        self.long_term_memory = LongTermMemoryWithPrompting(memory_dir)
        
    def process_replay_buffer(self, 
                            examples, 
                            features, 
                            labels,
                            task_id,
                            replay_budget,
                            current_task_vulnerabilities,
                            min_samples_per_class=2):
        """
        Process replay buffer with all improvements
        
        Args:
            examples: Historical examples
            features: Feature representations
            labels: Vulnerability labels
            task_id: Current task ID
            replay_budget: Number of samples to select
            current_task_vulnerabilities: Vulnerabilities in current task
            min_samples_per_class: Minimum samples per class
            
        Returns:
            selected_examples: Processed and selected examples
            selection_info: Information about selection process
        """
        print(f"\n{'='*60}")
        print(f"SCALABLE REPLAY PROCESSING FOR TASK {task_id}")
        print(f"{'='*60}")
        
        # Step 1: Semantic redundancy filtering
        print("Step 1: Semantic Redundancy Filtering...")
        filtered_examples, kept_indices = self.redundancy_filter.filter_redundant_samples(examples)
        filtered_features = features[kept_indices] if len(features) > 0 else []
        filtered_labels = [labels[i] for i in kept_indices]
        
        # Step 2: Code summarization
        print("Step 2: Vulnerability Code Summarization...")
        summarized_examples = self.code_summarizer.summarize_examples(filtered_examples)
        
        # Step 3: Update clustering and compute priorities
        print("Step 3: Clustering-based Priority Computation...")
        if len(filtered_features) > 0:
            self.clustering_priority.update_clusters(filtered_features, filtered_labels, task_id)
            priorities = self.clustering_priority.get_replay_priorities(
                filtered_features, filtered_labels, current_task_vulnerabilities
            )
        else:
            priorities = [1.0] * len(summarized_examples)
        
        # Step 4: Priority-based selection with stratification
        print("Step 4: Priority-based Sample Selection...")
        selected_indices = self._priority_stratified_selection(
            summarized_examples, priorities, replay_budget, min_samples_per_class
        )
        
        selected_examples = [summarized_examples[i] for i in selected_indices]
        
        # Step 5: Store in long-term memory
        print("Step 5: Long-term Memory Storage...")
        performance_metrics = {"replay_selected": len(selected_examples)}
        self.long_term_memory.store_task_memory(
            task_id, selected_examples, 
            filtered_features[selected_indices] if len(filtered_features) > 0 else [],
            performance_metrics
        )
        
        selection_info = {
            'original_count': len(examples),
            'after_filtering': len(filtered_examples),
            'after_summarization': len(summarized_examples),
            'final_selected': len(selected_examples),
            'selection_indices': selected_indices,
            'priorities': [priorities[i] for i in selected_indices]
        }
        
        print(f"Replay processing complete: {len(examples)} → {len(selected_examples)} samples")
        print(f"{'='*60}\n")
        
        return selected_examples, selection_info
    
    def _priority_stratified_selection(self, examples, priorities, budget, min_samples_per_class):
        """
        Select samples based on priorities while maintaining class balance
        """
        if len(examples) <= budget:
            return list(range(len(examples)))
            
        # Group by class
        class_to_indices = defaultdict(list)
        for i, ex in enumerate(examples):
            class_to_indices[ex.tgt_text].append(i)
        
        selected_indices = []
        remaining_budget = budget
        
        # Phase 1: Ensure minimum samples per class
        for class_label, indices in class_to_indices.items():
            n_min = min(min_samples_per_class, len(indices), remaining_budget)
            if n_min > 0:
                # Sort by priority (descending)
                indices_with_priority = [(i, priorities[i]) for i in indices]
                indices_with_priority.sort(key=lambda x: x[1], reverse=True)
                
                selected_indices.extend([i for i, _ in indices_with_priority[:n_min]])
                remaining_budget -= n_min
                
                # Update available indices
                class_to_indices[class_label] = [i for i, _ in indices_with_priority[n_min:]]
        
        # Phase 2: Distribute remaining budget by priority
        if remaining_budget > 0:
            all_remaining = []
            for indices in class_to_indices.values():
                all_remaining.extend([(i, priorities[i]) for i in indices])
            
            # Sort by priority and select top samples
            all_remaining.sort(key=lambda x: x[1], reverse=True)
            selected_indices.extend([i for i, _ in all_remaining[:remaining_budget]])
        
        return selected_indices[:budget]  # Ensure we don't exceed budget
    
    def get_historical_context(self, current_task_id):
        """Get historical context for prompting"""
        return self.long_term_memory.get_historical_context_prompt(current_task_id)


# Utility functions for integration
def create_scalable_replay_manager(config=None):
    """Factory function to create ScalableReplayManager with configuration"""
    if config is None:
        config = {
            'similarity_threshold': 0.85,
            'max_code_lines': 10,
            'n_clusters': 10,
            'memory_dir': 'long_term_memory'
        }
    
    return ScalableReplayManager(**config)


def integrate_with_existing_code(replay_manager, prompt_model, examples, features, labels, 
                                task_id, replay_budget, current_task_vulnerabilities):
    """
    Integration function for existing codebase
    
    Usage in main training loop:
        replay_manager = create_scalable_replay_manager()
        selected_examples, info = integrate_with_existing_code(
            replay_manager, prompt_model, prev_examples, features, labels,
            task_id, replay_budget, current_task_vulnerabilities
        )
    """
    return replay_manager.process_replay_buffer(
        examples, features, labels, task_id, replay_budget, 
        current_task_vulnerabilities
    )