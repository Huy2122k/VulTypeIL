"""
Cấu hình cho Cải tiến Scalable Replay
====================================

Cấu hình tập trung cho tất cả tham số cải tiến replay.
Cho phép điều chỉnh dễ dàng mà không cần sửa đổi core code.

Tác giả: AI Assistant
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SemanticFilterConfig:
    """Cấu hình cho lọc dư thừa ngữ nghĩa"""
    similarity_threshold: float = 0.85  # Ngưỡng cosine similarity cho dư thừa
    max_features: int = 5000           # Số features TF-IDF tối đa
    ngram_range: tuple = (1, 2)        # Phạm vi N-gram cho TF-IDF
    
    def __post_init__(self):
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold phải nằm trong khoảng 0.0 đến 1.0")


@dataclass
class CodeSummarizerConfig:
    """Cấu hình cho tóm tắt mã nguồn vulnerability"""
    max_code_lines: int = 10           # Số dòng tối đa để giữ lại mỗi mẫu code
    vulnerability_keywords: List[str] = None  # Từ khóa vulnerability tùy chỉnh
    
    def __post_init__(self):
        if self.vulnerability_keywords is None:
            self.vulnerability_keywords = [
                'malloc', 'free', 'strcpy', 'strcat', 'sprintf', 'gets', 'scanf',
                'memcpy', 'memmove', 'strncpy', 'strncat', 'snprintf',
                'buffer', 'overflow', 'underflow', 'null', 'pointer', 'dereference',
                'bounds', 'check', 'validate', 'sanitize', 'length', 'size',
                'if', 'while', 'for', 'return', 'goto', 'break', 'continue'
            ]


@dataclass
class ClusteringConfig:
    """Cấu hình cho ưu tiên replay dựa trên clustering"""
    n_clusters: int = 10               # Số clusters cho K-means
    random_state: int = 42             # Random seed để tái tạo kết quả
    rarity_boost_factor: float = 1.5   # Hệ số tăng cho vulnerabilities hiếm
    current_task_boost: float = 1.5    # Hệ số tăng cho vulnerabilities task hiện tại


@dataclass
class GradientImportanceConfig:
    """Cấu hình cho đánh giá tầm quan trọng dựa trên gradient"""
    enabled: bool = False              # Có sử dụng gradient importance không
    weight_in_combination: float = 0.3 # Trọng số khi kết hợp với các điểm khác
    batch_size_for_computation: int = 1 # Batch size cho tính toán gradient


@dataclass
class LongTermMemoryConfig:
    """Cấu hình cho lưu trữ bộ nhớ dài hạn"""
    memory_dir: str = "long_term_memory"  # Thư mục lưu trữ memory
    max_summary_length: int = 500         # Độ dài tối đa cho tóm tắt task
    store_features: bool = True           # Có lưu trữ feature representations không
    compress_features: bool = True        # Có nén features đã lưu không


@dataclass
class ReplaySelectionConfig:
    """Cấu hình cho chiến lược chọn replay tổng thể"""
    min_samples_per_class: int = 2      # Số mẫu tối thiểu mỗi class trong replay
    stratified_selection: bool = True    # Sử dụng chọn lựa phân tầng theo class
    priority_weight: float = 0.7         # Trọng số cho chọn lựa dựa trên ưu tiên
    diversity_weight: float = 0.3        # Trọng số cho đa dạng trong chọn lựa


@dataclass
class ScalableReplayConfig:
    """Main configuration class combining all sub-configurations"""
    semantic_filter: SemanticFilterConfig = None
    code_summarizer: CodeSummarizerConfig = None
    clustering: ClusteringConfig = None
    gradient_importance: GradientImportanceConfig = None
    long_term_memory: LongTermMemoryConfig = None
    replay_selection: ReplaySelectionConfig = None
    
    # Global settings
    verbose: bool = True                 # Enable verbose logging
    save_intermediate_results: bool = True  # Save intermediate processing results
    
    def __post_init__(self):
        if self.semantic_filter is None:
            self.semantic_filter = SemanticFilterConfig()
        if self.code_summarizer is None:
            self.code_summarizer = CodeSummarizerConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if self.gradient_importance is None:
            self.gradient_importance = GradientImportanceConfig()
        if self.long_term_memory is None:
            self.long_term_memory = LongTermMemoryConfig()
        if self.replay_selection is None:
            self.replay_selection = ReplaySelectionConfig()
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'semantic_filter': {
                'similarity_threshold': self.semantic_filter.similarity_threshold,
                'max_features': self.semantic_filter.max_features,
                'ngram_range': self.semantic_filter.ngram_range,
            },
            'code_summarizer': {
                'max_code_lines': self.code_summarizer.max_code_lines,
                'vulnerability_keywords': self.code_summarizer.vulnerability_keywords,
            },
            'clustering': {
                'n_clusters': self.clustering.n_clusters,
                'random_state': self.clustering.random_state,
                'rarity_boost_factor': self.clustering.rarity_boost_factor,
                'current_task_boost': self.clustering.current_task_boost,
            },
            'gradient_importance': {
                'enabled': self.gradient_importance.enabled,
                'weight_in_combination': self.gradient_importance.weight_in_combination,
                'batch_size_for_computation': self.gradient_importance.batch_size_for_computation,
            },
            'long_term_memory': {
                'memory_dir': self.long_term_memory.memory_dir,
                'max_summary_length': self.long_term_memory.max_summary_length,
                'store_features': self.long_term_memory.store_features,
                'compress_features': self.long_term_memory.compress_features,
            },
            'replay_selection': {
                'min_samples_per_class': self.replay_selection.min_samples_per_class,
                'stratified_selection': self.replay_selection.stratified_selection,
                'priority_weight': self.replay_selection.priority_weight,
                'diversity_weight': self.replay_selection.diversity_weight,
            },
            'global': {
                'verbose': self.verbose,
                'save_intermediate_results': self.save_intermediate_results,
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ScalableReplayConfig':
        """Create configuration from dictionary"""
        config = cls()
        
        if 'semantic_filter' in config_dict:
            sf_config = config_dict['semantic_filter']
            config.semantic_filter = SemanticFilterConfig(
                similarity_threshold=sf_config.get('similarity_threshold', 0.85),
                max_features=sf_config.get('max_features', 5000),
                ngram_range=tuple(sf_config.get('ngram_range', (1, 2)))
            )
        
        if 'code_summarizer' in config_dict:
            cs_config = config_dict['code_summarizer']
            config.code_summarizer = CodeSummarizerConfig(
                max_code_lines=cs_config.get('max_code_lines', 10),
                vulnerability_keywords=cs_config.get('vulnerability_keywords', None)
            )
        
        if 'clustering' in config_dict:
            cl_config = config_dict['clustering']
            config.clustering = ClusteringConfig(
                n_clusters=cl_config.get('n_clusters', 10),
                random_state=cl_config.get('random_state', 42),
                rarity_boost_factor=cl_config.get('rarity_boost_factor', 1.5),
                current_task_boost=cl_config.get('current_task_boost', 1.5)
            )
        
        if 'gradient_importance' in config_dict:
            gi_config = config_dict['gradient_importance']
            config.gradient_importance = GradientImportanceConfig(
                enabled=gi_config.get('enabled', False),
                weight_in_combination=gi_config.get('weight_in_combination', 0.3),
                batch_size_for_computation=gi_config.get('batch_size_for_computation', 1)
            )
        
        if 'long_term_memory' in config_dict:
            ltm_config = config_dict['long_term_memory']
            config.long_term_memory = LongTermMemoryConfig(
                memory_dir=ltm_config.get('memory_dir', 'long_term_memory'),
                max_summary_length=ltm_config.get('max_summary_length', 500),
                store_features=ltm_config.get('store_features', True),
                compress_features=ltm_config.get('compress_features', True)
            )
        
        if 'replay_selection' in config_dict:
            rs_config = config_dict['replay_selection']
            config.replay_selection = ReplaySelectionConfig(
                min_samples_per_class=rs_config.get('min_samples_per_class', 2),
                stratified_selection=rs_config.get('stratified_selection', True),
                priority_weight=rs_config.get('priority_weight', 0.7),
                diversity_weight=rs_config.get('diversity_weight', 0.3)
            )
        
        if 'global' in config_dict:
            global_config = config_dict['global']
            config.verbose = global_config.get('verbose', True)
            config.save_intermediate_results = global_config.get('save_intermediate_results', True)
        
        return config
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ScalableReplayConfig':
        """Load configuration from JSON file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Cấu hình định sẵn cho các tình huống khác nhau
def get_memory_efficient_config() -> ScalableReplayConfig:
    """Cấu hình tối ưu cho hiệu quả bộ nhớ"""
    config = ScalableReplayConfig()
    config.semantic_filter.similarity_threshold = 0.9  # Lọc tích cực hơn
    config.code_summarizer.max_code_lines = 5         # Tóm tắt code ngắn hơn
    config.clustering.n_clusters = 5                  # Ít clusters hơn
    config.long_term_memory.compress_features = True  # Bật nén
    return config


def get_quality_focused_config() -> ScalableReplayConfig:
    """Cấu hình tối ưu cho chất lượng replay"""
    config = ScalableReplayConfig()
    config.semantic_filter.similarity_threshold = 0.75  # Lọc ít tích cực hơn
    config.code_summarizer.max_code_lines = 15         # Tóm tắt code dài hơn
    config.clustering.n_clusters = 15                  # Nhiều clusters để chính xác hơn
    config.gradient_importance.enabled = True          # Bật gradient importance
    return config


def get_balanced_config() -> ScalableReplayConfig:
    """Cấu hình cân bằng cho sử dụng chung"""
    return ScalableReplayConfig()  # Sử dụng giá trị mặc định


def get_fast_config() -> ScalableReplayConfig:
    """Cấu hình tối ưu cho tốc độ"""
    config = ScalableReplayConfig()
    config.semantic_filter.max_features = 1000        # Ít TF-IDF features hơn
    config.clustering.n_clusters = 5                  # Ít clusters hơn
    config.gradient_importance.enabled = False        # Tắt tính toán gradient
    config.long_term_memory.store_features = False    # Không lưu features
    return config


# Factory cấu hình
def create_config(config_type: str = "balanced") -> ScalableReplayConfig:
    """
    Hàm factory để tạo cấu hình
    
    Args:
        config_type: Loại cấu hình ('memory_efficient', 'quality_focused', 
                    'balanced', 'fast')
    
    Returns:
        ScalableReplayConfig: Instance đã cấu hình
    """
    config_map = {
        'memory_efficient': get_memory_efficient_config,
        'quality_focused': get_quality_focused_config,
        'balanced': get_balanced_config,
        'fast': get_fast_config
    }
    
    if config_type not in config_map:
        raise ValueError(f"Loại config không xác định: {config_type}. "
                        f"Các loại có sẵn: {list(config_map.keys())}")
    
    return config_map[config_type]()


if __name__ == "__main__":
    # Ví dụ sử dụng
    print("Ví dụ Cấu hình Scalable Replay")
    print("=" * 50)
    
    # Tạo các cấu hình khác nhau
    configs = {
        'cân bằng': create_config('balanced'),
        'tiết kiệm bộ nhớ': create_config('memory_efficient'),
        'tập trung chất lượng': create_config('quality_focused'),
        'nhanh': create_config('fast')
    }
    
    for name, config in configs.items():
        print(f"\nCấu hình {name.upper()}:")
        print(f"  Ngưỡng tương tự: {config.semantic_filter.similarity_threshold}")
        print(f"  Số dòng code tối đa: {config.code_summarizer.max_code_lines}")
        print(f"  Số clusters: {config.clustering.n_clusters}")
        print(f"  Gradient importance: {config.gradient_importance.enabled}")
    
    # Lưu cấu hình ví dụ
    example_config = create_config('balanced')
    example_config.save_to_file('replay_config_example.json')
    print(f"\nCấu hình ví dụ đã lưu vào 'replay_config_example.json'")