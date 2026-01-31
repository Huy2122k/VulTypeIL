"""
Unit tests for configuration management system.

This module tests the configuration classes, validation, and the main
ReplayOptimizationManager class.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from optimization.config import (
    SemanticFilterConfig, SummarizationConfig, PrioritizationConfig,
    OptimizationConfig, ReplayOptimizationManager
)
from optimization.base import ConfigurationError, OptimizationError


class TestSemanticFilterConfig:
    """Test SemanticFilterConfig data structure."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = SemanticFilterConfig()
        
        assert config.enabled is True
        assert config.similarity_threshold == 0.95
        assert config.min_cluster_size == 2
        assert config.clustering_method == "hierarchical"
        assert config.feature_extraction_method == "embeddings"
        assert config.batch_size == 1000


class TestSummarizationConfig:
    """Test SummarizationConfig data structure."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = SummarizationConfig()
        
        assert config.enabled is True
        assert config.summarization_method == "vulnerability_lines"
        assert config.llm_model is None
        assert config.max_code_length == 2048
        assert config.compression_target == 0.5
        assert config.quality_threshold == 0.8
        assert config.fallback_enabled is True


class TestPrioritizationConfig:
    """Test PrioritizationConfig data structure."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PrioritizationConfig()
        
        assert config.enabled is True
        assert config.clustering_method == "kmeans"
        assert config.n_clusters == 10
        assert config.temporal_weight == 0.3
        assert config.frequency_weight == 0.4
        assert config.diversity_weight == 0.3
        assert config.min_samples_per_cluster == 1


class TestOptimizationConfig:
    """Test OptimizationConfig main configuration class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        
        assert isinstance(config.semantic_filtering, SemanticFilterConfig)
        assert isinstance(config.summarization, SummarizationConfig)
        assert isinstance(config.prioritization, PrioritizationConfig)
        assert config.memory_limit_mb is None
        assert config.performance_threshold == 0.98
        assert config.max_processing_time_ms is None
        assert config.optimization_strategy == "balanced"
        assert config.enable_fallback is True
        assert config.enable_detailed_logging is True
        assert config.log_level == "INFO"
        assert config.metrics_collection_enabled is True
    
    def test_validate_success(self):
        """Test successful configuration validation."""
        config = OptimizationConfig()
        errors = config.validate()
        
        assert errors == []
    
    def test_validate_semantic_filtering_errors(self):
        """Test validation errors for semantic filtering config."""
        config = OptimizationConfig()
        config.semantic_filtering.similarity_threshold = 1.5  # Invalid
        config.semantic_filtering.min_cluster_size = 0  # Invalid
        config.semantic_filtering.clustering_method = "invalid"  # Invalid
        
        errors = config.validate()
        
        assert len(errors) == 3
        assert any("similarity_threshold must be between 0.0 and 1.0" in error for error in errors)
        assert any("min_cluster_size must be >= 1" in error for error in errors)
        assert any("clustering_method must be 'hierarchical' or 'kmeans'" in error for error in errors)
    
    def test_validate_summarization_errors(self):
        """Test validation errors for summarization config."""
        config = OptimizationConfig()
        config.summarization.summarization_method = "invalid"  # Invalid
        config.summarization.compression_target = 1.5  # Invalid
        config.summarization.quality_threshold = -0.1  # Invalid
        
        errors = config.validate()
        
        assert len(errors) >= 3
        assert any("Summarization method must be" in error for error in errors)
        assert any("compression_target must be between 0.0 and 1.0" in error for error in errors)
        assert any("quality_threshold must be between 0.0 and 1.0" in error for error in errors)
    
    def test_validate_llm_model_required(self):
        """Test validation error when LLM model is required but not specified."""
        config = OptimizationConfig()
        config.summarization.summarization_method = "llm"
        config.summarization.llm_model = None
        
        errors = config.validate()
        
        assert any("LLM model must be specified" in error for error in errors)
    
    def test_validate_prioritization_errors(self):
        """Test validation errors for prioritization config."""
        config = OptimizationConfig()
        config.prioritization.clustering_method = "invalid"  # Invalid
        config.prioritization.n_clusters = 0  # Invalid
        config.prioritization.temporal_weight = 0.5
        config.prioritization.frequency_weight = 0.3
        config.prioritization.diversity_weight = 0.3  # Sum > 1.0
        
        errors = config.validate()
        
        assert len(errors) >= 3
        assert any("clustering_method must be" in error for error in errors)
        assert any("n_clusters must be >= 1" in error for error in errors)
        assert any("weights" in error and "sum to 1.0" in error for error in errors)
    
    def test_validate_global_constraint_errors(self):
        """Test validation errors for global constraints."""
        config = OptimizationConfig()
        config.memory_limit_mb = -100  # Invalid
        config.performance_threshold = 1.5  # Invalid
        config.optimization_strategy = "invalid"  # Invalid
        
        errors = config.validate()
        
        assert len(errors) >= 3
        assert any("Memory limit must be positive" in error for error in errors)
        assert any("Performance threshold must be between 0.0 and 1.0" in error for error in errors)
        assert any("Optimization strategy must be" in error for error in errors)
    
    def test_adjust_for_memory_constraint_aggressive(self):
        """Test memory constraint adjustment with aggressive strategy."""
        config = OptimizationConfig()
        config.optimization_strategy = "aggressive"
        config.semantic_filtering.similarity_threshold = 0.95
        config.summarization.compression_target = 0.5
        
        config.adjust_for_memory_constraint(2000)  # 2GB estimated, no limit set
        
        # Should not change without memory limit
        assert config.semantic_filtering.similarity_threshold == 0.95
        assert config.summarization.compression_target == 0.5
        
        # Set memory limit and test adjustment
        config.memory_limit_mb = 1000  # 1GB limit
        config.adjust_for_memory_constraint(2000)  # 2GB estimated
        
        assert config.semantic_filtering.similarity_threshold <= 0.85
        assert config.summarization.compression_target <= 0.3
    
    def test_adjust_for_memory_constraint_balanced(self):
        """Test memory constraint adjustment with balanced strategy."""
        config = OptimizationConfig()
        config.optimization_strategy = "balanced"
        config.memory_limit_mb = 1000
        config.semantic_filtering.similarity_threshold = 0.95
        config.summarization.compression_target = 0.5
        
        config.adjust_for_memory_constraint(2000)
        
        assert config.semantic_filtering.similarity_threshold <= 0.90
        assert config.summarization.compression_target <= 0.4
    
    def test_adjust_for_memory_constraint_conservative(self):
        """Test memory constraint adjustment with conservative strategy."""
        config = OptimizationConfig()
        config.optimization_strategy = "conservative"
        config.memory_limit_mb = 1000
        config.semantic_filtering.similarity_threshold = 0.95
        config.summarization.compression_target = 0.5
        
        config.adjust_for_memory_constraint(2000)
        
        assert config.semantic_filtering.similarity_threshold <= 0.93
        assert config.summarization.compression_target <= 0.6
    
    def test_to_dict_conversion(self):
        """Test converting configuration to dictionary."""
        config = OptimizationConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "semantic_filtering" in config_dict
        assert "summarization" in config_dict
        assert "prioritization" in config_dict
        assert "memory_limit_mb" in config_dict
        assert "performance_threshold" in config_dict
    
    def test_from_dict_creation(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "semantic_filtering": {
                "enabled": False,
                "similarity_threshold": 0.9
            },
            "summarization": {
                "enabled": True,
                "summarization_method": "llm",
                "llm_model": "gpt-3.5-turbo"
            },
            "prioritization": {
                "enabled": True,
                "n_clusters": 5
            },
            "memory_limit_mb": 2048,
            "optimization_strategy": "aggressive"
        }
        
        config = OptimizationConfig.from_dict(config_dict)
        
        assert config.semantic_filtering.enabled is False
        assert config.semantic_filtering.similarity_threshold == 0.9
        assert config.summarization.summarization_method == "llm"
        assert config.summarization.llm_model == "gpt-3.5-turbo"
        assert config.prioritization.n_clusters == 5
        assert config.memory_limit_mb == 2048
        assert config.optimization_strategy == "aggressive"
    
    def test_save_and_load_json(self):
        """Test saving and loading configuration as JSON."""
        config = OptimizationConfig()
        config.memory_limit_mb = 1024
        config.optimization_strategy = "aggressive"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_to_file(temp_path)
            loaded_config = OptimizationConfig.load_from_file(temp_path)
            
            assert loaded_config.memory_limit_mb == 1024
            assert loaded_config.optimization_strategy == "aggressive"
            
        finally:
            Path(temp_path).unlink()
    
    def test_save_and_load_yaml(self):
        """Test saving and loading configuration as YAML."""
        config = OptimizationConfig()
        config.memory_limit_mb = 2048
        config.optimization_strategy = "conservative"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_to_file(temp_path)
            loaded_config = OptimizationConfig.load_from_file(temp_path)
            
            assert loaded_config.memory_limit_mb == 2048
            assert loaded_config.optimization_strategy == "conservative"
            
        finally:
            Path(temp_path).unlink()
    
    def test_load_from_nonexistent_file(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(ConfigurationError) as exc_info:
            OptimizationConfig.load_from_file("nonexistent.json")
        
        assert "Configuration file not found" in str(exc_info.value)
    
    def test_load_from_invalid_file(self):
        """Test loading configuration from invalid file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                OptimizationConfig.load_from_file(temp_path)
            
            assert "Failed to load configuration" in str(exc_info.value)
            
        finally:
            Path(temp_path).unlink()


class TestReplayOptimizationManager:
    """Test ReplayOptimizationManager main class."""
    
    def test_initialization_success(self):
        """Test successful manager initialization."""
        config = OptimizationConfig()
        manager = ReplayOptimizationManager(config)
        
        assert manager.config == config
        assert manager.logger is not None
        assert manager._semantic_filter is None
        assert manager._summarization_engine is None
        assert manager._prioritization_engine is None
    
    def test_initialization_with_invalid_config(self):
        """Test manager initialization with invalid configuration."""
        config = OptimizationConfig()
        config.performance_threshold = 1.5  # Invalid
        
        with pytest.raises(ConfigurationError) as exc_info:
            ReplayOptimizationManager(config)
        
        assert "Configuration validation failed" in str(exc_info.value)
    
    def test_estimate_memory_usage_empty(self):
        """Test memory usage estimation with empty samples."""
        config = OptimizationConfig()
        manager = ReplayOptimizationManager(config)
        
        memory_mb = manager.estimate_memory_usage([])
        
        assert memory_mb == 0
    
    def test_estimate_memory_usage_with_samples(self, sample_input_examples):
        """Test memory usage estimation with sample data."""
        config = OptimizationConfig()
        manager = ReplayOptimizationManager(config)
        
        memory_mb = manager.estimate_memory_usage(sample_input_examples)
        
        assert memory_mb > 0
        assert isinstance(memory_mb, int)
    
    def test_validate_configuration(self):
        """Test configuration validation through manager."""
        config = OptimizationConfig()
        config.memory_limit_mb = -100  # Invalid
        
        # Should not raise during initialization since we're testing validation
        config_copy = OptimizationConfig()
        manager = ReplayOptimizationManager(config_copy)
        
        # Test validation of invalid config
        errors = config.validate()
        assert len(errors) > 0
    
    def test_get_config(self):
        """Test getting configuration from manager."""
        config = OptimizationConfig()
        manager = ReplayOptimizationManager(config)
        
        retrieved_config = manager.get_config()
        
        assert retrieved_config == config
    
    def test_update_config_success(self):
        """Test successful configuration update."""
        config = OptimizationConfig()
        manager = ReplayOptimizationManager(config)
        
        new_config = OptimizationConfig()
        new_config.memory_limit_mb = 2048
        
        manager.update_config(new_config)
        
        assert manager.config.memory_limit_mb == 2048
        # Components should be reset
        assert manager._semantic_filter is None
        assert manager._summarization_engine is None
        assert manager._prioritization_engine is None
    
    def test_update_config_invalid(self):
        """Test configuration update with invalid config."""
        config = OptimizationConfig()
        manager = ReplayOptimizationManager(config)
        
        invalid_config = OptimizationConfig()
        invalid_config.performance_threshold = 1.5  # Invalid
        
        with pytest.raises(ConfigurationError):
            manager.update_config(invalid_config)
    
    @patch('optimization.config.ReplayOptimizationManager._apply_semantic_filtering')
    @patch('optimization.config.ReplayOptimizationManager._apply_summarization')
    @patch('optimization.config.ReplayOptimizationManager._apply_prioritization')
    def test_optimize_replay_buffer_all_enabled(self, mock_prioritization, mock_summarization, 
                                               mock_filtering, sample_input_examples):
        """Test optimization pipeline with all components enabled."""
        config = OptimizationConfig()
        manager = ReplayOptimizationManager(config)
        
        # Mock component results
        mock_filtering.return_value = (sample_input_examples[:4], {"filtered": 1})
        mock_summarization.return_value = (sample_input_examples[:4], {"summarized": 1})
        mock_prioritization.return_value = (sample_input_examples[:3], {"prioritized": 1})
        
        result_samples, metrics = manager.optimize_replay_buffer(sample_input_examples, budget=3)
        
        assert len(result_samples) == 3
        assert "semantic_filtering" in metrics["component_metrics"]
        assert "summarization" in metrics["component_metrics"]
        assert "prioritization" in metrics["component_metrics"]
        
        mock_filtering.assert_called_once()
        mock_summarization.assert_called_once()
        mock_prioritization.assert_called_once()
    
    @patch('optimization.config.ReplayOptimizationManager._apply_semantic_filtering')
    def test_optimize_replay_buffer_filtering_only(self, mock_filtering, sample_input_examples):
        """Test optimization pipeline with only filtering enabled."""
        config = OptimizationConfig()
        config.summarization.enabled = False
        config.prioritization.enabled = False
        manager = ReplayOptimizationManager(config)
        
        mock_filtering.return_value = (sample_input_examples[:3], {"filtered": 1})
        
        result_samples, metrics = manager.optimize_replay_buffer(sample_input_examples, budget=5)
        
        assert len(result_samples) == 3
        assert "semantic_filtering" in metrics["component_metrics"]
        assert "summarization" not in metrics["component_metrics"]
        assert "prioritization" not in metrics["component_metrics"]
    
    def test_optimize_replay_buffer_with_fallback(self, sample_input_examples):
        """Test optimization pipeline with fallback on error."""
        config = OptimizationConfig()
        config.enable_fallback = True
        manager = ReplayOptimizationManager(config)
        
        # Mock a component to raise an error
        with patch.object(manager, '_apply_semantic_filtering', side_effect=Exception("Test error")):
            result_samples, metrics = manager.optimize_replay_buffer(sample_input_examples, budget=3)
            
            assert len(result_samples) == 3  # Budget applied to original samples
            assert metrics["fallback_used"] is True
            assert "error" in metrics
    
    def test_optimize_replay_buffer_without_fallback(self, sample_input_examples):
        """Test optimization pipeline without fallback on error."""
        config = OptimizationConfig()
        config.enable_fallback = False
        manager = ReplayOptimizationManager(config)
        
        # Mock a component to raise an error
        with patch.object(manager, '_apply_semantic_filtering', side_effect=Exception("Test error")):
            with pytest.raises(OptimizationError) as exc_info:
                manager.optimize_replay_buffer(sample_input_examples, budget=3)
            
            assert "Replay optimization failed" in str(exc_info.value)