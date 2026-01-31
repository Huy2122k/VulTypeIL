"""
Unit tests for base optimization components and interfaces.

This module tests the core interfaces, data structures, and utility functions
used across all optimization components.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from optimization.base import (
    OptimizedInputExample, OptimizationMetrics, OptimizationComponent,
    FilterComponent, TransformComponent, PrioritizationComponent,
    OptimizationError, ConfigurationError, ProcessingError, ValidationError
)
from openprompt.data_utils import InputExample


class TestOptimizedInputExample:
    """Test OptimizedInputExample data structure."""
    
    def test_creation_with_defaults(self):
        """Test creating OptimizedInputExample with default values."""
        example = OptimizedInputExample(
            guid="test_1",
            text_a="test code",
            text_b="test description",
            tgt_text="CWE-119"
        )
        
        assert example.guid == "test_1"
        assert example.text_a == "test code"
        assert example.text_b == "test description"
        assert example.tgt_text == "CWE-119"
        assert example.original_size == 0
        assert example.compression_ratio == 1.0
        assert example.semantic_cluster_id == -1
        assert example.priority_score == 0.0
        assert example.optimization_metadata == {}
    
    def test_creation_with_optimization_data(self):
        """Test creating OptimizedInputExample with optimization metadata."""
        example = OptimizedInputExample(
            guid="test_1",
            text_a="test code",
            text_b="test description", 
            tgt_text="CWE-119",
            original_size=1000,
            compression_ratio=0.5,
            semantic_cluster_id=3,
            priority_score=0.8,
            optimization_metadata={"method": "test"}
        )
        
        assert example.original_size == 1000
        assert example.compression_ratio == 0.5
        assert example.semantic_cluster_id == 3
        assert example.priority_score == 0.8
        assert example.optimization_metadata == {"method": "test"}


class TestOptimizationMetrics:
    """Test OptimizationMetrics data structure."""
    
    def test_creation_with_defaults(self):
        """Test creating OptimizationMetrics with default values."""
        metrics = OptimizationMetrics()
        
        assert metrics.original_sample_count == 0
        assert metrics.filtered_sample_count == 0
        assert metrics.memory_reduction_ratio == 0.0
        assert metrics.processing_time_ms == 0
        assert metrics.accuracy_impact == 0.0
        assert metrics.cluster_distribution == {}
        assert metrics.compression_stats == {}
    
    def test_to_dict_conversion(self):
        """Test converting metrics to dictionary."""
        metrics = OptimizationMetrics(
            original_sample_count=100,
            filtered_sample_count=80,
            memory_reduction_ratio=0.2,
            processing_time_ms=1500,
            accuracy_impact=-0.01,
            cluster_distribution={0: 30, 1: 25, 2: 25},
            compression_stats={"avg_compression": 0.6}
        )
        
        result = metrics.to_dict()
        
        assert result["original_sample_count"] == 100
        assert result["filtered_sample_count"] == 80
        assert result["memory_reduction_ratio"] == 0.2
        assert result["processing_time_ms"] == 1500
        assert result["accuracy_impact"] == -0.01
        assert result["cluster_distribution"] == {0: 30, 1: 25, 2: 25}
        assert result["compression_stats"] == {"avg_compression": 0.6}


class MockOptimizationComponent(OptimizationComponent):
    """Mock implementation for testing abstract base class."""
    
    def initialize(self):
        self._is_initialized = True
    
    def process(self, samples: List[InputExample], **kwargs):
        if not self._is_initialized:
            self.initialize()
        
        # Mock processing - just return samples unchanged
        self.metrics.original_sample_count = len(samples)
        self.metrics.filtered_sample_count = len(samples)
        return samples, self.metrics
    
    def validate_config(self) -> List[str]:
        errors = []
        if "required_param" not in self.config:
            errors.append("Missing required_param")
        return errors


class TestOptimizationComponent:
    """Test OptimizationComponent abstract base class."""
    
    def test_initialization(self):
        """Test component initialization."""
        config = {"required_param": "value"}
        component = MockOptimizationComponent(config)
        
        assert component.config == config
        assert isinstance(component.metrics, OptimizationMetrics)
        assert not component._is_initialized
    
    def test_get_metrics(self):
        """Test getting metrics from component."""
        component = MockOptimizationComponent({})
        metrics = component.get_metrics()
        
        assert isinstance(metrics, OptimizationMetrics)
        assert metrics.original_sample_count == 0
    
    def test_reset_metrics(self):
        """Test resetting component metrics."""
        component = MockOptimizationComponent({})
        component.metrics.original_sample_count = 100
        
        component.reset_metrics()
        
        assert component.metrics.original_sample_count == 0
    
    def test_process_with_initialization(self, sample_input_examples):
        """Test processing samples with automatic initialization."""
        component = MockOptimizationComponent({"required_param": "value"})
        
        result_samples, metrics = component.process(sample_input_examples)
        
        assert component._is_initialized
        assert len(result_samples) == len(sample_input_examples)
        assert metrics.original_sample_count == len(sample_input_examples)
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        component = MockOptimizationComponent({"required_param": "value"})
        errors = component.validate_config()
        
        assert errors == []
    
    def test_validate_config_failure(self):
        """Test failed configuration validation."""
        component = MockOptimizationComponent({})
        errors = component.validate_config()
        
        assert len(errors) == 1
        assert "Missing required_param" in errors[0]


class MockFilterComponent(FilterComponent):
    """Mock filter component for testing."""
    
    def initialize(self):
        self._is_initialized = True
    
    def process(self, samples: List[InputExample], **kwargs):
        indices, metadata = self.filter_samples(samples, **kwargs)
        filtered_samples = [samples[i] for i in indices]
        
        self.metrics.original_sample_count = len(samples)
        self.metrics.filtered_sample_count = len(filtered_samples)
        
        return filtered_samples, self.metrics
    
    def filter_samples(self, samples: List[InputExample], features=None):
        # Mock filtering - keep every other sample
        indices = list(range(0, len(samples), 2))
        metadata = {"filter_method": "mock"}
        return indices, metadata
    
    def validate_config(self):
        return []


class TestFilterComponent:
    """Test FilterComponent abstract base class."""
    
    def test_filter_samples_implementation(self, sample_input_examples):
        """Test filter samples method."""
        component = MockFilterComponent({})
        
        indices, metadata = component.filter_samples(sample_input_examples)
        
        assert len(indices) == 3  # Every other sample from 5 samples
        assert indices == [0, 2, 4]
        assert metadata["filter_method"] == "mock"
    
    def test_process_uses_filter_samples(self, sample_input_examples):
        """Test that process method uses filter_samples."""
        component = MockFilterComponent({})
        
        result_samples, metrics = component.process(sample_input_examples)
        
        assert len(result_samples) == 3  # Filtered result
        assert metrics.original_sample_count == 5
        assert metrics.filtered_sample_count == 3


class MockTransformComponent(TransformComponent):
    """Mock transform component for testing."""
    
    def initialize(self):
        self._is_initialized = True
    
    def process(self, samples: List[InputExample], **kwargs):
        transformed_samples = self.transform_samples(samples)
        
        self.metrics.original_sample_count = len(samples)
        self.metrics.filtered_sample_count = len(transformed_samples)
        
        return transformed_samples, self.metrics
    
    def transform_samples(self, samples: List[InputExample]):
        # Mock transformation - add prefix to text_a
        transformed = []
        for sample in samples:
            transformed_sample = InputExample(
                guid=sample.guid,
                text_a=f"TRANSFORMED: {sample.text_a}",
                text_b=sample.text_b,
                tgt_text=sample.tgt_text
            )
            transformed.append(transformed_sample)
        return transformed
    
    def validate_config(self):
        return []


class TestTransformComponent:
    """Test TransformComponent abstract base class."""
    
    def test_transform_samples_implementation(self, sample_input_examples):
        """Test transform samples method."""
        component = MockTransformComponent({})
        
        transformed = component.transform_samples(sample_input_examples)
        
        assert len(transformed) == len(sample_input_examples)
        assert all(sample.text_a.startswith("TRANSFORMED:") for sample in transformed)
    
    def test_process_uses_transform_samples(self, sample_input_examples):
        """Test that process method uses transform_samples."""
        component = MockTransformComponent({})
        
        result_samples, metrics = component.process(sample_input_examples)
        
        assert len(result_samples) == len(sample_input_examples)
        assert all(sample.text_a.startswith("TRANSFORMED:") for sample in result_samples)


class MockPrioritizationComponent(PrioritizationComponent):
    """Mock prioritization component for testing."""
    
    def initialize(self):
        self._is_initialized = True
    
    def process(self, samples: List[InputExample], budget: int, **kwargs):
        indices, metadata = self.prioritize_samples(samples, budget)
        prioritized_samples = [samples[i] for i in indices]
        
        self.metrics.original_sample_count = len(samples)
        self.metrics.filtered_sample_count = len(prioritized_samples)
        
        return prioritized_samples, self.metrics
    
    def prioritize_samples(self, samples: List[InputExample], budget: int):
        # Mock prioritization - select first N samples
        indices = list(range(min(budget, len(samples))))
        metadata = {"prioritization_method": "mock"}
        return indices, metadata
    
    def validate_config(self):
        return []


class TestPrioritizationComponent:
    """Test PrioritizationComponent abstract base class."""
    
    def test_prioritize_samples_implementation(self, sample_input_examples):
        """Test prioritize samples method."""
        component = MockPrioritizationComponent({})
        budget = 3
        
        indices, metadata = component.prioritize_samples(sample_input_examples, budget)
        
        assert len(indices) == budget
        assert indices == [0, 1, 2]
        assert metadata["prioritization_method"] == "mock"
    
    def test_process_uses_prioritize_samples(self, sample_input_examples):
        """Test that process method uses prioritize_samples."""
        component = MockPrioritizationComponent({})
        budget = 3
        
        result_samples, metrics = component.process(sample_input_examples, budget=budget)
        
        assert len(result_samples) == budget
        assert metrics.original_sample_count == 5
        assert metrics.filtered_sample_count == 3


class TestOptimizationExceptions:
    """Test optimization-specific exceptions."""
    
    def test_optimization_error(self):
        """Test OptimizationError exception."""
        with pytest.raises(OptimizationError) as exc_info:
            raise OptimizationError("Test error")
        
        assert str(exc_info.value) == "Test error"
    
    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Config error")
        
        assert str(exc_info.value) == "Config error"
        assert isinstance(exc_info.value, OptimizationError)
    
    def test_processing_error(self):
        """Test ProcessingError exception."""
        with pytest.raises(ProcessingError) as exc_info:
            raise ProcessingError("Processing error")
        
        assert str(exc_info.value) == "Processing error"
        assert isinstance(exc_info.value, OptimizationError)
    
    def test_validation_error(self):
        """Test ValidationError exception."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Validation error")
        
        assert str(exc_info.value) == "Validation error"
        assert isinstance(exc_info.value, OptimizationError)