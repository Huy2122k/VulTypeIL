"""
Pytest configuration and fixtures for optimization tests.

This module provides common fixtures and test utilities for property-based
testing and unit testing of optimization components.
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from hypothesis import strategies as st
from openprompt.data_utils import InputExample

from optimization.base import OptimizedInputExample
from optimization.config import OptimizationConfig


# Hypothesis strategies for generating test data
@st.composite
def input_example_strategy(draw):
    """Generate InputExample instances for testing."""
    guid = draw(st.text(min_size=1, max_size=20))
    text_a = draw(st.text(min_size=10, max_size=1000))  # Code content
    text_b = draw(st.one_of(st.none(), st.text(min_size=5, max_size=200)))  # Description
    tgt_text = draw(st.sampled_from([
        "CWE-119", "CWE-125", "CWE-787", "CWE-476", "CWE-20", 
        "CWE-190", "CWE-200", "CWE-120", "CWE-399", "CWE-401"
    ]))
    
    return InputExample(guid=guid, text_a=text_a, text_b=text_b, tgt_text=tgt_text)


@st.composite
def optimized_input_example_strategy(draw):
    """Generate OptimizedInputExample instances for testing."""
    base_example = draw(input_example_strategy())
    original_size = draw(st.integers(min_value=100, max_value=10000))
    compression_ratio = draw(st.floats(min_value=0.1, max_value=1.0))
    semantic_cluster_id = draw(st.integers(min_value=-1, max_value=50))
    priority_score = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return OptimizedInputExample(
        guid=base_example.guid,
        text_a=base_example.text_a,
        text_b=base_example.text_b,
        tgt_text=base_example.tgt_text,
        original_size=original_size,
        compression_ratio=compression_ratio,
        semantic_cluster_id=semantic_cluster_id,
        priority_score=priority_score,
        optimization_metadata={}
    )


@st.composite
def sample_list_strategy(draw, min_size=1, max_size=100):
    """Generate lists of InputExample instances."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(st.lists(input_example_strategy(), min_size=size, max_size=size))


@st.composite
def feature_matrix_strategy(draw, n_samples=None, n_features=None):
    """Generate feature matrices for testing."""
    if n_samples is None:
        n_samples = draw(st.integers(min_value=2, max_value=100))
    if n_features is None:
        n_features = draw(st.integers(min_value=5, max_value=512))
    
    # Generate normalized features (common for embeddings)
    features = draw(st.lists(
        st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=n_features, max_size=n_features),
        min_size=n_samples, max_size=n_samples
    ))
    
    return np.array(features)


@st.composite
def similarity_matrix_strategy(draw, n_samples=None):
    """Generate symmetric similarity matrices."""
    if n_samples is None:
        n_samples = draw(st.integers(min_value=2, max_value=50))
    
    # Generate upper triangular matrix
    matrix = np.eye(n_samples)  # Diagonal is 1.0 (self-similarity)
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            similarity = draw(st.floats(min_value=0.0, max_value=1.0))
            matrix[i, j] = similarity
            matrix[j, i] = similarity  # Make symmetric
    
    return matrix


# Pytest fixtures
@pytest.fixture
def sample_config():
    """Provide a basic optimization configuration for testing."""
    return OptimizationConfig()


@pytest.fixture
def sample_input_examples():
    """Provide a list of sample InputExample instances."""
    return [
        InputExample(
            guid="test_1",
            text_a="void vulnerable_function(char *input) { char buffer[100]; strcpy(buffer, input); }",
            text_b="Buffer overflow vulnerability in strcpy",
            tgt_text="CWE-119"
        ),
        InputExample(
            guid="test_2", 
            text_a="int process_data(int *data, int size) { return data[size]; }",
            text_b="Out-of-bounds read vulnerability",
            tgt_text="CWE-125"
        ),
        InputExample(
            guid="test_3",
            text_a="void write_data(char *dest, char *src) { strcpy(dest, src); }",
            text_b="Buffer overflow in strcpy operation",
            tgt_text="CWE-119"
        ),
        InputExample(
            guid="test_4",
            text_a="void *allocate_memory(size_t size) { void *ptr = malloc(size); return ptr; }",
            text_b="Memory leak - missing free",
            tgt_text="CWE-401"
        ),
        InputExample(
            guid="test_5",
            text_a="int validate_input(char *input) { if (input == NULL) return -1; return strlen(input); }",
            text_b="Null pointer dereference in strlen",
            tgt_text="CWE-476"
        )
    ]


@pytest.fixture
def sample_features():
    """Provide sample feature vectors for testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.rand(5, 128)  # 5 samples, 128 features


@pytest.fixture
def vulnerability_classes():
    """Provide list of vulnerability classes for testing."""
    return [
        "CWE-119", "CWE-125", "CWE-787", "CWE-476", "CWE-20",
        "CWE-190", "CWE-200", "CWE-120", "CWE-399", "CWE-401",
        "CWE-78", "CWE-79", "CWE-89", "CWE-94", "CWE-400",
        "CWE-415", "CWE-122", "CWE-770", "CWE-22"
    ]


@pytest.fixture
def mock_historical_data():
    """Provide mock historical data for temporal analysis testing."""
    return {
        'task_1': {
            'samples': 100,
            'classes': {'CWE-119': 30, 'CWE-125': 25, 'CWE-787': 20, 'CWE-476': 25},
            'timestamp': 1000
        },
        'task_2': {
            'samples': 120,
            'classes': {'CWE-119': 40, 'CWE-125': 30, 'CWE-190': 25, 'CWE-200': 25},
            'timestamp': 2000
        },
        'task_3': {
            'samples': 90,
            'classes': {'CWE-787': 35, 'CWE-476': 30, 'CWE-190': 25},
            'timestamp': 3000
        }
    }


# Test utilities
def create_similar_samples(base_sample: InputExample, similarity_level: float = 0.9, count: int = 3) -> List[InputExample]:
    """Create samples with controlled similarity to a base sample."""
    similar_samples = [base_sample]
    
    for i in range(count - 1):
        # Create variations by modifying the code slightly
        modified_code = base_sample.text_a
        if similarity_level > 0.8:
            # High similarity - minor changes
            modified_code = modified_code.replace("buffer", f"buffer{i}")
        elif similarity_level > 0.5:
            # Medium similarity - moderate changes
            modified_code = modified_code.replace("char", "unsigned char")
        else:
            # Low similarity - major changes
            modified_code = "completely different code here"
        
        similar_sample = InputExample(
            guid=f"{base_sample.guid}_similar_{i}",
            text_a=modified_code,
            text_b=base_sample.text_b,
            tgt_text=base_sample.tgt_text
        )
        similar_samples.append(similar_sample)
    
    return similar_samples


def assert_samples_preserved_per_class(original_samples: List[InputExample], 
                                     filtered_samples: List[InputExample],
                                     min_samples_per_class: int):
    """Assert that minimum samples per class constraint is maintained."""
    from collections import Counter
    
    original_counts = Counter(sample.tgt_text for sample in original_samples)
    filtered_counts = Counter(sample.tgt_text for sample in filtered_samples)
    
    for class_label, original_count in original_counts.items():
        expected_min = min(original_count, min_samples_per_class)
        actual_count = filtered_counts.get(class_label, 0)
        assert actual_count >= expected_min, \
            f"Class {class_label}: expected >= {expected_min}, got {actual_count}"


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Simple semantic similarity calculation for testing."""
    # Simple token-based similarity for testing
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union)


# Property-based testing configuration
def configure_hypothesis():
    """Configure Hypothesis settings for property-based tests."""
    from hypothesis import settings, Verbosity
    
    # Configure for thorough testing
    settings.register_profile("thorough", 
                            max_examples=100,
                            verbosity=Verbosity.verbose,
                            deadline=None)
    
    # Configure for quick testing during development
    settings.register_profile("quick",
                            max_examples=20,
                            verbosity=Verbosity.normal,
                            deadline=1000)
    
    # Use thorough by default
    settings.load_profile("thorough")


# Call configuration on import
configure_hypothesis()