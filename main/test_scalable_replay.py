"""
Test Suite for Scalable Replay Improvements
==========================================

Simple tests to verify the implementation works correctly.

Author: AI Assistant
"""

import unittest
import sys
import os
import tempfile
import shutil
import numpy as np
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scalable_replay_improvements import (
    SemanticRedundancyFilter,
    VulnerabilityCodeSummarizer,
    ClusteringBasedReplayPriority,
    ScalableReplayManager
)
from replay_integration import EnhancedReplaySelector
from replay_config import create_config, ScalableReplayConfig
from openprompt.data_utils import InputExample


class TestSemanticRedundancyFilter(unittest.TestCase):
    """Test semantic redundancy filtering"""
    
    def setUp(self):
        self.filter = SemanticRedundancyFilter(similarity_threshold=0.8)
        
    def test_filter_redundant_samples(self):
        """Test basic redundancy filtering"""
        examples = [
            InputExample(guid=0, text_a="buffer overflow vulnerability", text_b="desc1", tgt_text=0),
            InputExample(guid=1, text_a="buffer overflow issue", text_b="desc2", tgt_text=0),  # Similar
            InputExample(guid=2, text_a="null pointer dereference", text_b="desc3", tgt_text=1),  # Different
        ]
        
        filtered_examples, kept_indices = self.filter.filter_redundant_samples(examples)
        
        # Should keep at least 2 examples (different topics)
        self.assertGreaterEqual(len(filtered_examples), 2)
        self.assertEqual(len(kept_indices), len(filtered_examples))
        
    def test_empty_input(self):
        """Test with empty input"""
        examples = []
        filtered_examples, kept_indices = self.filter.filter_redundant_samples(examples)
        
        self.assertEqual(len(filtered_examples), 0)
        self.assertEqual(len(kept_indices), 0)


class TestVulnerabilityCodeSummarizer(unittest.TestCase):
    """Test vulnerability code summarization"""
    
    def setUp(self):
        self.summarizer = VulnerabilityCodeSummarizer()
        
    def test_extract_vulnerability_lines(self):
        """Test vulnerability line extraction"""
        code = """
        int main() {
            char buffer[10];
            strcpy(buffer, user_input);  // Vulnerable line
            printf("Hello");
            return 0;
        }
        """
        
        summarized = self.summarizer.extract_vulnerability_lines(code, max_lines=3)
        
        # Should contain the vulnerable line
        self.assertIn("strcpy", summarized)
        # Should be shorter than original
        self.assertLess(len(summarized.split('\n')), len(code.split('\n')))
        
    def test_summarize_examples(self):
        """Test example summarization"""
        examples = [
            InputExample(
                guid=0, 
                text_a="char buf[10]; strcpy(buf, input); printf('done');", 
                text_b="Buffer overflow", 
                tgt_text=0
            )
        ]
        
        summarized = self.summarizer.summarize_examples(examples, max_code_lines=2)
        
        self.assertEqual(len(summarized), 1)
        # Should still contain vulnerability keyword
        self.assertIn("strcpy", summarized[0].text_a)


class TestClusteringBasedReplayPriority(unittest.TestCase):
    """Test clustering-based replay priority"""
    
    def setUp(self):
        self.clustering = ClusteringBasedReplayPriority(n_clusters=3)
        
    def test_update_clusters(self):
        """Test cluster updating"""
        features = np.random.randn(10, 5)
        labels = [0, 1, 0, 1, 2, 0, 1, 2, 0, 1]
        
        self.clustering.update_clusters(features, labels, task_id=1)
        
        # Should have stored task mapping
        self.assertIn(1, self.clustering.task_cluster_mapping)
        # Should have vulnerability frequencies
        self.assertGreater(len(self.clustering.cluster_vulnerability_freq), 0)
        
    def test_get_replay_priorities(self):
        """Test priority computation"""
        # Setup clusters first
        features = np.random.randn(6, 5)
        labels = [0, 1, 0, 1, 2, 2]
        self.clustering.update_clusters(features, labels, task_id=1)
        
        # Get priorities
        priorities = self.clustering.get_replay_priorities(
            features, labels, current_task_vulnerabilities={0, 1}
        )
        
        self.assertEqual(len(priorities), len(features))
        # All priorities should be positive
        self.assertTrue(all(p >= 0 for p in priorities))


class TestScalableReplayManager(unittest.TestCase):
    """Test the main scalable replay manager"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ScalableReplayManager(
            similarity_threshold=0.8,
            max_code_lines=5,
            n_clusters=3,
            memory_dir=self.temp_dir
        )
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_process_replay_buffer(self):
        """Test complete replay buffer processing"""
        examples = [
            InputExample(guid=i, text_a=f"code {i}", text_b=f"desc {i}", tgt_text=i%3)
            for i in range(10)
        ]
        features = np.random.randn(10, 5)
        labels = [i%3 for i in range(10)]
        
        selected_examples, selection_info = self.manager.process_replay_buffer(
            examples=examples,
            features=features,
            labels=labels,
            task_id=1,
            replay_budget=5,
            current_task_vulnerabilities={0, 1},
            min_samples_per_class=1
        )
        
        # Should select requested number of samples
        self.assertLessEqual(len(selected_examples), 5)
        # Should have selection info
        self.assertIn('original_count', selection_info)
        self.assertIn('final_selected', selection_info)


class TestEnhancedReplaySelector(unittest.TestCase):
    """Test the integration wrapper"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.selector = EnhancedReplaySelector(
            similarity_threshold=0.8,
            max_code_lines=5,
            n_clusters=3,
            memory_dir=self.temp_dir
        )
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_mock_integration(self):
        """Test with mock model and dataloader"""
        # Mock model
        class MockModel:
            def eval(self): pass
            def cuda(self): return self
            def __call__(self, inputs):
                return torch.randn(len(inputs['tgt_text']), 5)
        
        # Mock dataloader
        class MockDataLoader:
            def __init__(self, examples):
                self.examples = examples
            def __iter__(self):
                yield {'tgt_text': [ex.tgt_text for ex in self.examples]}
            def __len__(self):
                return 1
        
        examples = [
            InputExample(guid=i, text_a=f"code {i}", text_b=f"desc {i}", tgt_text=i%3)
            for i in range(8)
        ]
        
        model = MockModel()
        dataloader = MockDataLoader(examples)
        
        # This should not crash
        try:
            indices, info = self.selector.select_enhanced_replay_samples(
                prompt_model=model,
                dataloader=dataloader,
                examples=examples,
                num_samples=3,
                task_id=1,
                min_samples_per_class=1
            )
            # Basic checks
            self.assertIsInstance(indices, list)
            self.assertIsInstance(info, dict)
        except Exception as e:
            self.fail(f"Mock integration failed: {e}")


class TestReplayConfig(unittest.TestCase):
    """Test configuration management"""
    
    def test_create_config(self):
        """Test configuration creation"""
        config_types = ['balanced', 'memory_efficient', 'quality_focused', 'fast']
        
        for config_type in config_types:
            config = create_config(config_type)
            self.assertIsInstance(config, ScalableReplayConfig)
            
    def test_config_serialization(self):
        """Test config save/load"""
        config = create_config('balanced')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_to_file(f.name)
            
            # Load it back
            loaded_config = ScalableReplayConfig.load_from_file(f.name)
            
            # Should have same similarity threshold
            self.assertEqual(
                config.semantic_filter.similarity_threshold,
                loaded_config.semantic_filter.similarity_threshold
            )
            
        os.unlink(f.name)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create manager
            manager = ScalableReplayManager(memory_dir=temp_dir)
            
            # Create mock data
            examples = [
                InputExample(guid=i, text_a=f"vulnerability code {i}", text_b=f"description {i}", tgt_text=i%2)
                for i in range(6)
            ]
            features = np.random.randn(6, 4)
            labels = [i%2 for i in range(6)]
            
            # Process replay buffer
            selected_examples, info = manager.process_replay_buffer(
                examples=examples,
                features=features,
                labels=labels,
                task_id=1,
                replay_budget=3,
                current_task_vulnerabilities={0},
                min_samples_per_class=1
            )
            
            # Verify results
            self.assertLessEqual(len(selected_examples), 3)
            self.assertGreater(len(selected_examples), 0)
            
            # Test historical context
            context = manager.get_historical_context(2)
            self.assertIsInstance(context, str)
            
        finally:
            shutil.rmtree(temp_dir)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_classes = [
        TestSemanticRedundancyFilter,
        TestVulnerabilityCodeSummarizer,
        TestClusteringBasedReplayPriority,
        TestScalableReplayManager,
        TestEnhancedReplaySelector,
        TestReplayConfig,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running Scalable Replay Improvements Tests")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        print("The scalable replay improvements are ready to use.")
    else:
        print("\n❌ Some tests failed!")
        print("Please check the implementation before using.")
        sys.exit(1)