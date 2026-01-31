"""
Demo: Tích hợp Cải tiến Scalable Replay
======================================

Demo này cho thấy cách tích hợp các cải tiến scalable replay
vào vul_main2.py hiện có với thay đổi code tối thiểu.

Chạy demo này để xem các cải tiến hoạt động.

Tác giả: AI Assistant
"""

import sys
import os
import torch
import numpy as np
from collections import Counter

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from replay_integration import EnhancedReplaySelector
from replay_config import create_config
from scalable_replay_improvements import ScalableReplayManager


def create_mock_examples(num_examples=100, num_classes=5):
    """Tạo mock examples để demo"""
    from openprompt.data_utils import InputExample
    
    examples = []
    for i in range(num_examples):
        # Mock vulnerability code
        code_snippets = [
            "char buffer[10]; strcpy(buffer, user_input);",
            "int *ptr = malloc(sizeof(int)); free(ptr); *ptr = 5;",
            "for(int i = 0; i <= array_size; i++) { array[i] = value; }",
            "if(user_input) { process_data(user_input); }",
            "while(condition) { /* potential infinite loop */ }"
        ]
        
        descriptions = [
            "Lỗ hổng buffer overflow trong string copy",
            "Lỗ hổng use after free",
            "Tràn biên mảng",
            "Thiếu kiểm tra input",
            "Vòng lặp vô hạn tiềm ẩn"
        ]
        
        code_idx = i % len(code_snippets)
        class_label = i % num_classes
        
        example = InputExample(
            guid=i,
            text_a=code_snippets[code_idx] + f" // Ví dụ {i}",
            text_b=descriptions[code_idx],
            tgt_text=class_label
        )
        examples.append(example)
    
    return examples


def create_mock_dataloader(examples):
    """Tạo mock dataloader để demo"""
    class MockDataLoader:
        def __init__(self, examples):
            self.examples = examples
            self.batch_size = 16
        
        def __iter__(self):
            for i in range(0, len(self.examples), self.batch_size):
                batch_examples = self.examples[i:i+self.batch_size]
                batch_data = {
                    'tgt_text': [ex.tgt_text for ex in batch_examples]
                }
                yield batch_data
        
        def __len__(self):
            return (len(self.examples) + self.batch_size - 1) // self.batch_size
    
    return MockDataLoader(examples)


def create_mock_model():
    """Tạo mock model để demo"""
    class MockModel:
        def __init__(self):
            self.eval_mode = False
        
        def eval(self):
            self.eval_mode = True
        
        def cuda(self):
            return self
        
        def __call__(self, inputs):
            # Trả về mock logits
            batch_size = len(inputs['tgt_text'])
            return torch.randn(batch_size, 5)  # 5 classes
    
    return MockModel()


def demo_basic_integration():
    """Demonstrate basic integration with existing code"""
    print("🚀 DEMO: Basic Integration with Existing Code")
    print("=" * 60)
    
    # Create mock data
    examples = create_mock_examples(num_examples=50, num_classes=5)
    dataloader = create_mock_dataloader(examples)
    model = create_mock_model()
    current_task_examples = create_mock_examples(num_examples=20, num_classes=3)
    
    # Create enhanced replay selector
    enhanced_selector = EnhancedReplaySelector(
        similarity_threshold=0.8,
        max_code_lines=8,
        n_clusters=5,
        memory_dir="demo_memory"
    )
    
    # Perform enhanced replay selection
    selected_indices, selection_info = enhanced_selector.select_enhanced_replay_samples(
        prompt_model=model,
        dataloader=dataloader,
        examples=examples,
        num_samples=15,
        task_id=2,
        min_samples_per_class=2,
        current_task_examples=current_task_examples
    )
    
    print(f"✅ Selected {len(selected_indices)} samples from {len(examples)} total")
    print(f"📊 Selection info: {selection_info}")
    
    return selected_indices, selection_info


def demo_configuration_options():
    """Demonstrate different configuration options"""
    print("\n🔧 DEMO: Configuration Options")
    print("=" * 60)
    
    configs = {
        'Memory Efficient': create_config('memory_efficient'),
        'Quality Focused': create_config('quality_focused'),
        'Balanced': create_config('balanced'),
        'Fast': create_config('fast')
    }
    
    for name, config in configs.items():
        print(f"\n{name} Configuration:")
        print(f"  Similarity threshold: {config.semantic_filter.similarity_threshold}")
        print(f"  Max code lines: {config.code_summarizer.max_code_lines}")
        print(f"  Number of clusters: {config.clustering.n_clusters}")
        print(f"  Gradient importance: {config.gradient_importance.enabled}")


def demo_scalable_replay_manager():
    """Demonstrate the ScalableReplayManager directly"""
    print("\n⚙️  DEMO: ScalableReplayManager Direct Usage")
    print("=" * 60)
    
    # Create manager with custom config
    config = create_config('quality_focused')
    manager = ScalableReplayManager(
        similarity_threshold=config.semantic_filter.similarity_threshold,
        max_code_lines=config.code_summarizer.max_code_lines,
        n_clusters=config.clustering.n_clusters,
        memory_dir="demo_scalable_memory"
    )
    
    # Create mock data
    examples = create_mock_examples(num_examples=30, num_classes=4)
    features = np.random.randn(30, 10)  # Mock features
    labels = [ex.tgt_text for ex in examples]
    current_task_vulnerabilities = {0, 1, 2}  # Mock current task vulnerabilities
    
    # Process replay buffer
    selected_examples, selection_info = manager.process_replay_buffer(
        examples=examples,
        features=features,
        labels=labels,
        task_id=3,
        replay_budget=10,
        current_task_vulnerabilities=current_task_vulnerabilities,
        min_samples_per_class=1
    )
    
    print(f"✅ Processed replay buffer: {len(examples)} → {len(selected_examples)} samples")
    
    # Get historical context
    historical_context = manager.get_historical_context(4)
    if historical_context:
        print(f"\n📚 Historical Context:\n{historical_context}")
    else:
        print("\n📚 No historical context available yet")


def demo_integration_with_existing_code():
    """Show exact integration steps for vul_main2.py"""
    print("\n🔗 DEMO: Integration Steps for vul_main2.py")
    print("=" * 60)
    
    integration_code = '''
# STEP 1: Add imports at the top of vul_main2.py
from replay_integration import upgrade_existing_replay_function, log_replay_improvements
from replay_config import create_config

# STEP 2: Create enhanced selector (add after model initialization)
enhanced_selector = upgrade_existing_replay_function()

# STEP 3: Replace existing replay selection in main loop (around line 700)
# OLD CODE:
# indices_to_replay, _ = select_uncertain_samples_with_stratified_class(
#     prompt_model, train_dataloader_prev, prev_examples,
#     num_samples=replay_budget, min_samples_per_class=args.min_samples_per_class
# )

# NEW CODE:
current_examples = read_prompt_examples(data_paths[i - 1])
indices_to_replay, selection_info = enhanced_selector.select_enhanced_replay_samples(
    prompt_model=prompt_model,
    dataloader=train_dataloader_prev,
    examples=prev_examples,
    num_samples=replay_budget,
    task_id=i,
    min_samples_per_class=args.min_samples_per_class,
    current_task_examples=current_examples
)

# STEP 4: Log improvements (optional)
log_replay_improvements(selection_info, i)

# STEP 5: Use historical context for enhanced prompting (optional)
historical_context = enhanced_selector.get_historical_context(i)
if historical_context:
    print(f"Historical Context for Task {i}:")
    print(historical_context)
'''
    
    print(integration_code)


def demo_performance_comparison():
    """Demonstrate performance comparison between old and new methods"""
    print("\n📈 DEMO: Performance Comparison")
    print("=" * 60)
    
    # Simulate old method (random selection)
    examples = create_mock_examples(num_examples=100, num_classes=5)
    
    # Old method: Random selection
    import random
    random.seed(42)
    old_indices = random.sample(range(len(examples)), 20)
    old_class_dist = Counter([examples[i].tgt_text for i in old_indices])
    
    # New method: Enhanced selection
    dataloader = create_mock_dataloader(examples)
    model = create_mock_model()
    enhanced_selector = EnhancedReplaySelector()
    
    new_indices, _ = enhanced_selector.select_enhanced_replay_samples(
        prompt_model=model,
        dataloader=dataloader,
        examples=examples,
        num_samples=20,
        task_id=1,
        min_samples_per_class=2
    )
    new_class_dist = Counter([examples[i].tgt_text for i in new_indices])
    
    print("📊 Class Distribution Comparison:")
    print(f"Old method (random): {dict(old_class_dist)}")
    print(f"New method (enhanced): {dict(new_class_dist)}")
    
    # Calculate balance score (lower is better)
    def balance_score(dist):
        values = list(dist.values())
        if not values:
            return float('inf')
        return max(values) - min(values)
    
    old_balance = balance_score(old_class_dist)
    new_balance = balance_score(new_class_dist)
    
    print(f"\n📏 Balance Score (lower is better):")
    print(f"Old method: {old_balance}")
    print(f"New method: {new_balance}")
    
    if new_balance < old_balance:
        print("✅ Enhanced method provides better class balance!")
    else:
        print("⚠️  Results may vary with different random seeds")


def cleanup_demo_files():
    """Clean up demo files"""
    import shutil
    
    demo_dirs = ["demo_memory", "demo_scalable_memory"]
    for dir_name in demo_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"🧹 Cleaned up {dir_name}")
    
    demo_files = ["replay_improvements.log", "replay_config_example.json"]
    for file_name in demo_files:
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f"🧹 Cleaned up {file_name}")


def main():
    """Run all demos"""
    print("🎯 SCALABLE REPLAY IMPROVEMENTS DEMO")
    print("=" * 70)
    print("This demo shows how to integrate scalable replay improvements")
    print("into your existing continual learning pipeline.")
    print("=" * 70)
    
    try:
        # Run demos
        demo_basic_integration()
        demo_configuration_options()
        demo_scalable_replay_manager()
        demo_integration_with_existing_code()
        demo_performance_comparison()
        
        print("\n🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("1. Review the integration code above")
        print("2. Choose appropriate configuration for your use case")
        print("3. Integrate into your vul_main2.py following the examples")
        print("4. Monitor replay_improvements.log for performance metrics")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up demo files
        cleanup_demo_files()


if __name__ == "__main__":
    main()