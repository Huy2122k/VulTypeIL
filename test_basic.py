#!/usr/bin/env python3
"""
Basic test script to check if libraries can be imported without segfault
"""

import sys
import traceback

def test_imports():
    """Test basic imports"""
    try:
        print("Testing basic imports...")
        
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        
        import pandas as pd
        print("✓ Pandas")
        
        import numpy as np
        print("✓ NumPy")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ CUDA not available")
            
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test loading a small model"""
    try:
        print("\nTesting model loading...")
        
        from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer
        
        print("Loading tokenizer...")
        tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
        print("✓ Tokenizer loaded")
        
        print("Loading model config...")
        config = T5Config.from_pretrained("Salesforce/codet5-base")
        print("✓ Config loaded")
        
        print("Loading model (this may take a while)...")
        model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
        print("✓ Model loaded")
        
        # Test a simple forward pass
        print("Testing forward pass...")
        inputs = tokenizer("def hello():", return_tensors="pt", max_length=32, padding=True, truncation=True)
        
        if torch.cuda.is_available():
            model = model.cuda()
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model(**inputs)
            
        print("✓ Forward pass successful")
        
        # Cleanup
        del model, tokenizer, inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True
        
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=== Basic System Test ===")
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False
        
    # Test model loading
    if not test_model_loading():
        print("❌ Model loading test failed")
        return False
        
    print("\n✅ All tests passed! System appears to be working correctly.")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠ Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)