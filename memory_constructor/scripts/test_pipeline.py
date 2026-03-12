#!/usr/bin/env python3
"""
End-to-end pipeline test.

This script validates that all components work together:
1. Data preprocessing ✓
2. Model loading
3. Dataset creation
4. Inference test
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("="*80)
print("Memory Constructor Pipeline Test")
print("="*80)

# Test 1: Check processed data
print("\n[1] Checking processed data...")
train_file = project_root / "data/webshop/processed/train.jsonl"
if not train_file.exists():
    print(f"❌ Training data not found: {train_file}")
    sys.exit(1)

with open(train_file) as f:
    samples = [json.loads(line) for line in f]
print(f"✓ Found {len(samples)} training samples")

# Test 2: Load model and tokenizer
print("\n[2] Loading model and tokenizer...")
model_name = "Qwen/Qwen2.5-0.5B"  # Use smallest model for testing
print(f"   Model: {model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer loaded")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✓ Model loaded")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Test 3: Format sample for inference
print("\n[3] Testing sample formatting...")
sample = samples[0]

# Create prompt
prompt = f"""Task: Decide whether to write a memory and what to write.

Observation: {sample['observation']}
Local History: {sample['local_history']}
Current Memories: {sample['memory_store']}
Budget Remaining: {sample['budget_remaining']}

Output a JSON with:
- write: true/false
- keys: list of search keys
- value: memory content

Output:"""

print(f"✓ Created prompt ({len(prompt)} chars)")

# Test 4: Run inference
print("\n[4] Testing inference...")
try:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"✓ Tokenized input ({inputs['input_ids'].shape[1]} tokens)")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
        )

    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"✓ Generated output ({len(generated_text)} chars)")
    print(f"\nGenerated:\n{generated_text[:200]}...")

except Exception as e:
    print(f"❌ Inference failed: {e}")
    sys.exit(1)

# Test 5: Check all modules import correctly
print("\n[5] Testing module imports...")
try:
    from memory_constructor.data import (
        WebShopParser,
        HindsightLabeler,
        HeuristicLabeler,
    )
    print("✓ Data modules")

    from memory_constructor.environment import (
        MemoryConstructorAgentInstance,
        MemoryStore,
    )
    print("✓ Environment modules")

    from memory_constructor.models import GPT5Agent, HybridRetriever
    print("✓ Model modules")

    # Try to import training modules
    try:
        from memory_constructor.training import CandidateSampler, CandidateScorer
        print("✓ Training modules")
    except ImportError as e:
        print(f"⊘ Training modules (vLLM not available: {str(e)[:50]}...)")

    from memory_constructor.evaluation import Evaluator
    print("✓ Evaluation modules")

except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ All pipeline tests passed!")
print("="*80)
print("\nNext steps:")
print("1. Prepare larger WebShop dataset")
print("2. Run full SFT training with OpenRLHF")
print("3. Generate candidates and train Best-of-N")
print("4. Run evaluation on test set")
