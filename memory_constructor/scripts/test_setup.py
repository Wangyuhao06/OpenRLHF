#!/usr/bin/env python3
"""
Quick test script to verify the memory constructor setup.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Testing memory constructor imports...")

try:
    # Test data imports
    from memory_constructor.data.schemas import MemoryItem, SFTSample
    print("✓ Data schemas imported")

    from memory_constructor.data.memory_store import MemoryStore
    print("✓ Memory store imported")

    from memory_constructor.data.webshop_parser import WebShopParser
    print("✓ WebShop parser imported")

    from memory_constructor.data.hindsight_labeling import HindsightLabeler
    print("✓ Hindsight labeler imported")

    from memory_constructor.data.sft_dataset import MemoryConstructorSFTDataset
    print("✓ SFT dataset imported")

    # Test model imports
    from memory_constructor.models.retriever import HybridRetriever
    print("✓ Retriever imported")

    from memory_constructor.models.agent import GPT5Agent, MockAgent
    print("✓ Agent imported")

    # Test environment imports
    from memory_constructor.environment.agent_instance import (
        MemoryConstructorAgentInstance,
    )
    print("✓ Agent instance imported")

    # Test training imports
    try:
        from memory_constructor.training.candidate_sampler import CandidateSampler
        print("✓ Candidate sampler imported")
    except ImportError as e:
        print(f"⚠ Candidate sampler skipped (missing vLLM): {e}")

    from memory_constructor.training.candidate_scorer import CandidateScorer
    print("✓ Candidate scorer imported")

    from memory_constructor.training.best_of_n_trainer import BestOfNTrainer
    print("✓ Best-of-N trainer imported")

    # Test evaluation imports
    from memory_constructor.evaluation import Evaluator, MetricsCalculator
    print("✓ Evaluation framework imported")

    print("\n✅ All imports successful!")

    # Test basic functionality
    print("\nTesting basic functionality...")

    # Test memory store
    store = MemoryStore()
    store.add({"keys": ["test"], "value": "test memory", "step": 0})
    assert len(store.memories) == 1
    print("✓ Memory store works")

    # Test retriever
    retriever = HybridRetriever()
    results = retriever.retrieve("test query", store, k=1)
    assert len(results) <= 1
    print("✓ Retriever works")

    # Test mock agent
    agent = MockAgent()
    action = agent.act("test observation", [])
    assert "action" in action
    print("✓ Mock agent works")

    print("\n✅ All basic tests passed!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
