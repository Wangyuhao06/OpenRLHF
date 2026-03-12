#!/usr/bin/env python3
"""
Quick import test - skips model downloads.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Testing memory constructor imports...")

try:
    # Core imports
    from memory_constructor.data.schemas import MemoryItem, SFTSample
    print("✓ Data schemas")

    from memory_constructor.data.memory_store import MemoryStore
    print("✓ Memory store")

    from memory_constructor.data.webshop_parser import WebShopParser
    print("✓ WebShop parser")

    from memory_constructor.data.hindsight_labeling import HindsightLabeler
    print("✓ Hindsight labeler")

    from memory_constructor.data.sft_dataset import MemoryConstructorSFTDataset
    print("✓ SFT dataset")

    from memory_constructor.models.agent import GPT5Agent, MockAgent
    print("✓ Agent")

    from memory_constructor.environment.agent_instance import MemoryConstructorAgentInstance
    print("✓ Agent instance")

    from memory_constructor.training.candidate_scorer import CandidateScorer
    print("✓ Candidate scorer")

    from memory_constructor.training.best_of_n_trainer import BestOfNTrainer
    print("✓ Best-of-N trainer")

    from memory_constructor.evaluation import Evaluator, MetricsCalculator
    print("✓ Evaluation framework")

    print("\n✅ All core imports successful!")

    # Test basic functionality
    print("\nTesting basic functionality...")

    store = MemoryStore()
    store.add({"keys": ["test"], "value": "test memory", "step": 0})
    assert len(store.memories) == 1
    print("✓ Memory store works")

    agent = MockAgent()
    action = agent.act("test observation", [])
    assert isinstance(action, str)
    print("✓ Mock agent works")

    print("\n✅ All tests passed!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
