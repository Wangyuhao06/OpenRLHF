#!/usr/bin/env python3
"""
Test with real GPT-5 API and models.
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from memory_constructor.models.agent import GPT5Agent
from memory_constructor.data.memory_store import MemoryStore
from memory_constructor.data.schemas import MemoryItem

# Load API key
load_dotenv("/home/yuhao/code/.env")
api_key = os.getenv("OPENAI_KEY")
base_url = os.getenv("OPENAI_BASEURL")

if not api_key:
    print("❌ OPENAI_KEY not found in /home/yuhao/code/.env")
    sys.exit(1)

print("Testing with real GPT-5 API...")
print("=" * 60)

# Initialize agent
print("\n[1] Initializing GPT-5 agent...")
agent = GPT5Agent(api_key=api_key, model="gpt-4", base_url=base_url)
print("✓ Agent initialized")

# Create test memory store
print("\n[2] Creating test memory store...")
store = MemoryStore()
store.add({
    "keys": ["product", "search"],
    "value": "User is looking for wireless headphones under $100",
    "step": 0,
})
store.add({
    "keys": ["preference", "brand"],
    "value": "User prefers Sony or Bose brands",
    "step": 1,
})
print(f"✓ Memory store created with {len(store.memories)} memories")

# Test query generation
print("\n[3] Testing query generation...")
observation = "You see a list of wireless headphones. Some are from Sony, Bose, and JBL."
local_history = ["search[wireless headphones]", "click[filter by price]"]

query = agent.generate_query(observation, local_history)
print(f"✓ Generated query: {query}")

# Test action generation
print("\n[4] Testing action generation...")
retrieved_memories = store.get_all()
action = agent.act(observation, retrieved_memories)
print(f"✓ Generated action: {action}")

print("\n" + "=" * 60)
print("✅ All API tests passed!")
print("\nNext steps:")
print("1. Prepare WebShop trajectory data")
print("2. Run preprocessing: python scripts/01_preprocess_webshop.py")
print("3. Train SFT model: python scripts/02_train_sft.py")
