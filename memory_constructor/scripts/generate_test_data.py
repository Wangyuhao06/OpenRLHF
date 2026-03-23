#!/usr/bin/env python3
"""Generate synthetic SFT training data for testing the training pipeline."""

import json
import random
import os

random.seed(42)

INSTRUCTIONS = [
    "I need a laptop stand that is adjustable and portable, and price lower than 50.00 dollars",
    "Find me a wireless mouse with ergonomic design under 30 dollars",
    "I want a USB-C hub with at least 4 ports and HDMI output, budget 40 dollars",
    "Looking for a mechanical keyboard with blue switches, budget under 60 dollars",
    "I need noise-cancelling earbuds for running, waterproof, under 45 dollars",
    "Find a portable phone charger with 20000mAh capacity under 35 dollars",
    "I want a webcam with 1080p resolution and built-in microphone under 50 dollars",
    "Looking for a tablet stand that works for both iPad and Kindle under 25 dollars",
    "I need a desk lamp with adjustable brightness and USB charging port under 30 dollars",
    "Find me a laptop bag that fits 15.6 inch laptops with water resistance under 40 dollars",
]

OBSERVATIONS = [
    "WebShop search results page showing 10 products matching 'laptop stand adjustable'. Product 1: LIFELONG Adjustable Laptop Stand - $29.99, 4.5 stars. Product 2: Rain Design mStand - $44.90, 4.7 stars.",
    "Product detail page: LIFELONG Adjustable Laptop Stand. Price: $29.99. Features: Adjustable height, foldable design, aluminum alloy, compatible with 10-17 inch laptops. Customer rating: 4.5/5 (2,341 reviews).",
    "Product options page showing: Color options: Silver ($29.99), Black ($31.99), Rose Gold ($34.99). Size: Standard. In stock. Free shipping with Prime.",
    "Search results for 'wireless ergonomic mouse': Mouse A - $19.99 4.3★, Mouse B - $24.99 4.6★, Mouse C - $28.99 4.8★, Mouse D - $15.99 3.9★",
    "Product page: Logitech MX Ergo. Price: $24.99. Features: Adjustable trackball angle, 2000 DPI, Bluetooth + USB receiver, rechargeable battery lasts 4 months.",
    "Comparison page: Product A vs Product B. A: cheaper but lower rating. B: better ergonomics, longer battery. Both have wireless connectivity and USB-C charging.",
    "Cart page: 1 item added. LIFELONG Laptop Stand (Silver) - $29.99. Subtotal: $29.99. Proceed to checkout.",
    "Search results: USB-C Hub VAVA 8-in-1 $35.99, Anker PowerExpand $39.99, Satechi Slim $42.99. All have HDMI + USB-C + USB-A ports.",
    "Product page: Mechanical keyboard - Redragon K552. Blue switches, RGB backlit, compact TKL layout. Price: $34.99. Rating: 4.4/5.",
    "Product listing page: Showing earbuds filtered by 'waterproof noise-cancelling'. Top results: JBL Reflect $39.99, Sony WF-SP800N $44.99, Jabra Elite Active $42.99.",
]

ACTIONS = [
    "search[laptop stand adjustable portable]",
    "click[product_1]",
    "click[silver]",
    "click[buy now]",
    "search[wireless ergonomic mouse under 30]",
    "click[product_2]",
    "think[Product B has better reviews and ergonomics, within budget]",
    "click[add to cart]",
    "back",
    "search[USB-C hub HDMI 4 ports]",
]

MEMORY_VALUES = [
    "Found adjustable laptop stand at $29.99, well within $50 budget. Aluminum, foldable, fits 10-17 inch. Good candidate.",
    "Wireless mouse options: best rated is Mouse C at $28.99 (4.8★) but Mouse B at $24.99 (4.6★) offers better value.",
    "USB-C hub comparison: VAVA 8-in-1 at $35.99 is best value with all required ports including HDMI.",
    "Keyboard found: Redragon K552 with blue switches at $34.99, well under $60 budget. TKL layout.",
    "Earbuds shortlist: JBL Reflect ($39.99) cheapest waterproof option. Sony ($44.99) best noise cancelling.",
    "Product A has free shipping. Product B charges $4.99 shipping. Factor in total cost comparison.",
    "Customer reviews mention Product 1 has durability issues after 6 months. Product 2 more reliable.",
    "Cart total: $29.99. Under budget with room for accessories if needed.",
    "Phone charger 20000mAh options: Anker $25.99 (best brand), Baseus $19.99 (budget pick).",
    "Webcam comparison: Logitech C920 ($49.99) vs Aukey ($29.99). Logitech has better low-light performance.",
]

MEMORY_KEYS_OPTIONS = [
    ["laptop stand", "adjustable", "price", "portable"],
    ["wireless mouse", "ergonomic", "under 30"],
    ["USB-C hub", "HDMI", "ports", "price"],
    ["mechanical keyboard", "blue switches", "budget"],
    ["earbuds", "waterproof", "noise-cancelling", "running"],
    ["shipping", "cost comparison", "total price"],
    ["durability", "reviews", "reliability"],
    ["cart", "budget remaining", "total"],
    ["phone charger", "20000mAh", "brand comparison"],
    ["webcam", "1080p", "microphone", "price"],
]


def generate_sample(sample_id, traj_id, step_id, instruction_idx):
    """Generate a single SFT sample."""
    instruction = INSTRUCTIONS[instruction_idx % len(INSTRUCTIONS)]
    obs_idx = (instruction_idx * 3 + step_id) % len(OBSERVATIONS)
    observation = OBSERVATIONS[obs_idx]

    # Build local history
    local_history = []
    for prev_step in range(max(0, step_id - 3), step_id):
        act_idx = (instruction_idx * 5 + prev_step) % len(ACTIONS)
        obs_short = OBSERVATIONS[(instruction_idx * 3 + prev_step) % len(OBSERVATIONS)][:80]
        local_history.append(f"Step {prev_step}: Action: {ACTIONS[act_idx]} | Obs: {obs_short}...")

    # Build memory store (accumulated memories from previous steps)
    memory_store = []
    for prev_step in range(step_id):
        if random.random() < 0.4:  # 40% chance a previous step wrote a memory
            mem_idx = (instruction_idx * 7 + prev_step) % len(MEMORY_VALUES)
            memory_store.append({
                "write": True,
                "keys": MEMORY_KEYS_OPTIONS[mem_idx % len(MEMORY_KEYS_OPTIONS)][:random.randint(2, 4)],
                "value": MEMORY_VALUES[mem_idx],
                "timestamp": float(prev_step * 2 + 1),
                "step_id": prev_step,
                "metadata": {"method": "heuristic"},
            })

    budget_remaining = 20 - len(memory_store)
    total_steps = random.randint(4, 10)
    episode_progress = step_id / total_steps if total_steps > 0 else 0.0

    # Decide write vs no-write (roughly 40% write)
    should_write = random.random() < 0.4

    if should_write:
        mem_idx = (instruction_idx * 7 + step_id) % len(MEMORY_VALUES)
        target_memory = {
            "write": True,
            "keys": MEMORY_KEYS_OPTIONS[mem_idx % len(MEMORY_KEYS_OPTIONS)][:random.randint(2, 4)],
            "value": MEMORY_VALUES[mem_idx],
            "timestamp": float(step_id * 2 + 1),
            "step_id": step_id,
            "metadata": {"method": "heuristic"},
        }
        contribution_score = random.uniform(0.3, 0.9)
    else:
        target_memory = {
            "write": False,
            "keys": [],
            "value": "",
            "timestamp": float(step_id * 2 + 1),
            "step_id": step_id,
            "metadata": {"method": "heuristic"},
        }
        contribution_score = 0.0

    return {
        "sample_id": sample_id,
        "trajectory_id": traj_id,
        "step_id": step_id,
        "observation": observation,
        "local_history": local_history,
        "memory_store": memory_store,
        "budget_remaining": budget_remaining,
        "episode_progress": episode_progress,
        "target_memory": target_memory,
        "future_task": "",
        "hindsight_label": "heuristic",
        "contribution_score": contribution_score,
        "metadata": {
            "instruction": instruction,
            "final_success": random.random() < 0.7,
        },
    }


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "webshop", "processed")
    os.makedirs(out_dir, exist_ok=True)

    # Generate training data: 20 trajectories, 3-8 steps each
    train_samples = []
    sample_counter = 0
    for traj_idx in range(20):
        traj_id = f"traj_{traj_idx:04d}"
        num_steps = random.randint(3, 8)
        instr_idx = traj_idx % len(INSTRUCTIONS)
        for step_id in range(num_steps):
            sample_id = f"{traj_id}_step{step_id}"
            sample = generate_sample(sample_id, traj_id, step_id, instr_idx)
            train_samples.append(sample)
            sample_counter += 1

    # Generate validation data: 5 trajectories
    val_samples = []
    for traj_idx in range(20, 25):
        traj_id = f"traj_{traj_idx:04d}"
        num_steps = random.randint(3, 6)
        instr_idx = traj_idx % len(INSTRUCTIONS)
        for step_id in range(num_steps):
            sample_id = f"{traj_id}_step{step_id}"
            sample = generate_sample(sample_id, traj_id, step_id, instr_idx)
            val_samples.append(sample)

    # Generate test data: 5 trajectories
    test_samples = []
    for traj_idx in range(25, 30):
        traj_id = f"traj_{traj_idx:04d}"
        num_steps = random.randint(3, 6)
        instr_idx = traj_idx % len(INSTRUCTIONS)
        for step_id in range(num_steps):
            sample_id = f"{traj_id}_step{step_id}"
            sample = generate_sample(sample_id, traj_id, step_id, instr_idx)
            test_samples.append(sample)

    # Write files
    for name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {len(samples)} samples to {path}")

    # Write metadata
    metadata = {
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "source": "synthetic_test_data",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written to {os.path.join(out_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
