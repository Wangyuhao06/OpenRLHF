#!/usr/bin/env python3
"""
Script 07a: Prepare RL prompt dataset for memory constructor training.

Generates a JSONL file with WebShop task indices for use as the prompt
dataset in OpenRLHF's RL training pipeline. Each line contains:
  {"prompt": "Task <idx>", "label": "<idx>"}

The actual prompt is constructed dynamically in AgentInstance.reset().
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Prepare RL prompt dataset")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "webshop" / "rl_prompts.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=500,
        help="Number of WebShop task indices to include (0-based)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting task index",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for idx in range(args.start_idx, args.start_idx + args.num_tasks):
            sample = {"prompt": f"Task {idx}", "label": str(idx)}
            f.write(json.dumps(sample) + "\n")
            count += 1

    logger.info(f"Wrote {count} task prompts to {output_path}")
    logger.info(f"Task index range: [{args.start_idx}, {args.start_idx + args.num_tasks - 1}]")


if __name__ == "__main__":
    main()
