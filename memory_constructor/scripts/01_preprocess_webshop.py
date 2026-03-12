#!/usr/bin/env python3
"""
Script 01: Preprocess WebShop trajectories.

This script:
1. Parses raw WebShop trajectory files
2. Filters for successful trajectories
3. Applies hindsight labeling (LLM-as-judge or heuristic)
4. Splits into train/val/test sets
5. Saves processed SFT samples
"""

import argparse
import logging
import sys
from pathlib import Path
import json
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_constructor.data.webshop_parser import WebShopParser
from memory_constructor.data.hindsight_labeling import HindsightLabeler, HeuristicLabeler
from memory_constructor.data.schemas import SFTSample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess WebShop trajectories for SFT training"
    )

    # Input/output paths
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="List of input JSONL trajectory files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/webshop/processed",
        help="Output directory for processed data",
    )

    # Filtering
    parser.add_argument(
        "--use_only_successful",
        action="store_true",
        default=True,
        help="Only use successful trajectories",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to process (for testing)",
    )

    # Labeling method
    parser.add_argument(
        "--labeling_method",
        type=str,
        choices=["llm_judge", "heuristic"],
        default="llm_judge",
        help="Labeling method: llm_judge (GPT-5) or heuristic",
    )
    parser.add_argument(
        "--contribution_threshold",
        type=float,
        default=0.3,
        help="Minimum contribution score to write memory (0-1)",
    )

    # Model constraints
    parser.add_argument(
        "--max_value_tokens",
        type=int,
        default=256,
        help="Maximum tokens in memory value",
    )
    parser.add_argument(
        "--max_key_tokens",
        type=int,
        default=64,
        help="Maximum tokens per key",
    )
    parser.add_argument(
        "--max_num_keys",
        type=int,
        default=4,
        help="Maximum number of keys per memory",
    )

    # Splitting
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )

    return parser.parse_args()


def save_samples(samples: list, output_path: Path):
    """Save SFT samples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in samples:
            # Convert to dict
            sample_dict = asdict(sample)
            # Write as JSON line
            f.write(json.dumps(sample_dict) + "\n")

    logger.info(f"Saved {len(samples)} samples to {output_path}")


def main():
    """Main preprocessing pipeline."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("WebShop Trajectory Preprocessing")
    logger.info("=" * 80)

    # Step 1: Parse trajectories
    logger.info("\n[Step 1] Parsing trajectory files...")
    parser = WebShopParser(use_only_successful=args.use_only_successful)

    all_trajectories = []
    for input_file in args.input_files:
        logger.info(f"Parsing {input_file}...")
        try:
            trajectories = parser.parse_file(input_file)
            all_trajectories.extend(trajectories)
        except Exception as e:
            logger.error(f"Failed to parse {input_file}: {e}")
            continue

    if not all_trajectories:
        logger.error("No trajectories parsed. Exiting.")
        return

    # Limit trajectories if specified
    if args.max_trajectories is not None:
        all_trajectories = all_trajectories[: args.max_trajectories]
        logger.info(f"Limited to {len(all_trajectories)} trajectories")

    # Print statistics
    stats = parser.get_statistics(all_trajectories)
    logger.info(f"\nTrajectory Statistics:")
    logger.info(f"  Total trajectories: {stats['num_trajectories']}")
    logger.info(f"  Successful: {stats['num_successful']}")
    logger.info(f"  Success rate: {stats['success_rate']:.2%}")
    logger.info(f"  Avg steps: {stats['avg_steps']:.1f}")
    logger.info(f"  Avg reward: {stats['avg_reward']:.2f}")

    # Step 2: Split trajectories
    logger.info("\n[Step 2] Splitting into train/val/test...")
    train_traj, val_traj, test_traj = parser.split_trajectories(
        all_trajectories,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Step 3: Apply hindsight labeling
    logger.info(f"\n[Step 3] Applying hindsight labeling ({args.labeling_method})...")

    if args.labeling_method == "llm_judge":
        labeler = HindsightLabeler(
            contribution_threshold=args.contribution_threshold,
            max_value_tokens=args.max_value_tokens,
            max_key_tokens=args.max_key_tokens,
            max_num_keys=args.max_num_keys,
        )
    else:
        labeler = HeuristicLabeler(
            max_value_tokens=args.max_value_tokens,
            max_key_tokens=args.max_key_tokens,
            max_num_keys=args.max_num_keys,
        )

    # Label each split
    logger.info("Labeling training set...")
    train_samples = labeler.label_trajectories(train_traj)

    logger.info("Labeling validation set...")
    val_samples = labeler.label_trajectories(val_traj)

    logger.info("Labeling test set...")
    test_samples = labeler.label_trajectories(test_traj)

    # Print labeling statistics
    logger.info(f"\nLabeling Statistics:")
    logger.info(f"  Train samples: {len(train_samples)} ({sum(1 for s in train_samples if s.target_memory.write)} writes)")
    logger.info(f"  Val samples: {len(val_samples)} ({sum(1 for s in val_samples if s.target_memory.write)} writes)")
    logger.info(f"  Test samples: {len(test_samples)} ({sum(1 for s in test_samples if s.target_memory.write)} writes)")

    # Step 4: Save processed data
    logger.info("\n[Step 4] Saving processed data...")
    output_dir = Path(args.output_dir)

    save_samples(train_samples, output_dir / "train.jsonl")
    save_samples(val_samples, output_dir / "val.jsonl")
    save_samples(test_samples, output_dir / "test.jsonl")

    # Save metadata
    metadata = {
        "num_trajectories": len(all_trajectories),
        "num_train": len(train_traj),
        "num_val": len(val_traj),
        "num_test": len(test_traj),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
        "labeling_method": args.labeling_method,
        "contribution_threshold": args.contribution_threshold,
        "max_value_tokens": args.max_value_tokens,
        "max_key_tokens": args.max_key_tokens,
        "max_num_keys": args.max_num_keys,
        "statistics": stats,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {output_dir / 'metadata.json'}")

    logger.info("\n" + "=" * 80)
    logger.info("Preprocessing complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
