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
import copy
import logging
import sys
from pathlib import Path
import json
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_constructor.data.webshop_parser import WebShopParser
from memory_constructor.data.hindsight_labeling import (
    DemandAwareHindsightLabeler,
    HeuristicLabeler,
    HindsightLabeler,
)
from memory_constructor.data.schemas import SFTSample, MemoryItem
from memory_constructor.data.memory_store import MemoryStore
from memory_constructor.utils.logging_utils import LOG_FORMAT, setup_file_logging

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
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
        choices=["llm_judge", "heuristic", "demand_aware"],
        default="llm_judge",
        help="Labeling method: llm_judge, heuristic, or demand_aware",
    )
    parser.add_argument(
        "--contribution_threshold",
        type=float,
        default=0.3,
        help="Minimum contribution score to write memory (0-1)",
    )
    parser.add_argument(
        "--budget_ratio",
        type=float,
        default=2.0,
        help="Memory budget ratio for demand-aware labeling (budget = steps * ratio)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Strict mode: raise error on any fallback instead of silently falling back",
    )

    # LLM API settings
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="LLM model name for labeling (default: deepseek-chat)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="API base URL (auto-detected from .env if not set)",
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


def compute_retrieval_hit_metrics(all_samples: list, retrieval_k: int = 3):
    """
    Compute retrieval hit metrics for demand-aware labeled samples.

    For each sample that has write=True (a memory was stored), we check
    whether that memory is retrievable from the memory_store at each
    future step that references it (via demanded_by in metadata).

    Also performs a simulated full-trajectory retrieval test:
    for each trajectory, rebuild the memory store step-by-step, and
    at each step with existing memories, test if the retrieval of
    the current observation returns relevant memories.
    """
    from memory_constructor.models.retriever import HybridRetriever

    retriever = HybridRetriever(retrieval_k=retrieval_k)

    # Group samples by trajectory
    traj_samples = {}
    for sample in all_samples:
        traj_id = sample.trajectory_id
        if traj_id not in traj_samples:
            traj_samples[traj_id] = []
        traj_samples[traj_id].append(sample)

    # Sort by step_id within each trajectory
    for traj_id in traj_samples:
        traj_samples[traj_id].sort(key=lambda s: s.step_id)

    total_retrieval_queries = 0
    total_hits = 0
    total_misses = 0
    per_trajectory_stats = []

    for traj_id, samples in traj_samples.items():
        # Build memory store step-by-step and test retrieval
        accumulated_memories = []
        traj_queries = 0
        traj_hits = 0
        traj_misses = 0
        miss_details = []

        for sample in samples:
            # Test retrieval at this step (if we have memories)
            if accumulated_memories:
                store = MemoryStore()
                for mem in accumulated_memories:
                    store.add(copy.deepcopy(mem))

                results = retriever.retrieve(
                    sample.observation,
                    store,
                    k=retrieval_k,
                    return_scores=True,
                )
                retrieved_step_ids = {mem.step_id for mem, _ in results}

                # Check: is any memory in the top-k relevant?
                # A memory is relevant if this step was in its demanded_by list
                for mem in accumulated_memories:
                    demanded_by = mem.metadata.get("demanded_by", [])
                    if sample.step_id in demanded_by:
                        traj_queries += 1
                        if mem.step_id in retrieved_step_ids:
                            traj_hits += 1
                        else:
                            traj_misses += 1
                            scores_text = ", ".join(
                                f"step{m.step_id}={s:.3f}" for m, s in results
                            )
                            miss_details.append(
                                f"  step {sample.step_id} needed memory from "
                                f"step {mem.step_id} (keys={mem.keys}), "
                                f"but top-{retrieval_k} was [{scores_text}]"
                            )

            # Add this step's memory to accumulated store
            if sample.target_memory.write:
                accumulated_memories.append(copy.deepcopy(sample.target_memory))

        total_retrieval_queries += traj_queries
        total_hits += traj_hits
        total_misses += traj_misses

        hit_rate = traj_hits / max(1, traj_queries)
        per_trajectory_stats.append({
            "trajectory_id": traj_id,
            "queries": traj_queries,
            "hits": traj_hits,
            "misses": traj_misses,
            "hit_rate": hit_rate,
        })

        if miss_details:
            logger.warning(
                "Trajectory %s retrieval misses (%d/%d):\n%s",
                traj_id,
                traj_misses,
                traj_queries,
                "\n".join(miss_details),
            )

    overall_hit_rate = total_hits / max(1, total_retrieval_queries)

    logger.info("=" * 80)
    logger.info("RETRIEVAL HIT METRICS")
    logger.info("=" * 80)
    logger.info(f"  Total demand-based retrieval queries: {total_retrieval_queries}")
    logger.info(f"  Hits (memory in top-{retrieval_k}):    {total_hits}")
    logger.info(f"  Misses:                               {total_misses}")
    logger.info(f"  Overall Hit Rate:                     {overall_hit_rate:.2%}")
    logger.info("-" * 40)
    for ts in per_trajectory_stats:
        logger.info(
            "  Trajectory %-12s: %d/%d hits (%.2f%%)",
            ts["trajectory_id"],
            ts["hits"],
            ts["queries"],
            ts["hit_rate"] * 100,
        )
    logger.info("=" * 80)

    return {
        "total_queries": total_retrieval_queries,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "overall_hit_rate": overall_hit_rate,
        "per_trajectory": per_trajectory_stats,
    }


def main():
    """Main preprocessing pipeline."""
    args = parse_args()

    setup_file_logging("preprocess_webshop", project_root)

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
            model=args.model,
            base_url=args.base_url,
            contribution_threshold=args.contribution_threshold,
            max_value_tokens=args.max_value_tokens,
            max_key_tokens=args.max_key_tokens,
            max_num_keys=args.max_num_keys,
        )
    elif args.labeling_method == "demand_aware":
        labeler = DemandAwareHindsightLabeler(
            model=args.model,
            base_url=args.base_url,
            contribution_threshold=args.contribution_threshold,
            max_value_tokens=args.max_value_tokens,
            max_key_tokens=args.max_key_tokens,
            max_num_keys=args.max_num_keys,
            budget_ratio=args.budget_ratio,
            strict=args.strict,
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

    # Step 4: Compute retrieval hit metrics (demand-aware only)
    retrieval_metrics = None
    if args.labeling_method == "demand_aware":
        logger.info("\n[Step 4] Computing retrieval hit metrics...")
        all_labeled = train_samples + val_samples + test_samples
        retrieval_metrics = compute_retrieval_hit_metrics(all_labeled, retrieval_k=3)

    # Step 5: Save processed data
    logger.info("\n[Step 5] Saving processed data...")
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
        "budget_ratio": args.budget_ratio,
        "strict": args.strict,
        "statistics": stats,
    }
    if retrieval_metrics is not None:
        metadata["retrieval_metrics"] = retrieval_metrics

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {output_dir / 'metadata.json'}")

    logger.info("\n" + "=" * 80)
    logger.info("Preprocessing complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
