#!/usr/bin/env python3
"""
Script 06: Evaluate memory constructor model.

This script evaluates a trained memory constructor model on test episodes
and reports comprehensive metrics.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_constructor.evaluation import Evaluator
from memory_constructor.models.retriever import HybridRetriever
from memory_constructor.models.agent import GPT5Agent
from memory_constructor.utils.logging_utils import LOG_FORMAT, setup_file_logging

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate memory constructor model")

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )

    # Data
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/webshop/processed/test.jsonl",
        help="Test data file",
    )

    # Evaluation settings
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum episodes to evaluate (for testing)",
    )
    parser.add_argument(
        "--use_agent",
        action="store_true",
        help="Use downstream agent for full episode evaluation",
    )
    parser.add_argument(
        "--agent_api_key",
        type=str,
        default=None,
        help="API key for downstream agent (if use_agent=True)",
    )

    # Output
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/evaluation_results.json",
        help="Output file for results",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )

    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    model_short = args.model_path.rstrip("/").split("/")[-1]
    setup_file_logging(f"evaluate_{model_short}", project_root)

    logger.info("=" * 80)
    logger.info("Memory Constructor Evaluation")
    logger.info("=" * 80)

    # Initialize retriever
    logger.info("\n[Step 1] Initializing retriever...")
    retriever = HybridRetriever()

    # Initialize agent (optional)
    agent = None
    if args.use_agent:
        logger.info("\n[Step 2] Initializing downstream agent...")
        if not args.agent_api_key:
            logger.warning("No API key provided, skipping agent initialization")
        else:
            agent = GPT5Agent(api_key=args.agent_api_key)

    # Initialize evaluator
    logger.info("\n[Step 3] Initializing evaluator...")
    evaluator = Evaluator(
        model_path=args.model_path,
        retriever=retriever,
        agent=agent,
        device=args.device,
    )

    # Run evaluation
    logger.info("\n[Step 4] Running evaluation...")
    metrics = evaluator.evaluate_from_file(
        data_path=args.test_file,
        max_episodes=args.max_episodes,
    )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    logger.info("\nTask Performance:")
    logger.info(f"  Success Rate: {metrics.task_success_rate:.2%}")
    logger.info(f"  Reward Mean: {metrics.task_reward_mean:.4f}")
    logger.info(f"  Reward Std: {metrics.task_reward_std:.4f}")

    logger.info("\nMemory Quality:")
    logger.info(f"  Precision: {metrics.memory_precision:.2%}")
    logger.info(f"  Recall: {metrics.memory_recall:.2%}")
    logger.info(f"  F1: {metrics.memory_f1:.2%}")

    logger.info("\nMemory Efficiency:")
    logger.info(f"  Avg Memories/Episode: {metrics.avg_memories_per_episode:.2f}")
    logger.info(f"  Avg Memory Length: {metrics.avg_memory_length:.2f} words")
    logger.info(f"  Redundancy: {metrics.memory_redundancy:.2%}")

    logger.info("\nRetrieval Quality:")
    logger.info(f"  Precision: {metrics.retrieval_precision:.2%}")
    logger.info(f"  Recall: {metrics.retrieval_recall:.2%}")
    logger.info(f"  F1: {metrics.retrieval_f1:.2%}")

    logger.info("\nComputational Cost:")
    logger.info(f"  Avg Cost: {metrics.avg_computational_cost:.4f}")
    logger.info(f"  Cost Efficiency: {metrics.cost_efficiency:.4f}")

    # Save results
    logger.info("\n[Step 5] Saving results...")
    evaluator.save_results(metrics, args.output_file)

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
