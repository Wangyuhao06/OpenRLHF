#!/usr/bin/env python3
"""
Script 04: Score candidates for best-of-n training.

This script scores candidate memories using counterfactual evaluation
and other quality metrics.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_constructor.training.candidate_scorer import CandidateScorer
from memory_constructor.models.retriever import HybridRetriever
from memory_constructor.utils.logging_utils import LOG_FORMAT, setup_file_logging

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Score candidates for best-of-n")

    # Data
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/webshop/candidates/train_candidates.jsonl",
        help="Input candidates file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/webshop/candidates/train_scored.jsonl",
        help="Output scored candidates file",
    )
    parser.add_argument(
        "--trajectory_file",
        type=str,
        default=None,
        help="Optional trajectory file for hindsight scoring",
    )

    # Scoring weights
    parser.add_argument(
        "--hindsight_weight",
        type=float,
        default=0.5,
        help="Weight for hindsight score",
    )
    parser.add_argument(
        "--counterfactual_weight",
        type=float,
        default=0.5,
        help="Weight for counterfactual score",
    )
    parser.add_argument(
        "--retrieval_usefulness_weight",
        type=float,
        default=0.4,
        help="Weight for retrieval usefulness",
    )
    parser.add_argument(
        "--task_success_weight",
        type=float,
        default=0.3,
        help="Weight for task success",
    )
    parser.add_argument(
        "--compactness_weight",
        type=float,
        default=0.1,
        help="Weight for compactness",
    )
    parser.add_argument(
        "--redundancy_weight",
        type=float,
        default=0.1,
        help="Weight for redundancy",
    )
    parser.add_argument(
        "--faithfulness_weight",
        type=float,
        default=0.1,
        help="Weight for faithfulness",
    )

    # Testing
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)",
    )

    return parser.parse_args()


def main():
    """Main candidate scoring pipeline."""
    args = parse_args()

    setup_file_logging("score_candidates", project_root)

    logger.info("=" * 80)
    logger.info("Candidate Scoring for Best-of-N Training")
    logger.info("=" * 80)

    # Initialize retriever
    logger.info("\n[Step 1] Initializing retriever...")
    retriever = HybridRetriever()

    # Initialize scorer
    logger.info("\n[Step 2] Initializing candidate scorer...")
    scorer = CandidateScorer(
        retriever=retriever,
        hindsight_weight=args.hindsight_weight,
        counterfactual_weight=args.counterfactual_weight,
        retrieval_usefulness_weight=args.retrieval_usefulness_weight,
        task_success_weight=args.task_success_weight,
        compactness_weight=args.compactness_weight,
        redundancy_weight=args.redundancy_weight,
        faithfulness_weight=args.faithfulness_weight,
    )

    # Score candidates
    logger.info("\n[Step 3] Scoring candidates...")
    scorer.score_from_file(
        input_path=args.input_file,
        output_path=args.output_file,
        trajectory_file=args.trajectory_file,
        max_samples=args.max_samples,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Candidate scoring complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
