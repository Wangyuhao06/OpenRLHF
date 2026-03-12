#!/usr/bin/env python3
"""
Script 03: Generate candidates for best-of-n training.

This script uses the SFT-trained model to generate multiple candidate
memories per step using vLLM for fast batch inference.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_constructor.training.candidate_sampler import CandidateSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate candidates for best-of-n training"
    )

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to SFT-trained model checkpoint",
    )

    # Data
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/webshop/processed/train.jsonl",
        help="Input SFT samples file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/webshop/candidates/train_candidates.jsonl",
        help="Output candidates file",
    )

    # Sampling parameters
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=4,
        help="Number of candidates per sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )

    # vLLM parameters
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0-1)",
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
    """Main candidate generation pipeline."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Candidate Generation for Best-of-N Training")
    logger.info("=" * 80)

    # Initialize sampler
    logger.info("\n[Step 1] Initializing candidate sampler...")
    sampler = CandidateSampler(
        model_path=args.model_path,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Generate candidates
    logger.info("\n[Step 2] Generating candidates...")
    sampler.sample_from_file(
        input_path=args.input_file,
        output_path=args.output_file,
        max_samples=args.max_samples,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Candidate generation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
