#!/usr/bin/env python3
"""
Script 05: Train model using best-of-n candidates.

This script trains the memory constructor model on the best-scored
candidates from the candidate generation and scoring pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

from memory_constructor.data.sft_dataset import BestOfNDataset
from memory_constructor.training.best_of_n_trainer import BestOfNTrainer
from memory_constructor.utils.logging_utils import LOG_FORMAT, setup_file_logging

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train model with best-of-n")

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to SFT-trained model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/best_of_n",
        help="Output directory for checkpoints",
    )

    # Data
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/webshop/candidates/train_scored.jsonl",
        help="Training data with scored candidates",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/webshop/candidates/val_scored.jsonl",
        help="Validation data with scored candidates",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)

    # DeepSpeed
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def main():
    """Main best-of-n training pipeline."""
    args = parse_args()

    model_short = args.model_path.rstrip("/").split("/")[-1]
    setup_file_logging(f"best_of_n_{model_short}", project_root)

    logger.info("=" * 80)
    logger.info("Best-of-N Training")
    logger.info("=" * 80)

    # Initialize model and tokenizer
    logger.info("\n[Step 1] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )

    # Load datasets
    logger.info("\n[Step 2] Loading datasets...")
    train_dataset = BestOfNDataset(
        data_path=args.train_file,
        tokenizer=tokenizer,
    )
    val_dataset = BestOfNDataset(
        data_path=args.val_file,
        tokenizer=tokenizer,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Initialize trainer
    logger.info("\n[Step 3] Initializing trainer...")
    trainer = BestOfNTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        deepspeed=args.deepspeed,
        deepspeed_config=args.deepspeed_config,
        local_rank=args.local_rank,
    )

    # Train
    logger.info("\n[Step 4] Starting training...")
    trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("Best-of-N training complete!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
