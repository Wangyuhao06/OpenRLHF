#!/usr/bin/env python3
"""
Script 02: Train memory constructor with SFT.

This script trains the memory constructor model using supervised fine-tuning (SFT)
on hindsight-labeled trajectories.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer
from openrlhf.models import Actor
from openrlhf.trainer.sft_trainer import SFTTrainer
from openrlhf.utils import get_strategy, get_tokenizer

from memory_constructor.data.sft_dataset import create_sft_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train memory constructor with SFT")

    # Config files
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data.yaml",
        help="Path to data config file",
    )

    # Data paths
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/webshop/processed/train.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/webshop/processed/val.jsonl",
        help="Path to validation data",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/sft",
        help="Output directory for checkpoints",
    )

    # Training hyperparameters (override config)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)

    # Distributed training
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=None)

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training loop."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Memory Constructor SFT Training")
    logger.info("=" * 80)

    # Load configs
    logger.info("\n[Step 1] Loading configurations...")
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    data_config = load_config(args.data_config)

    # Override with command line args
    if args.batch_size is not None:
        training_config["sft"]["batch_size"] = args.batch_size
    if args.micro_batch_size is not None:
        training_config["sft"]["micro_batch_size"] = args.micro_batch_size
    if args.learning_rate is not None:
        training_config["sft"]["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        training_config["sft"]["num_epochs"] = args.num_epochs
    if args.max_length is not None:
        model_config["max_length"] = args.max_length
    if args.zero_stage is not None:
        training_config["sft"]["zero_stage"] = args.zero_stage

    logger.info(f"Model: {model_config['model_name']}")
    logger.info(f"Batch size: {training_config['sft']['batch_size']}")
    logger.info(f"Learning rate: {training_config['sft']['learning_rate']}")
    logger.info(f"Epochs: {training_config['sft']['num_epochs']}")

    # Setup distributed strategy
    logger.info("\n[Step 2] Setting up distributed strategy...")
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # Load tokenizer
    logger.info("\n[Step 3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"],
        trust_remote_code=True,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Tokenizer loaded: {model_config['model_name']}")
    logger.info(f"Vocab size: {len(tokenizer)}")

    # Create dataloaders
    logger.info("\n[Step 4] Creating dataloaders...")
    train_dataloader = create_sft_dataloader(
        data_path=args.train_data,
        tokenizer=tokenizer,
        batch_size=training_config["sft"]["micro_batch_size"],
        max_length=model_config["max_length"],
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = create_sft_dataloader(
        data_path=args.val_data,
        tokenizer=tokenizer,
        batch_size=training_config["sft"]["micro_batch_size"],
        max_length=model_config["max_length"],
        shuffle=False,
        num_workers=0,
    )

    logger.info(f"Train batches: {len(train_dataloader)}")
    logger.info(f"Val batches: {len(val_dataloader)}")

    # Initialize model
    logger.info("\n[Step 5] Initializing model...")
    model = Actor(
        pretrain_or_model=model_config["model_name"],
        use_flash_attention_2=model_config.get("use_flash_attention_2", True),
        bf16=model_config.get("bf16", True),
        load_in_4bit=False,
        lora_rank=model_config.get("lora_rank", 0),  # 0 = full fine-tuning
        lora_alpha=model_config.get("lora_alpha", 16),
        target_modules=model_config.get("lora_target_modules"),
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )

    logger.info(f"Model initialized: {model_config['model_name']}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    logger.info(
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B"
    )

    # Setup optimizer
    logger.info("\n[Step 6] Setting up optimizer...")
    optimizer = strategy.create_optimizer(
        model,
        lr=training_config["sft"]["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=training_config["sft"].get("weight_decay", 0.01),
    )

    # Setup scheduler
    num_update_steps_per_epoch = len(train_dataloader) // training_config["sft"].get(
        "gradient_accumulation_steps", 1
    )
    total_steps = num_update_steps_per_epoch * training_config["sft"]["num_epochs"]

    from transformers import get_scheduler

    scheduler = get_scheduler(
        name=training_config["sft"].get("lr_scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=training_config["sft"].get("warmup_steps", 100),
        num_training_steps=total_steps,
    )

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {training_config['sft'].get('warmup_steps', 100)}")

    # Setup trainer
    logger.info("\n[Step 7] Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        scheduler=scheduler,
        max_norm=training_config["sft"].get("max_grad_norm", 1.0),
        max_epochs=training_config["sft"]["num_epochs"],
        tokenizer=tokenizer,
    )

    # Train
    logger.info("\n[Step 8] Starting training...")
    logger.info("=" * 80)

    trainer.fit(
        args=args,
        consumed_samples=0,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
    )

    # Save final checkpoint
    logger.info("\n[Step 9] Saving final checkpoint...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy.save_model(
        model,
        tokenizer=tokenizer,
        output_dir=str(output_dir / "final"),
    )

    logger.info(f"Checkpoint saved to {output_dir / 'final'}")

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
