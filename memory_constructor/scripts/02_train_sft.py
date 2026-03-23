#!/usr/bin/env python3
"""
Script 02: Train memory constructor with SFT.

This script trains the memory constructor model using supervised fine-tuning (SFT)
on hindsight-labeled trajectories, via OpenRLHF's SFTTrainer.
"""

import argparse
import math
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from transformers.optimization import get_scheduler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openrlhf.models import Actor
from openrlhf.trainer.sft_trainer import SFTTrainer
from openrlhf.utils import get_strategy, get_tokenizer

from memory_constructor.data.sft_dataset import create_sft_dataloader
from memory_constructor.utils.logging_utils import LOG_FORMAT, setup_file_logging

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_args(model_config: dict, training_config: dict, cli_args) -> argparse.Namespace:
    """
    Build an argparse.Namespace with all attributes required by OpenRLHF's
    get_strategy, Actor, SFTTrainer, and SFTTrainer.fit.
    """
    sft_cfg = training_config["sft"]

    args = argparse.Namespace(
        # Model
        pretrain=cli_args.pretrain or model_config["base_model"],
        param_dtype=model_config.get("param_dtype", "bf16"),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
        load_in_4bit=False,
        lora_rank=model_config.get("lora_rank", 0),
        lora_alpha=model_config.get("lora_alpha", 16),
        lora_dropout=model_config.get("lora_dropout", 0),
        target_modules=model_config.get("target_modules", "all-linear"),
        packing_samples=False,
        use_liger_kernel=False,
        # DeepSpeed
        zero_stage=cli_args.zero_stage if cli_args.zero_stage is not None else sft_cfg.get("zero_stage", 2),
        micro_train_batch_size=cli_args.micro_batch_size or sft_cfg.get("micro_train_batch_size", 4),
        train_batch_size=cli_args.batch_size or sft_cfg.get("train_batch_size", 128),
        max_norm=sft_cfg.get("max_grad_norm", 1.0),
        seed=training_config.get("seed", 42),
        full_determinism=False,
        local_rank=cli_args.local_rank,
        zpg=1,
        adam_offload=cli_args.adam_offload,
        grad_accum_dtype=None,
        overlap_comm=False,
        ds_tensor_parallel_size=1,
        deepcompile=False,
        # Training
        max_epochs=cli_args.num_epochs or sft_cfg.get("max_epochs", 3),
        learning_rate=cli_args.learning_rate or sft_cfg.get("learning_rate", 1e-5),
        lr_scheduler="cosine_with_min_lr",
        lr_warmup_ratio=0.03,
        l2=sft_cfg.get("weight_decay", 0.01),
        adam_betas=(0.9, 0.95),
        pretrain_mode=False,
        gradient_checkpointing=sft_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_use_reentrant=False,
        aux_loss_coef=0,
        max_len=cli_args.max_length or model_config.get("max_length", 2048),
        # Logging & Checkpointing
        logging_steps=sft_cfg.get("logging_steps", 1),
        eval_steps=sft_cfg.get("eval_steps", -1),
        save_steps=cli_args.save_steps if cli_args.save_steps > 0 else sft_cfg.get("save_steps", -1),
        save_path=cli_args.output_dir,
        ckpt_path=os.path.join(cli_args.output_dir, "checkpoints"),
        max_ckpt_num=3,
        max_ckpt_mem=int(1e8),
        load_checkpoint=False,
        save_hf_ckpt=True,
        disable_ds_ckpt=False,
        # Wandb / Tensorboard
        use_wandb=None,
        wandb_org=None,
        wandb_group=None,
        wandb_project="memory-constructor-sft",
        wandb_run_name="mc_sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
        use_tensorboard=None,
        # Ring attention
        ring_attn_size=1,
        ring_head_stride=1,
        # Tokenizer
        disable_fast_tokenizer=False,
    )
    return args


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train memory constructor with SFT")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml")
    parser.add_argument("--training_config", type=str, default="configs/training.yaml")
    parser.add_argument("--data_config", type=str, default="configs/data.yaml")
    parser.add_argument("--train_data", type=str, default="data/webshop/processed/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/webshop/processed/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--pretrain", type=str, default=None, help="Override model name from config")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=None)
    parser.add_argument("--write_loss_weight", type=float, default=1.0,
                        help="Loss weight multiplier for write=True samples (>1.0 penalizes no-write bias)")
    parser.add_argument("--adam_offload", action="store_true", default=False,
                        help="Offload Adam optimizer states to CPU (saves GPU memory)")
    parser.add_argument("--save_steps", type=int, default=-1,
                        help="Save checkpoint every N update steps (-1 to disable)")
    parser.add_argument("--oversample_write", type=int, default=1,
                        help="Oversample write=True samples N times (1=no oversampling)")
    cli_args = parser.parse_args()

    # Load configs early to get model name for log filename
    model_config = load_config(cli_args.model_config)
    training_config = load_config(cli_args.training_config)
    pretrain_name = cli_args.pretrain or model_config["base_model"]

    # Setup file logging
    model_short = pretrain_name.split("/")[-1]
    log_path = setup_file_logging(f"sft_{model_short}", project_root)

    logger.info("=" * 80)
    logger.info("Memory Constructor SFT Training")
    logger.info("=" * 80)

    # Build full args namespace for OpenRLHF
    logger.info("[Step 1] Loading configurations...")
    args = build_args(model_config, training_config, cli_args)

    logger.info(f"Model: {args.pretrain}")
    logger.info(f"Train batch size: {args.train_batch_size}")
    logger.info(f"Micro batch size: {args.micro_train_batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max epochs: {args.max_epochs}")

    # Setup distributed strategy
    logger.info("[Step 2] Setting up distributed strategy...")
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # Initialize model
    logger.info("[Step 3] Initializing model...")
    model = Actor(
        args.pretrain,
        attn_implementation=args.attn_implementation,
        param_dtype=args.param_dtype,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
    )

    # Load tokenizer
    logger.info("[Step 4] Loading tokenizer...")
    tokenizer = get_tokenizer(
        args.pretrain, model.model, "right", strategy,
        use_fast=not args.disable_fast_tokenizer,
    )
    strategy.print(model)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": args.gradient_checkpointing_use_reentrant,
            }
        )

    # Create dataloaders
    logger.info("[Step 5] Creating dataloaders...")
    train_dataloader = create_sft_dataloader(
        data_path=cli_args.train_data,
        tokenizer=tokenizer,
        batch_size=args.micro_train_batch_size,
        max_length=args.max_len,
        shuffle=True,
        num_workers=0,
        write_loss_weight=cli_args.write_loss_weight,
        oversample_write=cli_args.oversample_write,
    )

    val_dataloader = None
    if cli_args.val_data and os.path.exists(cli_args.val_data) and os.path.getsize(cli_args.val_data) > 0:
        val_dataloader = create_sft_dataloader(
            data_path=cli_args.val_data,
            tokenizer=tokenizer,
            batch_size=args.micro_train_batch_size,
            max_length=args.max_len,
            shuffle=False,
            num_workers=0,
            write_loss_weight=cli_args.write_loss_weight,
        )

    logger.info(f"Train batches: {len(train_dataloader)}")
    if val_dataloader:
        logger.info(f"Val batches: {len(val_dataloader)}")
    else:
        logger.info("No validation data (empty or missing file)")

    # Setup optimizer
    logger.info("[Step 6] Setting up optimizer...")
    optim = strategy.create_optimizer(
        model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2,
    )

    # Scheduler
    num_update_steps_per_epoch = len(train_dataloader) // max(
        args.train_batch_size // args.micro_train_batch_size, 1
    )
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    logger.info(f"Num update steps per epoch: {num_update_steps_per_epoch}")
    logger.info(f"Total training steps: {max_steps}")

    # Prepare models with DeepSpeed
    logger.info("[Step 7] Preparing model with DeepSpeed...")
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # Setup trainer
    logger.info("[Step 8] Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )

    # Train
    logger.info("[Step 9] Starting training...")
    logger.info("=" * 80)
    train_start = time.time()
    trainer.fit(args, consumed_samples=0, num_update_steps_per_epoch=num_update_steps_per_epoch)
    train_elapsed = time.time() - train_start
    logger.info(f"Training completed in {train_elapsed:.1f}s ({train_elapsed/60:.1f}min)")

    # Save final checkpoint
    logger.info("[Step 10] Saving final checkpoint...")
    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy.save_model(model, tokenizer, str(output_dir / "final"))

    logger.info(f"Checkpoint saved to {output_dir / 'final'}")
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
