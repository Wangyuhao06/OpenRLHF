"""
Trainer for best-of-n training stage.

This module provides a trainer that trains on the best-scored candidates
from the candidate generation and scoring pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BestOfNTrainer:
    """Trainer for best-of-n training stage."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset,
        val_dataset,
        output_dir: str,
        batch_size: int = 64,
        learning_rate: float = 5e-6,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.01,
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        deepspeed: bool = False,
        deepspeed_config: Optional[str] = None,
        local_rank: int = -1,
    ):
        """Initialize trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        # Distributed training
        self.deepspeed = deepspeed
        self.deepspeed_config = deepspeed_config
        self.local_rank = local_rank
        self.device = torch.device(
            f"cuda:{local_rank}" if local_rank >= 0 else "cuda"
        )

        # Move model to device
        if not deepspeed:
            self.model = self.model.to(self.device)

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
        )

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train(self):
        """Run training loop."""
        logger.info("Starting training...")

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            self._train_epoch(epoch)

            # Evaluate at end of epoch
            val_loss = self._evaluate()
            logger.info(f"Epoch {epoch + 1} validation loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best")
                logger.info(f"New best model saved (val_loss: {val_loss:.4f})")

        logger.info("\nTraining complete!")

    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Logging
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Evaluation
            if self.global_step % self.eval_interval == 0:
                val_loss = self._evaluate()
                logger.info(
                    f"Step {self.global_step}: train_loss={avg_loss:.4f}, "
                    f"val_loss={val_loss:.4f}"
                )
                self.model.train()

            # Checkpointing
            if self.global_step % self.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step}")

    def _evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs.loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")
