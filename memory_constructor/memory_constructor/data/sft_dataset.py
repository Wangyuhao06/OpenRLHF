"""
SFT Dataset for memory constructor training.

This module implements PyTorch Dataset for supervised fine-tuning (SFT)
of the memory constructor model using hindsight-labeled trajectories.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset

from memory_constructor.data.schemas import SFTSample, MemoryItem
from memory_constructor.utils.json_utils import format_memory_prompt

logger = logging.getLogger(__name__)


class MemoryConstructorSFTDataset(Dataset):
    """
    PyTorch Dataset for SFT training of memory constructor.

    Loads hindsight-labeled samples and formats them as prompts for
    the memory constructor model.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize SFT dataset.

        Args:
            data_path: Path to JSONL file with SFTSample objects
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            prompt_template: Optional custom prompt template
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or self._get_default_prompt_template()

        # Load samples
        self.samples = self._load_samples()

        logger.info(
            f"Loaded {len(self.samples)} samples from {data_path} "
            f"(max_length={max_length})"
        )

    def _get_default_prompt_template(self) -> str:
        """Get default prompt template for memory constructor."""
        return """You are a memory constructor for a shopping agent. Your task is to decide whether to write a memory at this step, and if so, what keys and value to use.

Task: {instruction}

Current Observation:
{observation}

Local History (last 3 steps):
{local_history}

Current Memory Store:
{memory_store}

Budget Remaining: {budget_remaining}
Episode Progress: {episode_progress:.1%}

Should you write a memory at this step? If yes, provide:
1. Keys: Searchable keywords (max {max_num_keys} keys, each max {max_key_tokens} tokens)
2. Value: Compressed, faithful summary (max {max_value_tokens} tokens)

Respond in JSON format:
{{
    "write": true/false,
    "keys": ["key1", "key2", ...],
    "value": "compressed memory content"
}}"""

    def _load_samples(self) -> List[SFTSample]:
        """Load SFT samples from JSONL file."""
        samples = []

        with open(self.data_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    sample = SFTSample.from_dict(data)
                    samples.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        return samples

    def _format_prompt(self, sample: SFTSample) -> str:
        """
        Format prompt for a sample.

        Args:
            sample: SFTSample object

        Returns:
            Formatted prompt string
        """
        # Format local history
        if sample.local_history:
            history_text = "\n".join(sample.local_history)
        else:
            history_text = "(No previous steps)"

        # Format memory store
        if sample.memory_store:
            memory_lines = []
            for i, mem in enumerate(sample.memory_store, 1):
                memory_lines.append(
                    f"{i}. Keys: {', '.join(mem.keys)} | Value: {mem.value[:100]}..."
                )
            memory_text = "\n".join(memory_lines)
        else:
            memory_text = "(Empty)"

        # Fill template
        prompt = self.prompt_template.format(
            instruction=sample.metadata.get("instruction", ""),
            observation=sample.observation,
            local_history=history_text,
            memory_store=memory_text,
            budget_remaining=sample.budget_remaining,
            episode_progress=sample.episode_progress,
            max_num_keys=4,  # From config
            max_key_tokens=64,
            max_value_tokens=256,
        )

        return prompt

    def _format_target(self, sample: SFTSample) -> str:
        """
        Format target output for a sample.

        Args:
            sample: SFTSample object

        Returns:
            Target JSON string
        """
        target = {
            "write": sample.target_memory.write,
            "keys": sample.target_memory.keys,
            "value": sample.target_memory.value,
        }

        return json.dumps(target, ensure_ascii=False)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - input_ids: Tokenized input
                - attention_mask: Attention mask
                - labels: Tokenized target (for loss computation)
                - sample_id: Sample identifier
        """
        sample = self.samples[idx]

        # Format prompt and target
        prompt = self._format_prompt(sample)
        target = self._format_target(sample)

        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length - 256,  # Leave room for target
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target,
            max_length=256,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # Combine prompt + target
        input_ids = prompt_encoding["input_ids"] + target_encoding["input_ids"]
        attention_mask = prompt_encoding["attention_mask"] + target_encoding["attention_mask"]

        # Create labels (mask prompt, only compute loss on target)
        labels = [-100] * len(prompt_encoding["input_ids"]) + target_encoding["input_ids"]

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_id": sample.sample_id,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Batched tensors
        """
        # Find max length in batch
        max_len = max(len(item["input_ids"]) for item in batch)

        # Pad all sequences to max length
        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            # Pad input_ids
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(
                item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )

            # Pad attention_mask
            attention_mask.append(item["attention_mask"] + [0] * pad_len)

            # Pad labels
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class MemoryConstructorMultiTurnDataset(Dataset):
    """
    Multi-turn dataset for memory constructor training.

    Groups samples by trajectory and formats them as multi-turn conversations.
    This is useful for training with OpenRLHF's multi-turn support.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        max_turns_per_trajectory: int = 10,
    ):
        """
        Initialize multi-turn dataset.

        Args:
            data_path: Path to JSONL file with SFTSample objects
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            max_turns_per_trajectory: Maximum turns per trajectory
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_turns_per_trajectory = max_turns_per_trajectory

        # Load and group samples by trajectory
        self.trajectories = self._load_and_group_samples()

        logger.info(
            f"Loaded {len(self.trajectories)} trajectories from {data_path} "
            f"(max_turns={max_turns_per_trajectory})"
        )

    def _load_and_group_samples(self) -> List[List[SFTSample]]:
        """Load samples and group by trajectory."""
        # Load all samples
        all_samples = []
        with open(self.data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = SFTSample.from_dict(data)
                    all_samples.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to parse sample: {e}")
                    continue

        # Group by trajectory_id
        trajectory_dict = {}
        for sample in all_samples:
            traj_id = sample.trajectory_id
            if traj_id not in trajectory_dict:
                trajectory_dict[traj_id] = []
            trajectory_dict[traj_id].append(sample)

        # Sort samples within each trajectory by step_id
        trajectories = []
        for traj_id, samples in trajectory_dict.items():
            samples.sort(key=lambda s: s.step_id)
            # Limit to max turns
            samples = samples[: self.max_turns_per_trajectory]
            trajectories.append(samples)

        return trajectories

    def __len__(self) -> int:
        """Return number of trajectories."""
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single trajectory (multi-turn conversation).

        Args:
            idx: Trajectory index

        Returns:
            Dictionary with trajectory data
        """
        trajectory_samples = self.trajectories[idx]

        # Format as multi-turn conversation
        turns = []
        for sample in trajectory_samples:
            # Create turn with prompt and target
            prompt = self._format_prompt_simple(sample)
            target = self._format_target(sample)

            turns.append({"prompt": prompt, "target": target})

        return {
            "trajectory_id": trajectory_samples[0].trajectory_id,
            "turns": turns,
            "num_turns": len(turns),
        }

    def _format_prompt_simple(self, sample: SFTSample) -> str:
        """Format a simple prompt for multi-turn setting."""
        return f"Observation: {sample.observation}\nBudget: {sample.budget_remaining}\nProgress: {sample.episode_progress:.1%}"

    def _format_target(self, sample: SFTSample) -> str:
        """Format target output."""
        target = {
            "write": sample.target_memory.write,
            "keys": sample.target_memory.keys,
            "value": sample.target_memory.value,
        }
        return json.dumps(target, ensure_ascii=False)


def create_sft_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for SFT training.

    Args:
        data_path: Path to JSONL file
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    dataset = MemoryConstructorSFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )

    return dataloader
