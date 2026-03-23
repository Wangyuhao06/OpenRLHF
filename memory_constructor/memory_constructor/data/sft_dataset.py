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

from memory_constructor.data.schemas import SFTSample, MemoryItem, CandidateMemoryList
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
        write_loss_weight: float = 1.0,
        oversample_write: int = 1,
    ):
        """
        Initialize SFT dataset.

        Args:
            data_path: Path to JSONL file with SFTSample objects
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            prompt_template: Optional custom prompt template
            write_loss_weight: Loss weight multiplier for write=True samples.
                Values > 1.0 penalize the model more for getting write samples wrong,
                counteracting class imbalance where no-write dominates.
            oversample_write: Repeat write=True samples this many times (1=no oversampling).
                E.g. oversample_write=3 triples the write samples in the dataset.
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or self._get_default_prompt_template()
        self.write_loss_weight = write_loss_weight

        # Load samples
        self.samples = self._load_samples()

        # Oversample write samples to combat class imbalance
        if oversample_write > 1:
            write_samples = [s for s in self.samples if s.target_memory.write]
            for _ in range(oversample_write - 1):
                self.samples.extend(write_samples)

        num_writes = sum(1 for s in self.samples if s.target_memory.write)
        logger.info(
            f"Loaded {len(self.samples)} samples from {data_path} "
            f"(max_length={max_length}, write_loss_weight={write_loss_weight}, "
            f"oversample_write={oversample_write}, "
            f"writes={num_writes}, no-writes={len(self.samples) - num_writes})"
        )

    def _get_default_prompt_template(self) -> str:
        """Get default prompt template for memory constructor."""
        return """You are a memory constructor for a shopping agent. You decide what to store
in the agent's memory to help it complete its task.

Agent Task:
{instruction}

Key Requirements:
{task_requirements}

Current Observation:
{observation}

Local History (last 3 steps):
{local_history}

Current Memory Store:
{memory_store}

Currently Retrievable (top matches for this observation):
{retrieval_context}

Budget Remaining: {budget_remaining}
Episode Progress: {episode_progress:.1%}

Decide: should you write a memory now? Consider:
- Does this observation contain NEW information not already in memory?
- Would this information help the agent in future steps?
- Is this information retrievable from existing memory?

Respond in JSON:
{{
    "write": true/false,
    "keys": ["key1", ...],
    "value": "..."
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

    def _format_task_requirements(self, sample: SFTSample) -> str:
        """Format task requirements from sample metadata."""
        task_requirements = sample.metadata.get("task_requirements", [])
        if isinstance(task_requirements, dict):
            lines = [
                f"- {key}: {value}"
                for key, value in task_requirements.items()
                if value not in (None, "", [])
            ]
            return "\n".join(lines) if lines else "(Not available)"
        if isinstance(task_requirements, list):
            lines = [f"- {item}" for item in task_requirements if str(item).strip()]
            return "\n".join(lines) if lines else "(Not available)"
        if isinstance(task_requirements, str) and task_requirements.strip():
            return task_requirements
        return "(Not available)"

    def _format_retrieval_context(self, sample: SFTSample) -> str:
        """Format retrievable memory context from sample metadata."""
        retrieval_context = sample.metadata.get("retrieval_context", [])
        if isinstance(retrieval_context, list):
            lines = []
            for item in retrieval_context:
                if not isinstance(item, dict):
                    continue
                keys = ", ".join(item.get("keys", [])) or "(no keys)"
                score = item.get("score")
                score_text = f" | score={score}" if score is not None else ""
                value_preview = item.get("value_preview", "")
                lines.append(
                    f"- Step {item.get('step_id', '?')}: keys={keys}{score_text} | value={value_preview}"
                )
            return "\n".join(lines) if lines else "(Nothing retrievable yet)"
        if isinstance(retrieval_context, str) and retrieval_context.strip():
            return retrieval_context
        return "(Nothing retrievable yet)"

    def _format_prompt(self, sample: SFTSample) -> str:
        """
        Format prompt for a sample.

        Args:
            sample: SFTSample object

        Returns:
            Formatted prompt string
        """
        if sample.local_history:
            history_text = "\n".join(sample.local_history)
        else:
            history_text = "(No previous steps)"

        if sample.memory_store:
            memory_lines = []
            for i, mem in enumerate(sample.memory_store, 1):
                memory_lines.append(
                    f"{i}. Keys: {', '.join(mem.keys)} | Value: {mem.value[:160]}"
                )
            memory_text = "\n".join(memory_lines)
        else:
            memory_text = "(Empty)"

        prompt = self.prompt_template.format(
            instruction=sample.metadata.get("instruction", ""),
            task_requirements=self._format_task_requirements(sample),
            observation=sample.observation,
            local_history=history_text,
            memory_store=memory_text,
            retrieval_context=self._format_retrieval_context(sample),
            budget_remaining=sample.budget_remaining,
            episode_progress=sample.episode_progress,
            max_num_keys=4,
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

    def __getitem__(self, idx: int):
        """
        Get a single sample.

        Returns a tuple of (input_ids, attention_mask, loss_mask) tensors,
        each of shape [1, seq_len], compatible with OpenRLHF's SFTTrainer.
        """
        sample = self.samples[idx]

        # Format prompt and target
        prompt = self._format_prompt(sample)
        target = self._format_target(sample)

        # Combine prompt + target with EOS
        text = (prompt + target).rstrip("\n")
        if not text.endswith(self.tokenizer.eos_token):
            text += " " + self.tokenizer.eos_token

        # Tokenize combined text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"]  # [1, seq_len]
        attention_mask = encoding["attention_mask"]  # [1, seq_len]

        # Compute prompt length for loss masking
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids_len = prompt_encoding["attention_mask"].int().sum().item()

        # Build loss_mask: weight for target tokens, 0.0 for prompt tokens
        # Apply write_loss_weight for write=True samples to penalize no-write bias
        weight = self.write_loss_weight if sample.target_memory.write else 1.0
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        loss_mask[0, prompt_ids_len - 1 : -1] = weight

        # Ensure EOS token is present
        input_ids[0][-1] = self.tokenizer.eos_token_id
        attention_mask[0][-1] = True

        return input_ids, attention_mask, loss_mask

    def collate_fn(self, item_list):
        """
        Collate function compatible with OpenRLHF's SFTTrainer.

        Returns a tuple of (input_ids, attention_masks, loss_masks),
        each of shape [batch_size, max_seq_len].
        """
        from openrlhf.utils.utils import zero_pad_sequences

        input_ids = []
        attention_masks = []
        loss_masks = []

        for input_id, attention_mask, loss_mask in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            loss_masks.append(loss_mask)

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        loss_masks = zero_pad_sequences(loss_masks, "right")
        return input_ids, attention_masks, loss_masks


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


class BestOfNDataset(Dataset):
    """
    Dataset for best-of-n training stage.

    Loads CandidateMemoryList from scored JSONL, selects the best-scored
    candidate per sample, and formats as training examples.
    The collate_fn returns dict batches compatible with BestOfNTrainer.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []  # List of (prompt_str, target_str)

        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cml = CandidateMemoryList.from_dict(data)
                    best = self._pick_best(cml)
                    prompt = self._format_prompt(cml)
                    target = json.dumps(
                        {"write": best.memory_item.write,
                         "keys": best.memory_item.keys,
                         "value": best.memory_item.value},
                        ensure_ascii=False,
                    )
                    self.samples.append((prompt, target))
                except Exception as e:
                    logger.warning(f"Failed to parse sample: {e}")

        logger.info(f"Loaded {len(self.samples)} best-of-n samples from {data_path}")

    def _pick_best(self, cml: CandidateMemoryList):
        """Return the candidate with the highest total_score."""
        return max(cml.candidates, key=lambda c: c.total_score)

    def _format_prompt(self, cml: CandidateMemoryList) -> str:
        history_text = "\n".join(cml.local_history) if cml.local_history else "(No previous steps)"
        if cml.memory_store:
            memory_lines = [
                f"{i}. Keys: {', '.join(m.keys)} | Value: {m.value[:100]}..."
                for i, m in enumerate(cml.memory_store, 1)
            ]
            memory_text = "\n".join(memory_lines)
        else:
            memory_text = "(Empty)"
        return (
            f"You are a memory constructor for a shopping agent.\n\n"
            f"Current Observation:\n{cml.observation}\n\n"
            f"Local History:\n{history_text}\n\n"
            f"Memory Store:\n{memory_text}\n\n"
            f"Budget Remaining: {cml.budget_remaining}\n"
            f"Episode Progress: {cml.episode_progress:.1%}\n\n"
            f"Respond in JSON format:\n"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, target = self.samples[idx]
        text = (prompt + target).rstrip("\n") + " " + self.tokenizer.eos_token
        encoding = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding=False, return_tensors="pt", add_special_tokens=False,
        )
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        # labels: -100 for prompt tokens, token ids for target tokens
        prompt_len = self.tokenizer(
            prompt, max_length=self.max_length, truncation=True,
            padding=False, return_tensors="pt", add_special_tokens=False,
        )["attention_mask"].int().sum().item()
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return input_ids, attention_mask, labels

    def collate_fn(self, items):
        input_ids_list, attn_list, labels_list = zip(*items)
        max_len = max(x.size(0) for x in input_ids_list)
        pad_id = self.tokenizer.pad_token_id or 0

        def pad(tensors, pad_val):
            out = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                out[i, :t.size(0)] = t
            return out

        return {
            "input_ids": pad(input_ids_list, pad_id),
            "attention_mask": pad(attn_list, 0),
            "labels": pad(labels_list, -100),
        }


def create_sft_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    shuffle: bool = True,
    num_workers: int = 0,
    write_loss_weight: float = 1.0,
    oversample_write: int = 1,
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
        write_loss_weight: Loss weight multiplier for write=True samples
        oversample_write: Repeat write=True samples this many times

    Returns:
        DataLoader instance
    """
    dataset = MemoryConstructorSFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        write_loss_weight=write_loss_weight,
        oversample_write=oversample_write,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )

    return dataloader
