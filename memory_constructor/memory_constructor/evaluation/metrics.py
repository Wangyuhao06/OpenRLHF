"""
Evaluation metrics for memory constructor.

This module defines metrics for evaluating the quality of constructed memories
and their impact on downstream task performance.
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # Task performance
    task_success_rate: float
    task_reward_mean: float
    task_reward_std: float

    # Memory quality
    memory_precision: float  # Fraction of useful memories written
    memory_recall: float  # Fraction of useful moments captured
    memory_f1: float

    # Memory efficiency
    avg_memories_per_episode: float
    avg_memory_length: float
    memory_redundancy: float  # Fraction of redundant memories

    # Retrieval quality
    retrieval_precision: float  # Fraction of retrieved memories that are useful
    retrieval_recall: float  # Fraction of useful memories that are retrieved
    retrieval_f1: float

    # Computational cost
    avg_computational_cost: float
    cost_efficiency: float  # Task reward per unit cost

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "task_success_rate": self.task_success_rate,
            "task_reward_mean": self.task_reward_mean,
            "task_reward_std": self.task_reward_std,
            "memory_precision": self.memory_precision,
            "memory_recall": self.memory_recall,
            "memory_f1": self.memory_f1,
            "avg_memories_per_episode": self.avg_memories_per_episode,
            "avg_memory_length": self.avg_memory_length,
            "memory_redundancy": self.memory_redundancy,
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "retrieval_f1": self.retrieval_f1,
            "avg_computational_cost": self.avg_computational_cost,
            "cost_efficiency": self.cost_efficiency,
        }


class MetricsCalculator:
    """Calculate evaluation metrics from episode results."""

    def __init__(self):
        """Initialize calculator."""
        pass

    def calculate(self, episodes: List[Dict[str, Any]]) -> EvaluationMetrics:
        """
        Calculate metrics from episode results.

        Args:
            episodes: List of episode results, each containing:
                - task_reward: Final task reward
                - task_success: Whether task succeeded
                - memories_written: List of memories written
                - memories_retrieved: List of memories retrieved per step
                - ground_truth_memories: Optional ground truth memories
                - computational_cost: Total computational cost

        Returns:
            EvaluationMetrics object
        """
        # Task performance
        task_rewards = [ep["task_reward"] for ep in episodes]
        task_successes = [ep["task_success"] for ep in episodes]

        task_success_rate = np.mean(task_successes)
        task_reward_mean = np.mean(task_rewards)
        task_reward_std = np.std(task_rewards)

        # Memory quality (if ground truth available)
        if "ground_truth_memories" in episodes[0]:
            memory_precision, memory_recall, memory_f1 = self._calculate_memory_quality(
                episodes
            )
        else:
            memory_precision = memory_recall = memory_f1 = 0.0

        # Memory efficiency
        memories_per_episode = [len(ep["memories_written"]) for ep in episodes]
        avg_memories_per_episode = np.mean(memories_per_episode)

        memory_lengths = []
        for ep in episodes:
            for mem in ep["memories_written"]:
                memory_lengths.append(len(mem["value"].split()))
        avg_memory_length = np.mean(memory_lengths) if memory_lengths else 0.0

        memory_redundancy = self._calculate_redundancy(episodes)

        # Retrieval quality (if ground truth available)
        if "ground_truth_memories" in episodes[0]:
            (
                retrieval_precision,
                retrieval_recall,
                retrieval_f1,
            ) = self._calculate_retrieval_quality(episodes)
        else:
            retrieval_precision = retrieval_recall = retrieval_f1 = 0.0

        # Computational cost
        computational_costs = [ep["computational_cost"] for ep in episodes]
        avg_computational_cost = np.mean(computational_costs)

        # Cost efficiency (reward per unit cost)
        cost_efficiency = task_reward_mean / avg_computational_cost if avg_computational_cost > 0 else 0.0

        return EvaluationMetrics(
            task_success_rate=task_success_rate,
            task_reward_mean=task_reward_mean,
            task_reward_std=task_reward_std,
            memory_precision=memory_precision,
            memory_recall=memory_recall,
            memory_f1=memory_f1,
            avg_memories_per_episode=avg_memories_per_episode,
            avg_memory_length=avg_memory_length,
            memory_redundancy=memory_redundancy,
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            retrieval_f1=retrieval_f1,
            avg_computational_cost=avg_computational_cost,
            cost_efficiency=cost_efficiency,
        )

    def _calculate_memory_quality(
        self, episodes: List[Dict[str, Any]]
    ) -> tuple[float, float, float]:
        """Calculate memory precision, recall, and F1."""
        total_precision = 0.0
        total_recall = 0.0
        num_episodes = 0

        for ep in episodes:
            written = set(self._memory_to_key(m) for m in ep["memories_written"])
            ground_truth = set(
                self._memory_to_key(m) for m in ep["ground_truth_memories"]
            )

            if len(written) > 0:
                precision = len(written & ground_truth) / len(written)
            else:
                precision = 1.0  # No false positives

            if len(ground_truth) > 0:
                recall = len(written & ground_truth) / len(ground_truth)
            else:
                recall = 1.0  # No false negatives

            total_precision += precision
            total_recall += recall
            num_episodes += 1

        avg_precision = total_precision / num_episodes
        avg_recall = total_recall / num_episodes

        if avg_precision + avg_recall > 0:
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            f1 = 0.0

        return avg_precision, avg_recall, f1

    def _calculate_retrieval_quality(
        self, episodes: List[Dict[str, Any]]
    ) -> tuple[float, float, float]:
        """Calculate retrieval precision, recall, and F1."""
        total_precision = 0.0
        total_recall = 0.0
        num_retrievals = 0

        for ep in episodes:
            ground_truth = set(
                self._memory_to_key(m) for m in ep["ground_truth_memories"]
            )

            for retrieved in ep["memories_retrieved"]:
                if len(retrieved) == 0:
                    continue

                retrieved_keys = set(self._memory_to_key(m) for m in retrieved)

                if len(retrieved_keys) > 0:
                    precision = len(retrieved_keys & ground_truth) / len(
                        retrieved_keys
                    )
                else:
                    precision = 1.0

                if len(ground_truth) > 0:
                    recall = len(retrieved_keys & ground_truth) / len(ground_truth)
                else:
                    recall = 1.0

                total_precision += precision
                total_recall += recall
                num_retrievals += 1

        if num_retrievals > 0:
            avg_precision = total_precision / num_retrievals
            avg_recall = total_recall / num_retrievals
        else:
            avg_precision = avg_recall = 0.0

        if avg_precision + avg_recall > 0:
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            f1 = 0.0

        return avg_precision, avg_recall, f1

    def _calculate_redundancy(self, episodes: List[Dict[str, Any]]) -> float:
        """Calculate memory redundancy (fraction of similar memories)."""
        total_redundancy = 0.0
        num_episodes = 0

        for ep in episodes:
            memories = ep["memories_written"]
            if len(memories) <= 1:
                continue

            # Count pairs of similar memories
            redundant_pairs = 0
            total_pairs = 0

            for i in range(len(memories)):
                for j in range(i + 1, len(memories)):
                    total_pairs += 1
                    if self._are_similar(memories[i], memories[j]):
                        redundant_pairs += 1

            if total_pairs > 0:
                total_redundancy += redundant_pairs / total_pairs
                num_episodes += 1

        return total_redundancy / num_episodes if num_episodes > 0 else 0.0

    def _memory_to_key(self, memory: Dict[str, Any]) -> str:
        """Convert memory to a unique key for comparison."""
        keys = tuple(sorted(memory.get("keys", [])))
        value = memory.get("value", "")
        return f"{keys}|{value}"

    def _are_similar(self, mem1: Dict[str, Any], mem2: Dict[str, Any]) -> bool:
        """Check if two memories are similar (simple heuristic)."""
        # Check key overlap
        keys1 = set(mem1.get("keys", []))
        keys2 = set(mem2.get("keys", []))

        if len(keys1 & keys2) / max(len(keys1), len(keys2)) > 0.5:
            return True

        # Check value similarity (simple word overlap)
        words1 = set(mem1.get("value", "").lower().split())
        words2 = set(mem2.get("value", "").lower().split())

        if max(len(words1), len(words2)) > 0 and len(words1 & words2) / max(len(words1), len(words2)) > 0.7:
            return True

        return False
