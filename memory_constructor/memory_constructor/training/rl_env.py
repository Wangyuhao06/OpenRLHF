"""
RL environment wrapper for memory constructor training.

This module provides a gym-like environment interface for training
the memory constructor with RL algorithms like PPO.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

from memory_constructor.environment import MemoryStore, MemoryConstructorAgentInstance
from memory_constructor.models import GPT5Agent
from memory_constructor.data.schemas import MemoryItem

logger = logging.getLogger(__name__)


@dataclass
class RLReward:
    """Reward components for RL training."""

    # Task performance
    task_success: float = 0.0  # 1.0 if task succeeds, 0.0 otherwise
    step_reward: float = 0.0  # Intermediate reward per step

    # Memory quality
    retrieval_hit: float = 0.0  # Reward for writing useful memories
    write_cost: float = 0.0  # Penalty for writing (budget management)
    compactness: float = 0.0  # Reward for compact memories
    redundancy_penalty: float = 0.0  # Penalty for redundant memories

    # Auxiliary rewards
    distillation_reward: float = 0.0  # KL divergence from expert policy

    def total(self, weights: Dict[str, float]) -> float:
        """Compute weighted total reward."""
        return (
            weights.get("task_success", 1.0) * self.task_success +
            weights.get("step_reward", 0.1) * self.step_reward +
            weights.get("retrieval_hit", 0.1) * self.retrieval_hit +
            weights.get("write_cost", -0.05) * self.write_cost +
            weights.get("compactness", 0.02) * self.compactness +
            weights.get("redundancy_penalty", -0.1) * self.redundancy_penalty +
            weights.get("distillation", 0.1) * self.distillation_reward
        )


class MemoryConstructorRLEnv:
    """
    RL environment for memory constructor training.

    This wraps a task environment (e.g., WebShop) and provides
    a gym-like interface for RL training.
    """

    def __init__(
        self,
        task_env: Any,  # WebShop or ALFWorld environment
        agent: GPT5Agent,
        memory_budget: int = 20,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize RL environment.

        Args:
            task_env: Task environment (WebShop, ALFWorld, etc.)
            agent: Agent for taking actions in the task
            memory_budget: Maximum number of memories to write
            reward_weights: Weights for reward components
        """
        self.task_env = task_env
        self.agent = agent
        self.memory_budget = memory_budget
        self.reward_weights = reward_weights or {}

        # Episode state
        self.memory_store = None
        self.step_count = 0
        self.memories_written = 0
        self.episode_history = []

        logger.info(f"Initialized MemoryConstructorRLEnv with budget={memory_budget}")

    def reset(self) -> Dict[str, Any]:
        """
        Reset environment for new episode.

        Returns:
            Initial observation dict
        """
        # Reset task environment
        task_obs = self.task_env.reset()

        # Reset memory store
        self.memory_store = MemoryStore(budget=self.memory_budget)
        self.step_count = 0
        self.memories_written = 0
        self.episode_history = []

        # Create observation
        obs = self._create_observation(task_obs)

        return obs

    def step(
        self,
        memory_action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            memory_action: Memory constructor output
                {
                    "write": bool,
                    "keys": List[str],
                    "value": str
                }

        Returns:
            observation: Next observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        self.step_count += 1
        reward_components = RLReward()

        # 1. Process memory action
        if memory_action.get("write", False):
            memory_item = MemoryItem(
                write=True,
                keys=memory_action.get("keys", []),
                value=memory_action.get("value", ""),
                timestamp=float(self.step_count),
                step_id=self.step_count,
                metadata={}
            )

            try:
                self.memory_store.add(memory_item)
                self.memories_written += 1
                reward_components.write_cost = 1.0

                # Compactness reward
                value_len = len(memory_item.value)
                reward_components.compactness = max(0, 1.0 - value_len / 512)

            except Exception as e:
                logger.warning(f"Failed to add memory: {e}")

        # 2. Agent takes action in task environment
        # Generate query using memory store
        current_obs = self.episode_history[-1] if self.episode_history else ""
        query = self.agent.generate_query(
            observation=current_obs,
            local_history=self.episode_history[-5:],
            memory_store=self.memory_store
        )

        # Retrieve relevant memories
        retrieved = self.memory_store.retrieve(query, k=3)

        # Generate action
        action = self.agent.generate_action(
            observation=current_obs,
            local_history=self.episode_history[-5:],
            retrieved_memories=retrieved
        )

        # Execute action in task environment
        task_obs, task_reward, done, task_info = self.task_env.step(action)

        # 3. Compute rewards
        reward_components.step_reward = task_reward

        if done:
            reward_components.task_success = 1.0 if task_info.get("success", False) else 0.0

        # Retrieval hit reward (check if retrieved memories were useful)
        if len(retrieved) > 0 and task_reward > 0:
            reward_components.retrieval_hit = 1.0

        # Redundancy penalty (check similarity with existing memories)
        if memory_action.get("write", False):
            redundancy = self._compute_redundancy(memory_action)
            reward_components.redundancy_penalty = redundancy

        # Total reward
        total_reward = reward_components.total(self.reward_weights)

        # 4. Update history
        self.episode_history.append(task_obs)

        # 5. Create next observation
        next_obs = self._create_observation(task_obs)

        # 6. Info dict
        info = {
            **task_info,
            "reward_components": reward_components,
            "memories_written": self.memories_written,
            "budget_remaining": self.memory_budget - self.memories_written,
        }

        return next_obs, total_reward, done, info

    def _create_observation(self, task_obs: str) -> Dict[str, Any]:
        """Create observation dict for memory constructor."""
        return {
            "observation": task_obs,
            "local_history": self.episode_history[-5:],
            "memory_store": list(self.memory_store.memories.values()),
            "budget_remaining": self.memory_budget - self.memories_written,
            "episode_progress": self.step_count / 50.0,  # Assume max 50 steps
        }

    def _compute_redundancy(self, memory_action: Dict[str, Any]) -> float:
        """
        Compute redundancy score for a memory.

        Returns:
            Redundancy score (0-1, higher = more redundant)
        """
        if not memory_action.get("write", False):
            return 0.0

        new_value = memory_action.get("value", "")
        if not new_value:
            return 0.0

        # Simple word overlap-based redundancy
        new_words = set(new_value.lower().split())

        max_overlap = 0.0
        for memory in self.memory_store.memories.values():
            existing_words = set(memory.value.lower().split())
            if len(new_words) == 0:
                continue
            overlap = len(new_words & existing_words) / len(new_words)
            max_overlap = max(max_overlap, overlap)

        return max_overlap


class WebShopRLWrapper:
    """Wrapper for WebShop environment to work with RL training."""

    def __init__(self, webshop_env):
        """Initialize wrapper."""
        self.env = webshop_env
        self.current_obs = None
        self.done = False
        self.info = {}

    def reset(self) -> str:
        """Reset environment."""
        self.current_obs = self.env.reset()
        self.done = False
        self.info = {}
        return self.current_obs

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Take a step."""
        obs, reward, done, info = self.env.step(action)
        self.current_obs = obs
        self.done = done
        self.info = info
        return obs, reward, done, info
