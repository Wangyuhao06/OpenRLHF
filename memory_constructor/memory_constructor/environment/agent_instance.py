"""
Memory Constructor Agent Instance.

This module implements the AgentInstance for memory constructor training,
providing a unified interface for both static trajectory replay (SFT, best-of-n)
and dynamic trajectory generation (RL).
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
import torch

from openrlhf.utils.agent import AgentInstanceBase

from memory_constructor.environment.memory_store import MemoryStore, MemoryBudgetExceeded
from memory_constructor.models.retriever import HybridRetriever
from memory_constructor.models.agent import GPT5Agent, MockAgent
from memory_constructor.data.schemas import MemoryItem
from memory_constructor.utils.json_utils import parse_json_robust, validate_memory_item

logger = logging.getLogger(__name__)


class MemoryConstructorAgentInstance(AgentInstanceBase):
    """
    Agent instance for memory constructor training.

    Unified interface for:
    - Static trajectory replay (SFT, best-of-n)
    - Dynamic trajectory generation (RL)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize agent instance.

        Args:
            config: Configuration dictionary with:
                - episode_memory_budget: Maximum memories per episode
                - max_value_tokens: Maximum tokens in memory value
                - max_key_tokens: Maximum tokens per key
                - max_num_keys: Maximum number of keys
                - retrieval_k: Number of memories to retrieve
                - bm25_weight: Weight for BM25 retrieval
                - dense_weight: Weight for dense retrieval
                - api_key: OpenAI API key (optional)
                - use_mock_agent: Whether to use mock agent (for testing)
                - retrieval_hit_weight: Weight for retrieval hit reward
                - write_cost_penalty: Penalty for writing memory
                - compactness_weight: Weight for compactness reward
        """
        super().__init__()
        self.config = config

        # Initialize components
        self.retriever = HybridRetriever(
            bm25_weight=config.get("bm25_weight", 0.5),
            dense_weight=config.get("dense_weight", 0.5),
            retrieval_k=config.get("retrieval_k", 3),
        )

        # Initialize fixed agent
        if config.get("use_mock_agent", False):
            self.fixed_agent = MockAgent()
            logger.info("Using MockAgent for testing")
        else:
            self.fixed_agent = GPT5Agent(
                api_key=config.get("api_key"),
                temperature=config.get("agent_temperature", 0.7),
            )
            logger.info("Using GPT5Agent")

        # State variables
        self.memory_store = None
        self.mode = None  # "static" or "dynamic"
        self.trajectory_data = None  # For static mode
        self.current_step = 0
        self.instruction = None

        logger.info("Initialized MemoryConstructorAgentInstance")

    async def reset(self, states: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Initialize episode.

        For static trajectories:
            states = {
                "trajectory_id": "...",
                "trajectory_data": [...],  # List of step dicts
                "instruction": "..."
            }

        For dynamic trajectories:
            states = {
                "task": "...",
                "webshop_env": env,
                "instruction": "..."
            }

        Args:
            states: Initial state dictionary
            **kwargs: Additional arguments

        Returns:
            Initial observation dictionary
        """
        # Initialize memory store
        self.memory_store = MemoryStore(
            max_capacity=self.config.get("episode_memory_budget", 20)
        )
        self.current_step = 0

        # Check if static or dynamic
        if "trajectory_data" in states:
            # Static trajectory replay
            self.mode = "static"
            self.trajectory_data = states["trajectory_data"]
            self.instruction = states.get("instruction", "")

            if not self.trajectory_data:
                raise ValueError("Empty trajectory data")

            observation = self.trajectory_data[0]["observation"]

        else:
            # Dynamic trajectory generation
            self.mode = "dynamic"
            self.webshop_env = states.get("webshop_env")
            self.instruction = states.get("instruction", "")

            if self.webshop_env is None:
                raise ValueError("webshop_env required for dynamic mode")

            observation = self.webshop_env.reset()

        # Format initial state
        initial_state = self._format_observation_state(observation)

        logger.debug(f"Reset episode in {self.mode} mode")

        return initial_state

    async def step(self, states: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute one step of memory constructor + agent.

        Args:
            states: State dictionary with:
                - observation_text: Current observation
                - action_text: Constructor output (JSON string)
            **kwargs: Additional arguments

        Returns:
            Step result dictionary with:
                - rewards: Total reward
                - scores: Score (0-1)
                - environment_feedback: Next observation
                - done: Whether episode is done
                - extra_logs: Additional logging info
        """
        observation_text = states.get("observation_text", "")
        action_text = states.get("action_text", "")

        # 1. Parse constructor action
        constructor_action = parse_json_robust(action_text)
        constructor_action = validate_memory_item(
            constructor_action,
            max_key_tokens=self.config.get("max_key_tokens", 64),
            max_value_tokens=self.config.get("max_value_tokens", 256),
            max_num_keys=self.config.get("max_num_keys", 4),
        )

        # 2. Update memory store
        memory_written = False
        if constructor_action.get("write", False):
            try:
                memory_item = MemoryItem(
                    write=True,
                    keys=constructor_action["keys"],
                    value=constructor_action["value"],
                    timestamp=time.time(),
                    step_id=self.current_step,
                    metadata={},
                )
                self.memory_store.append(memory_item)
                memory_written = True
            except MemoryBudgetExceeded:
                logger.warning("Memory budget exceeded, skipping write")
                memory_written = False

        # 3. Agent queries retriever
        agent_query = self.fixed_agent.generate_query(
            observation=observation_text,
            instruction=self.instruction,
        )
        retrieved_memories = self.retriever.retrieve(agent_query, self.memory_store)

        # 4. Agent acts
        if self.mode == "static":
            # Use ground-truth action from trajectory
            if self.current_step >= len(self.trajectory_data):
                # Episode done
                return self._create_done_result()

            step_data = self.trajectory_data[self.current_step]
            agent_action = step_data.get("action", "")
            next_observation = (
                self.trajectory_data[self.current_step + 1]["observation"]
                if self.current_step + 1 < len(self.trajectory_data)
                else ""
            )
            step_reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False) or (
                self.current_step + 1 >= len(self.trajectory_data)
            )

        else:
            # Dynamic: agent generates action
            agent_action = self.fixed_agent.act(
                observation=observation_text,
                retrieved_memories=retrieved_memories,
                instruction=self.instruction,
            )

            # Execute action in environment
            next_observation, step_reward, done, info = self.webshop_env.step(
                agent_action
            )

        # 5. Compute rewards
        task_reward = step_reward
        auxiliary_rewards = self._compute_auxiliary_rewards(
            constructor_action=constructor_action,
            memory_written=memory_written,
            retrieved_memories=retrieved_memories,
        )

        total_reward = (
            task_reward
            + self.config.get("retrieval_hit_weight", 0.1)
            * auxiliary_rewards["retrieval_hit"]
            + self.config.get("write_cost_penalty", 0.05)
            * auxiliary_rewards["write_cost"]
            + self.config.get("compactness_weight", 0.02)
            * auxiliary_rewards["compactness"]
        )

        # 6. Update state
        self.current_step += 1

        # 7. Prepare next observation
        next_state = self._format_observation_state(next_observation)

        # 8. Create result
        result = {
            "rewards": total_reward,
            "scores": torch.sigmoid(torch.tensor(total_reward)).item(),
            "environment_feedback": next_state["observation"],
            "done": done,
            "sampling_params": None,  # Use default
            "extra_logs": {
                "task_reward": task_reward,
                "retrieval_hit": auxiliary_rewards["retrieval_hit"],
                "write_cost": auxiliary_rewards["write_cost"],
                "compactness": auxiliary_rewards["compactness"],
                "write_action": memory_written,
                "num_memories": len(self.memory_store.memories),
                "budget_remaining": self.memory_store.get_budget_remaining(),
                "num_retrieved": len(retrieved_memories),
                "agent_query": agent_query,
                "agent_action": agent_action,
            },
        }

        return result

    def _format_observation_state(self, observation: str) -> Dict[str, Any]:
        """
        Format observation into state dictionary.

        Args:
            observation: Observation text

        Returns:
            State dictionary
        """
        episode_progress = (
            self.current_step / len(self.trajectory_data)
            if self.mode == "static" and self.trajectory_data
            else 0.0
        )

        return {
            "observation": observation,
            "memory_store": self.memory_store.to_dict(),
            "budget_remaining": self.memory_store.get_budget_remaining(),
            "episode_progress": episode_progress,
            "instruction": self.instruction,
        }

    def _compute_auxiliary_rewards(
        self,
        constructor_action: Dict[str, Any],
        memory_written: bool,
        retrieved_memories: List[MemoryItem],
    ) -> Dict[str, float]:
        """
        Compute auxiliary rewards.

        Args:
            constructor_action: Constructor action dict
            memory_written: Whether memory was actually written
            retrieved_memories: Retrieved memories

        Returns:
            Dictionary of auxiliary rewards
        """
        rewards = {}

        # Retrieval hit reward (if memory was written and something was retrieved)
        if memory_written and len(retrieved_memories) > 0:
            rewards["retrieval_hit"] = 0.1
        else:
            rewards["retrieval_hit"] = 0.0

        # Write cost penalty
        if memory_written:
            rewards["write_cost"] = -0.05
        else:
            rewards["write_cost"] = 0.0

        # Compactness reward (encourage compression)
        if memory_written:
            value_length = len(constructor_action.get("value", "").split())
            max_tokens = self.config.get("max_value_tokens", 256)
            compactness = 1.0 - (value_length / max_tokens)
            rewards["compactness"] = 0.02 * max(0.0, compactness)
        else:
            rewards["compactness"] = 0.0

        return rewards

    def _create_done_result(self) -> Dict[str, Any]:
        """Create result for done episode."""
        return {
            "rewards": 0.0,
            "scores": 0.5,
            "environment_feedback": "",
            "done": True,
            "sampling_params": None,
            "extra_logs": {
                "task_reward": 0.0,
                "retrieval_hit": 0.0,
                "write_cost": 0.0,
                "compactness": 0.0,
                "write_action": False,
                "num_memories": len(self.memory_store.memories),
                "budget_remaining": self.memory_store.get_budget_remaining(),
            },
        }
