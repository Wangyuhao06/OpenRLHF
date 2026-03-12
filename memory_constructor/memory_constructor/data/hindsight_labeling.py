"""
Hindsight labeling for memory constructor training.

This module implements hindsight labeling using LLM-as-judge to determine:
1. Whether a memory should be written at each step
2. What keys and value to use
3. How much the memory contributes to task success (contribution score)
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import os

from memory_constructor.data.schemas import (
    WebShopTrajectory,
    MemoryItem,
    SFTSample,
)
from memory_constructor.data.webshop_parser import (
    extract_observations_and_actions,
    extract_local_history,
)

logger = logging.getLogger(__name__)


class HindsightLabeler:
    """
    Hindsight labeling using LLM-as-judge.

    This labeler analyzes successful trajectories and determines which
    observations should have been stored in memory for future use.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        temperature: float = 0.3,
        contribution_threshold: float = 0.3,
        max_value_tokens: int = 256,
        max_key_tokens: int = 64,
        max_num_keys: int = 4,
        base_url: Optional[str] = None,
    ):
        """
        Initialize hindsight labeler.

        Args:
            api_key: OpenAI API key (loads from .env if None)
            model: Model name for LLM-as-judge
            temperature: Sampling temperature
            contribution_threshold: Minimum score to write memory (0-1)
            max_value_tokens: Maximum tokens in memory value
            max_key_tokens: Maximum tokens per key
            max_num_keys: Maximum number of keys
            base_url: Custom API base URL (optional)
        """
        # Load API key
        if api_key is None:
            load_dotenv("/home/yuhao/code/.env")
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
            base_url = base_url or os.getenv("OPENAI_BASEURL")
            if api_key is None:
                raise ValueError("OpenAI API key not found")

        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.contribution_threshold = contribution_threshold
        self.max_value_tokens = max_value_tokens
        self.max_key_tokens = max_key_tokens
        self.max_num_keys = max_num_keys

        logger.info(
            f"Initialized HindsightLabeler with model={model}, "
            f"threshold={contribution_threshold}"
        )

    def _create_labeling_prompt(
        self,
        instruction: str,
        observation: str,
        step_id: int,
        total_steps: int,
        future_observations: List[str],
        final_success: bool,
    ) -> str:
        """
        Create prompt for LLM-as-judge to label a step.

        Args:
            instruction: Task instruction
            observation: Current observation
            step_id: Current step number
            total_steps: Total steps in trajectory
            future_observations: List of future observations (for context)
            final_success: Whether task succeeded

        Returns:
            Prompt string
        """
        # Limit future observations to avoid token overflow
        future_obs_preview = future_observations[:3]
        future_text = "\n".join([f"- {obs[:200]}..." for obs in future_obs_preview])

        prompt = f"""You are analyzing a shopping agent's trajectory to determine what information should be stored in memory.

Task: {instruction}

Current Step: {step_id + 1}/{total_steps}
Current Observation:
{observation}

Future Observations (preview):
{future_text}

Final Outcome: {"SUCCESS" if final_success else "FAILURE"}

Question: Should the agent write a memory at this step to help with future decisions?

Consider:
1. Does this observation contain information that will be needed later? (e.g., product details, prices, features)
2. Is this information likely to be forgotten or hard to retrieve later?
3. Would storing this help the agent make better decisions?

Respond in JSON format:
{{
    "should_write": true/false,
    "contribution_score": 0.0-1.0,  // How critical is this memory? (0=not useful, 1=critical)
    "reasoning": "brief explanation",
    "keys": ["key1", "key2", ...],  // Search keywords (max {self.max_num_keys}, each max {self.max_key_tokens} tokens)
    "value": "compressed memory content"  // Concise summary (max {self.max_value_tokens} tokens)
}}

Guidelines:
- Keys should be searchable terms (product names, attributes, constraints)
- Value should be a compressed, faithful summary of relevant information
- Don't write memories for trivial observations or navigation steps
- Focus on information that affects the final purchase decision
"""

        return prompt

    def label_step(
        self,
        trajectory: WebShopTrajectory,
        step_id: int,
        steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Label a single step using LLM-as-judge.

        Args:
            trajectory: WebShopTrajectory object
            step_id: Step ID to label
            steps: List of observation-action pairs from trajectory

        Returns:
            Dictionary with labeling results:
                - should_write: bool
                - contribution_score: float
                - reasoning: str
                - keys: List[str]
                - value: str
        """
        if step_id >= len(steps):
            raise ValueError(f"Step ID {step_id} out of range")

        current_step = steps[step_id]
        observation = current_step["observation"]

        # Get future observations for context
        future_steps = steps[step_id + 1 :]
        future_observations = [s["observation"] for s in future_steps]

        # Create prompt
        prompt = self._create_labeling_prompt(
            instruction=trajectory.instruction,
            observation=observation,
            step_id=step_id,
            total_steps=len(steps),
            future_observations=future_observations,
            final_success=trajectory.is_success,
        )

        try:
            # Call LLM-as-judge
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing agent trajectories and determining what information should be stored in memory.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=512,
                response_format={"type": "json_object"},
            )

            # Parse response
            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            # Validate and fix structure
            result.setdefault("should_write", False)
            result.setdefault("contribution_score", 0.0)
            result.setdefault("reasoning", "")
            result.setdefault("keys", [])
            result.setdefault("value", "")

            # Enforce constraints
            result["keys"] = result["keys"][: self.max_num_keys]
            result["keys"] = [k[: self.max_key_tokens * 4] for k in result["keys"]]  # Rough token limit

            # Truncate value (rough token limit)
            result["value"] = result["value"][: self.max_value_tokens * 4]

            # Apply threshold
            if result["contribution_score"] < self.contribution_threshold:
                result["should_write"] = False

            logger.debug(
                f"Step {step_id}: should_write={result['should_write']}, "
                f"score={result['contribution_score']:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error labeling step {step_id}: {e}")
            # Return safe default
            return {
                "should_write": False,
                "contribution_score": 0.0,
                "reasoning": f"Error: {str(e)}",
                "keys": [],
                "value": "",
            }

    def label_trajectory(
        self, trajectory: WebShopTrajectory
    ) -> List[SFTSample]:
        """
        Label all steps in a trajectory.

        Args:
            trajectory: WebShopTrajectory object

        Returns:
            List of SFTSample objects (one per step)
        """
        # Extract observation-action pairs
        steps = extract_observations_and_actions(trajectory)

        if not steps:
            logger.warning(f"No steps found in trajectory {trajectory.episode_id}")
            return []

        sft_samples = []

        for step_id, step in enumerate(steps):
            # Label this step
            label = self.label_step(trajectory, step_id, steps)

            # Extract local history
            local_history = extract_local_history(
                trajectory, step_id, history_length=3
            )

            # Create memory item
            memory_item = MemoryItem(
                write=label["should_write"],
                keys=label["keys"],
                value=label["value"],
                timestamp=step["timestamp"],
                step_id=step_id,
                metadata={
                    "reasoning": label["reasoning"],
                    "contribution_score": label["contribution_score"],
                },
            )

            # Create SFT sample
            sft_sample = SFTSample(
                sample_id=f"{trajectory.episode_id}_step{step_id}",
                trajectory_id=trajectory.episode_id,
                step_id=step_id,
                observation=step["observation"],
                local_history=local_history,
                memory_store=[],  # Will be populated during dataset creation
                budget_remaining=20,  # Default budget
                episode_progress=step_id / len(steps),
                target_memory=memory_item,
                future_task=steps[step_id + 1]["observation"] if step_id + 1 < len(steps) else "",
                hindsight_label=label["reasoning"],
                contribution_score=label["contribution_score"],
                metadata={
                    "instruction": trajectory.instruction,
                    "final_success": trajectory.is_success,
                },
            )

            sft_samples.append(sft_sample)

        logger.info(
            f"Labeled trajectory {trajectory.episode_id}: "
            f"{len(sft_samples)} samples, "
            f"{sum(1 for s in sft_samples if s.target_memory.write)} writes"
        )

        return sft_samples

    def label_trajectories(
        self, trajectories: List[WebShopTrajectory], max_trajectories: Optional[int] = None
    ) -> List[SFTSample]:
        """
        Label multiple trajectories.

        Args:
            trajectories: List of WebShopTrajectory objects
            max_trajectories: Maximum number of trajectories to label (for testing)

        Returns:
            List of all SFTSample objects
        """
        if max_trajectories is not None:
            trajectories = trajectories[:max_trajectories]

        all_samples = []

        for i, trajectory in enumerate(trajectories):
            logger.info(f"Labeling trajectory {i + 1}/{len(trajectories)}: {trajectory.episode_id}")

            try:
                samples = self.label_trajectory(trajectory)
                all_samples.extend(samples)
            except Exception as e:
                logger.error(f"Failed to label trajectory {trajectory.episode_id}: {e}")
                continue

        logger.info(
            f"Labeled {len(trajectories)} trajectories: "
            f"{len(all_samples)} total samples, "
            f"{sum(1 for s in all_samples if s.target_memory.write)} writes"
        )

        return all_samples


class HeuristicLabeler:
    """
    Simple heuristic-based labeling (fallback when LLM is not available).

    This labeler uses simple rules to determine what to store in memory:
    - Write memory when observation contains product information
    - Extract product IDs, prices, features as keys
    - Store compressed observation as value
    """

    def __init__(
        self,
        max_value_tokens: int = 256,
        max_key_tokens: int = 64,
        max_num_keys: int = 4,
    ):
        """
        Initialize heuristic labeler.

        Args:
            max_value_tokens: Maximum tokens in memory value
            max_key_tokens: Maximum tokens per key
            max_num_keys: Maximum number of keys
        """
        self.max_value_tokens = max_value_tokens
        self.max_key_tokens = max_key_tokens
        self.max_num_keys = max_num_keys

        logger.info("Initialized HeuristicLabeler")

    def _should_write_memory(self, observation: str) -> bool:
        """
        Heuristic to determine if memory should be written.

        Args:
            observation: Observation text

        Returns:
            True if memory should be written
        """
        # Write if observation contains product information
        keywords = ["$", "price", "features", "options", "details", "B0"]
        return any(kw in observation for kw in keywords)

    def _extract_keys(self, observation: str) -> List[str]:
        """
        Extract keys from observation using heuristics.

        Args:
            observation: Observation text

        Returns:
            List of key strings
        """
        keys = []

        # Extract product ID (starts with B0)
        words = observation.split()
        for word in words:
            if word.startswith("B0") and len(word) > 5:
                keys.append(word)
                break

        # Extract price
        for word in words:
            if "$" in word:
                keys.append(f"price {word}")
                break

        # Extract other keywords
        important_words = ["features", "options", "size", "color", "brand"]
        for word in important_words:
            if word in observation.lower():
                keys.append(word)

        return keys[: self.max_num_keys]

    def _compress_value(self, observation: str) -> str:
        """
        Compress observation into memory value.

        Args:
            observation: Observation text

        Returns:
            Compressed value string
        """
        # Simple compression: take first N characters
        max_chars = self.max_value_tokens * 4  # Rough estimate
        return observation[:max_chars]

    def label_trajectory(
        self, trajectory: WebShopTrajectory
    ) -> List[SFTSample]:
        """
        Label trajectory using heuristics.

        Args:
            trajectory: WebShopTrajectory object

        Returns:
            List of SFTSample objects
        """
        steps = extract_observations_and_actions(trajectory)

        if not steps:
            return []

        sft_samples = []

        for step_id, step in enumerate(steps):
            observation = step["observation"]

            # Determine if should write
            should_write = self._should_write_memory(observation)

            # Extract keys and value
            keys = self._extract_keys(observation) if should_write else []
            value = self._compress_value(observation) if should_write else ""

            # Create memory item
            memory_item = MemoryItem(
                write=should_write,
                keys=keys,
                value=value,
                timestamp=step["timestamp"],
                step_id=step_id,
                metadata={"method": "heuristic"},
            )

            # Extract local history
            local_history = extract_local_history(
                trajectory, step_id, history_length=3
            )

            # Create SFT sample
            sft_sample = SFTSample(
                sample_id=f"{trajectory.episode_id}_step{step_id}",
                trajectory_id=trajectory.episode_id,
                step_id=step_id,
                observation=observation,
                local_history=local_history,
                memory_store=[],
                budget_remaining=20,
                episode_progress=step_id / len(steps),
                target_memory=memory_item,
                future_task=steps[step_id + 1]["observation"] if step_id + 1 < len(steps) else "",
                hindsight_label="heuristic",
                contribution_score=0.5 if should_write else 0.0,
                metadata={
                    "instruction": trajectory.instruction,
                    "final_success": trajectory.is_success,
                },
            )

            sft_samples.append(sft_sample)

        return sft_samples

    def label_trajectories(
        self, trajectories: List[WebShopTrajectory]
    ) -> List[SFTSample]:
        """
        Label multiple trajectories.

        Args:
            trajectories: List of trajectories

        Returns:
            List of all SFTSample objects
        """
        all_samples = []

        for trajectory in trajectories:
            samples = self.label_trajectory(trajectory)
            all_samples.extend(samples)

        logger.info(
            f"Labeled {len(trajectories)} trajectories (heuristic): "
            f"{len(all_samples)} samples, "
            f"{sum(1 for s in all_samples if s.target_memory.write)} writes"
        )

        return all_samples
