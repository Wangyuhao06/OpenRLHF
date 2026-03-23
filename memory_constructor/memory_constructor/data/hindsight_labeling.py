"""
Hindsight labeling for memory constructor training.

This module implements hindsight labeling using LLM-as-judge to determine:
1. Whether a memory should be written at each step
2. What keys and value to use
3. How much the memory contributes to task success (contribution score)

Includes three labelers:
- HindsightLabeler: Per-step LLM-as-judge labeling
- HeuristicLabeler: Simple rule-based fallback
- DemandAwareHindsightLabeler: Global demand analysis + retrieval-verified labeling
"""

import copy
import logging
import os
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from memory_constructor.data.memory_store import MemoryStore
from memory_constructor.data.schemas import MemoryItem, SFTSample, WebShopTrajectory
from memory_constructor.data.webshop_parser import (
    extract_local_history,
    extract_observations_and_actions,
)
from memory_constructor.utils.json_utils import parse_json_robust, validate_memory_item

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
        model: str = "deepseek-chat",
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
        if OpenAI is None:
            raise ImportError(
                "openai package is required to use HindsightLabeler. "
                "Install the project dependencies first."
            )

        api_key, base_url = self._resolve_api_config(api_key, base_url)

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
            "Initialized HindsightLabeler with model=%s, threshold=%s",
            model,
            contribution_threshold,
        )

    def _candidate_env_paths(self) -> List[Path]:
        """Return likely .env locations for local development."""
        current_file = Path(__file__).resolve()
        candidates = [
            Path("/home/yuhao/work/code/.env"),
            Path("/home/yuhao/code/.env"),
            Path.cwd() / ".env",
            Path.cwd().parent / ".env",
        ]

        for parent_idx in (2, 3, 4):
            if len(current_file.parents) > parent_idx:
                candidates.append(current_file.parents[parent_idx] / ".env")

        unique_candidates = []
        seen = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            unique_candidates.append(path)
        return unique_candidates

    def _resolve_api_config(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        """Load API credentials from explicit args or common .env locations.

        Priority: DeepSeek > OpenAI (DeepSeek is more accessible in CN networks).
        """
        if api_key is None:
            for env_path in self._candidate_env_paths():
                if env_path.exists():
                    load_dotenv(env_path)

            # Prefer DeepSeek (more accessible) over OpenAI
            deepseek_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")
            openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")

            if deepseek_key:
                api_key = deepseek_key
                base_url = (
                    base_url
                    or os.getenv("DEEPSEEK_BASE_URL")
                    or "https://api.deepseek.com"
                )
                logger.info("Using DeepSeek API (base_url=%s)", base_url)
            elif openai_key:
                api_key = openai_key
                base_url = (
                    base_url
                    or os.getenv("OPENAI_BASEURL")
                    or os.getenv("OPENAI_BASE_URL")
                )

        if api_key is None:
            raise ValueError("OpenAI or compatible API key not found in args or .env")

        return api_key, base_url

    def _call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call the LLM and robustly parse a JSON response."""
        if fallback is None:
            fallback = {}

        request_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.debug(
                "Retrying completion without response_format after error: %s",
                exc,
            )
            request_kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**request_kwargs)

        content = response.choices[0].message.content or ""
        return parse_json_robust(content, fallback=fallback)

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
    "contribution_score": 0.0-1.0,
    "reasoning": "brief explanation",
    "keys": ["key1", "key2", ...],
    "value": "compressed memory content"
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
        future_steps = steps[step_id + 1 :]
        future_observations = [step["observation"] for step in future_steps]

        prompt = self._create_labeling_prompt(
            instruction=trajectory.instruction,
            observation=observation,
            step_id=step_id,
            total_steps=len(steps),
            future_observations=future_observations,
            final_success=trajectory.is_success,
        )

        try:
            result = self._call_json(
                system_prompt=(
                    "You are an expert at analyzing agent trajectories and "
                    "determining what information should be stored in memory."
                ),
                user_prompt=prompt,
                max_tokens=512,
                fallback={
                    "should_write": False,
                    "contribution_score": 0.0,
                    "reasoning": "",
                    "keys": [],
                    "value": "",
                },
            )

            result.setdefault("should_write", False)
            result.setdefault("contribution_score", 0.0)
            result.setdefault("reasoning", "")
            result.setdefault("keys", [])
            result.setdefault("value", "")

            validated = validate_memory_item(
                {
                    "write": result["should_write"],
                    "keys": result["keys"],
                    "value": result["value"],
                },
                max_key_tokens=self.max_key_tokens,
                max_value_tokens=self.max_value_tokens,
                max_num_keys=self.max_num_keys,
            )

            result["should_write"] = validated["write"]
            result["keys"] = validated["keys"]
            result["value"] = validated["value"]

            try:
                result["contribution_score"] = float(result["contribution_score"])
            except (TypeError, ValueError):
                result["contribution_score"] = 0.0
            result["contribution_score"] = max(
                0.0, min(1.0, result["contribution_score"])
            )

            if result["contribution_score"] < self.contribution_threshold:
                result["should_write"] = False
                result["keys"] = []
                result["value"] = ""

            logger.debug(
                "Step %s: should_write=%s, score=%.2f",
                step_id,
                result["should_write"],
                result["contribution_score"],
            )

            return result

        except Exception as exc:
            logger.error("Error labeling step %s: %s", step_id, exc)
            return {
                "should_write": False,
                "contribution_score": 0.0,
                "reasoning": f"Error: {exc}",
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
        steps = extract_observations_and_actions(trajectory)
        if not steps:
            logger.warning("No steps found in trajectory %s", trajectory.episode_id)
            return []

        sft_samples = []

        for step_id, step in enumerate(steps):
            label = self.label_step(trajectory, step_id, steps)
            local_history = extract_local_history(trajectory, step_id, history_length=3)

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

            sft_sample = SFTSample(
                sample_id=f"{trajectory.episode_id}_step{step_id}",
                trajectory_id=trajectory.episode_id,
                step_id=step_id,
                observation=step["observation"],
                local_history=local_history,
                memory_store=[],
                budget_remaining=20,
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
            "Labeled trajectory %s: %s samples, %s writes",
            trajectory.episode_id,
            len(sft_samples),
            sum(1 for sample in sft_samples if sample.target_memory.write),
        )

        return sft_samples

    def label_trajectories(
        self,
        trajectories: List[WebShopTrajectory],
        max_trajectories: Optional[int] = None,
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
        for idx, trajectory in enumerate(trajectories):
            logger.info(
                "Labeling trajectory %s/%s: %s",
                idx + 1,
                len(trajectories),
                trajectory.episode_id,
            )
            try:
                all_samples.extend(self.label_trajectory(trajectory))
            except Exception as exc:
                logger.error(
                    "Failed to label trajectory %s: %s",
                    trajectory.episode_id,
                    exc,
                )

        logger.info(
            "Labeled %s trajectories: %s total samples, %s writes",
            len(trajectories),
            len(all_samples),
            sum(1 for sample in all_samples if sample.target_memory.write),
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
        self.max_value_tokens = max_value_tokens
        self.max_key_tokens = max_key_tokens
        self.max_num_keys = max_num_keys
        logger.info("Initialized HeuristicLabeler")

    def _should_write_memory(self, observation: str) -> bool:
        keywords = ["$", "price", "features", "options", "details", "B0"]
        return any(keyword in observation for keyword in keywords)

    def _extract_keys(self, observation: str) -> List[str]:
        keys = []
        words = observation.split()

        for word in words:
            if word.startswith("B0") and len(word) > 5:
                keys.append(word)
                break

        for word in words:
            if "$" in word:
                keys.append(f"price {word}")
                break

        for word in ["features", "options", "size", "color", "brand"]:
            if word in observation.lower():
                keys.append(word)

        return keys[: self.max_num_keys]

    def _compress_value(self, observation: str) -> str:
        max_chars = self.max_value_tokens * 4
        return observation[:max_chars]

    def label_trajectory(self, trajectory: WebShopTrajectory) -> List[SFTSample]:
        steps = extract_observations_and_actions(trajectory)
        if not steps:
            return []

        sft_samples = []
        for step_id, step in enumerate(steps):
            observation = step["observation"]
            should_write = self._should_write_memory(observation)
            keys = self._extract_keys(observation) if should_write else []
            value = self._compress_value(observation) if should_write else ""

            memory_item = MemoryItem(
                write=should_write,
                keys=keys,
                value=value,
                timestamp=step["timestamp"],
                step_id=step_id,
                metadata={"method": "heuristic"},
            )

            local_history = extract_local_history(trajectory, step_id, history_length=3)
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
        all_samples = []
        for trajectory in trajectories:
            all_samples.extend(self.label_trajectory(trajectory))

        logger.info(
            "Labeled %s trajectories (heuristic): %s samples, %s writes",
            len(trajectories),
            len(all_samples),
            sum(1 for sample in all_samples if sample.target_memory.write),
        )
        return all_samples


class DemandAwareHindsightLabeler(HindsightLabeler):
    """Global hindsight labeler with demand analysis and retrieval verification."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        contribution_threshold: float = 0.3,
        max_value_tokens: int = 256,
        max_key_tokens: int = 64,
        max_num_keys: int = 4,
        base_url: Optional[str] = None,
        budget_ratio: float = 2.0,
        short_trajectory_threshold: int = 10,
        max_refinement_rounds: int = 5,
        retrieval_k: int = 3,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        strict: bool = False,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            contribution_threshold=contribution_threshold,
            max_value_tokens=max_value_tokens,
            max_key_tokens=max_key_tokens,
            max_num_keys=max_num_keys,
            base_url=base_url,
        )
        self.budget_ratio = budget_ratio
        self.short_trajectory_threshold = short_trajectory_threshold
        self.max_refinement_rounds = max_refinement_rounds
        self.retrieval_k = retrieval_k
        self.dense_model = dense_model
        self.strict = strict
        self.heuristic_labeler = HeuristicLabeler(
            max_value_tokens=max_value_tokens,
            max_key_tokens=max_key_tokens,
            max_num_keys=max_num_keys,
        )
        self._retriever = None
        self._retriever_init_attempted = False

        logger.info(
            "Initialized DemandAwareHindsightLabeler with budget_ratio=%s, short_threshold=%s, retrieval_k=%s, strict=%s",
            budget_ratio,
            short_trajectory_threshold,
            retrieval_k,
            strict,
        )

    def _get_retriever(self):
        """Lazily initialize the hybrid retriever."""
        if self._retriever_init_attempted:
            return self._retriever

        self._retriever_init_attempted = True
        try:
            from memory_constructor.models.retriever import HybridRetriever

            self._retriever = HybridRetriever(
                retrieval_k=self.retrieval_k,
                dense_model=self.dense_model,
            )
        except Exception as exc:
            if self.strict:
                raise RuntimeError(
                    f"[STRICT] HybridRetriever initialization failed: {exc}"
                ) from exc
            logger.warning(
                "HybridRetriever unavailable, retrieval verification will be skipped: %s",
                exc,
            )
            self._retriever = None

        return self._retriever

    def _summarize_observation(self, observation: str, max_chars: int = 300) -> str:
        text = re.sub(r"\s+", " ", observation or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _summarize_step(self, step: Dict[str, Any], max_obs_chars: int = 300) -> str:
        observation = self._summarize_observation(step.get("observation", ""), max_obs_chars)
        action = (step.get("action") or "").strip()
        return (
            f"Step {step['step_id']}: Observation: {observation}\n"
            f"Action: {action or '(none)'}"
        )

    def _summarize_trajectory(
        self,
        steps: List[Dict[str, Any]],
        max_obs_chars: int = 300,
    ) -> str:
        return "\n\n".join(
            self._summarize_step(step, max_obs_chars=max_obs_chars) for step in steps
        )

    def _normalize_task_requirements(
        self,
        task_requirements: Any,
        fallback: List[str],
    ) -> List[str]:
        normalized = []
        if isinstance(task_requirements, str):
            normalized = [task_requirements]
        elif isinstance(task_requirements, list):
            normalized = [str(item) for item in task_requirements]
        elif isinstance(task_requirements, dict):
            normalized = [
                f"{key}: {value}"
                for key, value in task_requirements.items()
                if value not in (None, "", [])
            ]

        cleaned = []
        seen = set()
        for item in normalized:
            item = re.sub(r"\s+", " ", item).strip(" -")
            if not item or item in seen:
                continue
            seen.add(item)
            cleaned.append(item)

        return cleaned or fallback

    def _extract_task_requirements_heuristic(self, instruction: str) -> List[str]:
        text = re.sub(r"\s+", " ", instruction or "").strip()
        if not text:
            return []

        parts = re.split(r",| and | with | but ", text)
        requirements = []
        seen = set()
        for part in parts:
            part = part.strip(" .")
            lowered = part.lower()
            if len(part) < 4 or lowered in seen:
                continue
            seen.add(lowered)
            requirements.append(part)

        budget_match = re.search(r"(under|below|less than)\s+\$?\d+(?:\.\d+)?", text, re.I)
        if budget_match:
            budget_requirement = budget_match.group(0)
            if budget_requirement.lower() not in seen:
                requirements.append(budget_requirement)

        return requirements[:6] or [text]

    def _parse_demands(
        self,
        raw_demands: Any,
        total_steps: int,
        expected_future_step: Optional[int] = None,
    ) -> Dict[int, Dict[int, str]]:
        parsed = defaultdict(dict)
        if not isinstance(raw_demands, list):
            return parsed

        for item in raw_demands:
            if not isinstance(item, dict):
                continue

            try:
                source_step = int(item.get("source_step"))
                future_step = int(item.get("future_step"))
            except (TypeError, ValueError):
                continue

            if not (0 <= source_step < total_steps and 0 <= future_step < total_steps):
                continue
            if source_step >= future_step:
                continue
            if expected_future_step is not None and future_step != expected_future_step:
                continue

            demand_text = str(
                item.get("demand")
                or item.get("description")
                or item.get("need")
                or ""
            ).strip()
            if not demand_text:
                demand_text = "Relevant product details needed for later decision-making."

            parsed[source_step][future_step] = demand_text

        return parsed

    def _is_memory_worthy_step(self, observation: str) -> bool:
        lowered = observation.lower()
        signal_terms = [
            "$",
            "price",
            "rating",
            "feature",
            "features",
            "option",
            "options",
            "color",
            "battery",
            "capacity",
            "inch",
            "review",
            "brand",
            "buy now",
            "search results",
            "comparison",
        ]
        return any(term in lowered for term in signal_terms) or bool(
            re.search(r"\bB0[A-Z0-9]{5,}\b", observation)
        )

    def _is_demanding_step(self, step: Dict[str, Any]) -> bool:
        observation = (step.get("observation") or "").lower()
        action = (step.get("action") or "").lower()
        demand_terms = [
            "buy now",
            "add to cart",
            "checkout",
            "comparison",
            "cart",
            "option",
            "options",
            "price",
            "rating",
            "filter",
            "sort",
        ]
        return any(term in observation or term in action for term in demand_terms)

    def _describe_source_signal(self, observation: str) -> str:
        lowered = observation.lower()
        aspects = []
        if "$" in observation or "price" in lowered:
            aspects.append("price")
        if "rating" in lowered or "review" in lowered:
            aspects.append("quality signals")
        if "feature" in lowered or "option" in lowered or "color" in lowered:
            aspects.append("product attributes")
        if "search results" in lowered or "comparison" in lowered:
            aspects.append("candidate comparison")
        if not aspects:
            aspects.append("product details")
        return f"{', '.join(aspects)} needed for a later decision or purchase step."

    def _fallback_demand_analysis(
        self,
        steps: List[Dict[str, Any]],
    ) -> Dict[int, Dict[int, str]]:
        demand_dict = defaultdict(dict)

        for source_step, step in enumerate(steps[:-1]):
            observation = step.get("observation", "")
            if not self._is_memory_worthy_step(observation):
                continue

            demand_text = self._describe_source_signal(observation)
            demanding_steps = [
                future_step
                for future_step in range(source_step + 1, len(steps))
                if self._is_demanding_step(steps[future_step])
            ]

            if not demanding_steps and source_step + 1 < len(steps):
                demanding_steps = [source_step + 1]

            for future_step in demanding_steps:
                demand_dict[source_step][future_step] = demand_text

        return demand_dict

    def _analyze_demands(
        self,
        trajectory: WebShopTrajectory,
        steps: List[Dict[str, Any]],
    ) -> Tuple[Dict[int, Dict[int, str]], List[str], Dict[str, Any]]:
        fallback_requirements = self._extract_task_requirements_heuristic(
            trajectory.instruction
        )
        demand_dict = defaultdict(dict)
        metadata = {
            "strategy": "short" if len(steps) <= self.short_trajectory_threshold else "chunked"
        }
        system_prompt = (
            "You analyze successful shopping trajectories. "
            "Identify only non-trivial information dependencies where an earlier "
            "step contains information genuinely needed by a later step."
        )

        if len(steps) <= self.short_trajectory_threshold:
            prompt = f"""Task instruction:
{trajectory.instruction}

Trajectory summary:
{self._summarize_trajectory(steps)}

Output JSON with:
{{
  "task_requirements": ["requirement 1", "..."],
  "demands": [
    {{
      "source_step": 1,
      "future_step": 4,
      "demand": "what later step needs from the earlier step"
    }}
  ]
}}

Rules:
- Only include dependencies where source_step < future_step.
- Ignore trivial navigation, repeated observations, and actions that do not consume prior information.
- Keep demand text concrete and short."""
            try:
                result = self._call_json(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    max_tokens=800,
                    fallback={"task_requirements": fallback_requirements, "demands": []},
                )
                parsed = self._parse_demands(result.get("demands"), len(steps))
                for source_step, future_map in parsed.items():
                    demand_dict[source_step].update(future_map)
                task_requirements = self._normalize_task_requirements(
                    result.get("task_requirements"),
                    fallback_requirements,
                )
                metadata["used_fallback"] = False
                return dict(demand_dict), task_requirements, metadata
            except Exception as exc:
                if self.strict:
                    raise RuntimeError(
                        f"[STRICT] Demand analysis LLM call failed for trajectory "
                        f"{trajectory.episode_id}: {exc}"
                    ) from exc
                logger.warning(
                    "Demand analysis failed for trajectory %s, using heuristic fallback: %s",
                    trajectory.episode_id,
                    exc,
                )

        task_requirements = fallback_requirements
        llm_success = False
        for future_step in range(len(steps) - 1, 0, -1):
            previous_steps = steps[:future_step]
            prompt = f"""Task instruction:
{trajectory.instruction}

Current future step to analyze:
{self._summarize_step(steps[future_step])}

Previous steps:
{self._summarize_trajectory(previous_steps)}

Return JSON:
{{
  "task_requirements": ["requirement 1", "..."],
  "demands": [
    {{
      "source_step": 1,
      "future_step": {future_step},
      "demand": "what this future step needs from that earlier step"
    }}
  ]
}}

Rules:
- Only return dependencies for future_step {future_step}.
- Ignore trivial navigation or repeated information.
- Use empty demands if this step does not rely on prior information."""
            try:
                result = self._call_json(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    max_tokens=512,
                    fallback={"task_requirements": fallback_requirements, "demands": []},
                )
                parsed = self._parse_demands(
                    result.get("demands"),
                    len(steps),
                    expected_future_step=future_step,
                )
                for source_step, future_map in parsed.items():
                    demand_dict[source_step].update(future_map)
                if task_requirements == fallback_requirements:
                    task_requirements = self._normalize_task_requirements(
                        result.get("task_requirements"),
                        fallback_requirements,
                    )
                llm_success = True
            except Exception as exc:
                if self.strict:
                    raise RuntimeError(
                        f"[STRICT] Chunked demand analysis failed at future step {future_step} "
                        f"for {trajectory.episode_id}: {exc}"
                    ) from exc
                logger.debug(
                    "Chunked demand analysis failed at future step %s for %s: %s",
                    future_step,
                    trajectory.episode_id,
                    exc,
                )

        if not demand_dict:
            if self.strict:
                raise RuntimeError(
                    f"[STRICT] No LLM demands found for trajectory {trajectory.episode_id}, "
                    f"heuristic fallback is disabled in strict mode"
                )
            demand_dict = self._fallback_demand_analysis(steps)
            metadata["used_fallback"] = True
        else:
            metadata["used_fallback"] = not llm_success

        return dict(demand_dict), task_requirements, metadata

    def _format_demand_summary(
        self,
        demand_dict: Dict[int, Dict[int, str]],
    ) -> str:
        if not demand_dict:
            return "(No future demands identified)"

        lines = []
        for source_step in sorted(demand_dict):
            demand_chunks = [
                f"future step {future_step}: {demand_text}"
                for future_step, demand_text in sorted(demand_dict[source_step].items())
            ]
            lines.append(f"Source step {source_step}: " + " | ".join(demand_chunks))
        return "\n".join(lines)

    def _normalize_memory_plan(
        self,
        plan_items: Any,
        demand_dict: Dict[int, Dict[int, str]],
        total_steps: int,
    ) -> List[Dict[str, Any]]:
        if not isinstance(plan_items, list):
            return []

        normalized = {}
        for item in plan_items:
            if not isinstance(item, dict):
                continue

            try:
                step_id = int(item.get("step_id"))
            except (TypeError, ValueError):
                continue

            if step_id not in demand_dict or not (0 <= step_id < total_steps):
                continue

            validated = validate_memory_item(
                {
                    "write": True,
                    "keys": item.get("keys", []),
                    "value": item.get("value", ""),
                },
                max_key_tokens=self.max_key_tokens,
                max_value_tokens=self.max_value_tokens,
                max_num_keys=self.max_num_keys,
            )
            if not validated["keys"] or not validated["value"]:
                continue

            demanded_by = []
            for future_step in item.get("demanded_by", list(demand_dict[step_id].keys())):
                try:
                    future_step = int(future_step)
                except (TypeError, ValueError):
                    continue
                if step_id < future_step < total_steps:
                    demanded_by.append(future_step)

            demanded_by = sorted(set(demanded_by)) or sorted(demand_dict[step_id].keys())
            if not demanded_by:
                continue

            candidate = {
                "step_id": step_id,
                "keys": validated["keys"],
                "value": validated["value"],
                "demanded_by": demanded_by,
                "reasoning": str(item.get("reasoning") or item.get("why") or "").strip(),
                "source_demands": {
                    future_step: demand_dict[step_id][future_step]
                    for future_step in demanded_by
                    if future_step in demand_dict[step_id]
                },
            }

            existing = normalized.get(step_id)
            if existing is None or len(candidate["demanded_by"]) > len(existing["demanded_by"]):
                normalized[step_id] = candidate

        return [normalized[step_id] for step_id in sorted(normalized)]

    def _apply_budget(
        self,
        memory_plan: List[Dict[str, Any]],
        budget: int,
    ) -> List[Dict[str, Any]]:
        if budget < 0:
            budget = 0
        if len(memory_plan) <= budget:
            return memory_plan

        ranked = sorted(
            memory_plan,
            key=lambda item: (len(item.get("demanded_by", [])), -item["step_id"]),
            reverse=True,
        )
        clipped = ranked[:budget]
        clipped.sort(key=lambda item: item["step_id"])
        return clipped

    def _fallback_memory_plan(
        self,
        trajectory: WebShopTrajectory,
        steps: List[Dict[str, Any]],
        demand_dict: Dict[int, Dict[int, str]],
    ) -> List[Dict[str, Any]]:
        memory_plan = []

        for source_step in sorted(demand_dict):
            label = super().label_step(trajectory, source_step, steps)
            if not label["should_write"] or not label["keys"] or not label["value"]:
                continue

            memory_plan.append(
                {
                    "step_id": source_step,
                    "keys": label["keys"],
                    "value": label["value"],
                    "demanded_by": sorted(demand_dict[source_step]),
                    "reasoning": label["reasoning"],
                    "source_demands": dict(demand_dict[source_step]),
                }
            )

        if memory_plan:
            return memory_plan

        heuristic_samples = self.heuristic_labeler.label_trajectory(trajectory)
        for sample in heuristic_samples:
            if sample.step_id not in demand_dict or not sample.target_memory.write:
                continue

            memory_plan.append(
                {
                    "step_id": sample.step_id,
                    "keys": sample.target_memory.keys,
                    "value": sample.target_memory.value,
                    "demanded_by": sorted(demand_dict[sample.step_id]),
                    "reasoning": sample.hindsight_label,
                    "source_demands": dict(demand_dict[sample.step_id]),
                }
            )

        return memory_plan

    def _generate_memory_plan(
        self,
        trajectory: WebShopTrajectory,
        steps: List[Dict[str, Any]],
        demand_dict: Dict[int, Dict[int, str]],
    ) -> List[Dict[str, Any]]:
        if not demand_dict:
            return []

        budget = max(0, int(len(steps) * self.budget_ratio))
        prompt = f"""Task instruction:
{trajectory.instruction}

Trajectory summary:
{self._summarize_trajectory(steps)}

Demand analysis:
{self._format_demand_summary(demand_dict)}

Create a minimal memory plan that satisfies these demands.

Constraints:
- At most 1 memory per step.
- Do not generate memories for steps with no demand.
- Merge multiple future demands for the same source step into one memory.
- Prefer fewer total memories when overlapping demands can be covered together.
- Keys must be retrieval-aware: include words likely to overlap with the later observations that need this memory.
- Each memory may use at most {self.max_num_keys} keys.
- Keep value concise and faithful, max about {self.max_value_tokens} tokens.
- Episode memory budget: {budget}

Respond in JSON:
{{
  "memories": [
    {{
      "step_id": 1,
      "keys": ["key 1", "key 2"],
      "value": "compressed memory value",
      "demanded_by": [3, 4],
      "reasoning": "why this memory covers those demands"
    }}
  ]
}}"""

        try:
            result = self._call_json(
                system_prompt=(
                    "You generate memory plans for a shopping agent. "
                    "Optimize for minimal total memories while preserving future usefulness."
                ),
                user_prompt=prompt,
                max_tokens=1200,
                fallback={"memories": []},
            )
            memory_plan = self._normalize_memory_plan(
                result.get("memories"),
                demand_dict,
                len(steps),
            )
        except Exception as exc:
            if self.strict:
                raise RuntimeError(
                    f"[STRICT] Memory plan generation LLM call failed for trajectory "
                    f"{trajectory.episode_id}: {exc}"
                ) from exc
            logger.warning(
                "Memory plan generation failed for trajectory %s, using fallback: %s",
                trajectory.episode_id,
                exc,
            )
            memory_plan = self._fallback_memory_plan(trajectory, steps, demand_dict)

        if not memory_plan:
            if self.strict:
                raise RuntimeError(
                    f"[STRICT] Memory plan is empty for trajectory {trajectory.episode_id}, "
                    f"fallback is disabled in strict mode"
                )
            memory_plan = self._fallback_memory_plan(trajectory, steps, demand_dict)

        return self._apply_budget(memory_plan, budget)

    def _build_store_before_step(
        self,
        memory_plan: List[Dict[str, Any]],
        future_step: int,
    ) -> MemoryStore:
        store = MemoryStore()
        for item in sorted(memory_plan, key=lambda memory: memory["step_id"]):
            if item["step_id"] >= future_step:
                break
            store.add(
                MemoryItem(
                    write=True,
                    keys=item["keys"],
                    value=item["value"],
                    timestamp=float(item.get("timestamp", item["step_id"])),
                    step_id=item["step_id"],
                    metadata=item.get("metadata", {}),
                )
            )
        return store

    def _collect_retrieval_failures(
        self,
        steps: List[Dict[str, Any]],
        memory_plan: List[Dict[str, Any]],
        demand_dict: Dict[int, Dict[int, str]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        retriever = self._get_retriever()
        if retriever is None:
            return {}

        failures = defaultdict(list)
        for item in memory_plan:
            source_step = item["step_id"]
            for future_step in item.get("demanded_by", []):
                query = steps[future_step]["observation"]
                store = self._build_store_before_step(memory_plan, future_step)

                if len(store) == 0:
                    failures[source_step].append(
                        {
                            "future_step": future_step,
                            "query": query,
                            "demand": demand_dict.get(source_step, {}).get(future_step, ""),
                            "reason": "target memory unavailable before demanding step",
                            "retrieved": [],
                        }
                    )
                    continue

                results = retriever.retrieve(
                    query,
                    store,
                    k=self.retrieval_k,
                    return_scores=True,
                )
                hit = any(memory.step_id == source_step for memory, _ in results)
                if hit:
                    continue

                failures[source_step].append(
                    {
                        "future_step": future_step,
                        "query": query,
                        "demand": demand_dict.get(source_step, {}).get(future_step, ""),
                        "reason": "target memory missed top-k retrieval",
                        "retrieved": [
                            {
                                "step_id": memory.step_id,
                                "keys": memory.keys,
                                "score": score,
                            }
                            for memory, score in results
                        ],
                    }
                )

        return failures

    def _heuristic_refine_keys(
        self,
        current_keys: List[str],
        memory_value: str,
        failures: List[Dict[str, Any]],
    ) -> List[str]:
        stopwords = {
            "the", "and", "with", "that", "this", "from", "into", "then", "than",
            "have", "will", "your", "showing", "search", "results", "page",
            "click", "buy", "cart", "item", "added", "price", "rating",
        }
        candidate_tokens = []
        for failure in failures:
            tokens = re.findall(r"[A-Za-z0-9$.-]+", failure.get("query", "").lower())
            for token in tokens:
                if len(token) < 3 or token in stopwords:
                    continue
                candidate_tokens.append(token)

        merged_keys = list(current_keys)
        for token in candidate_tokens:
            if any(token in key.lower() for key in merged_keys):
                continue
            merged_keys.append(token)
            if len(merged_keys) >= self.max_num_keys:
                break

        validated = validate_memory_item(
            {"write": True, "keys": merged_keys, "value": memory_value},
            max_key_tokens=self.max_key_tokens,
            max_value_tokens=self.max_value_tokens,
            max_num_keys=self.max_num_keys,
        )
        return validated["keys"] or current_keys

    def _refine_keys(
        self,
        plan_item: Dict[str, Any],
        failures: List[Dict[str, Any]],
    ) -> List[str]:
        failure_text = []
        for failure in failures:
            failure_text.append(
                f"- Future step {failure['future_step']}: query={self._summarize_observation(failure['query'], 220)} | "
                f"demand={failure.get('demand', '')} | top results={failure.get('retrieved', [])}"
            )

        prompt = f"""You are refining memory retrieval keys for a shopping agent.

Memory value:
{plan_item['value']}

Current keys:
{plan_item['keys']}

Retriever details:
- BM25 tokenization lowercases text and splits on whitespace.
- Dense retrieval embeds keys + value and compares cosine similarity.
- The goal is for later observation queries to retrieve this memory in the top-{self.retrieval_k}.

Failed retrieval cases:
{chr(10).join(failure_text)}

Requirements:
- Keep keys faithful to the memory value.
- Prefer concise, high-overlap phrases that will appear in the future queries.
- Use at most {self.max_num_keys} keys.

Respond in JSON:
{{
  "keys": ["refined key 1", "refined key 2"],
  "reasoning": "short explanation"
}}"""

        try:
            result = self._call_json(
                system_prompt=(
                    "You refine retrieval keys so a hybrid BM25 + dense retriever "
                    "can recover the intended memory."
                ),
                user_prompt=prompt,
                max_tokens=400,
                fallback={"keys": plan_item["keys"], "reasoning": ""},
            )
            validated = validate_memory_item(
                {
                    "write": True,
                    "keys": result.get("keys", plan_item["keys"]),
                    "value": plan_item["value"],
                },
                max_key_tokens=self.max_key_tokens,
                max_value_tokens=self.max_value_tokens,
                max_num_keys=self.max_num_keys,
            )
            if validated["keys"]:
                return validated["keys"]
        except Exception as exc:
            if self.strict:
                raise RuntimeError(
                    f"[STRICT] Key refinement LLM call failed for step {plan_item['step_id']}: {exc}"
                ) from exc
            logger.debug("Key refinement failed for step %s: %s", plan_item["step_id"], exc)

        return self._heuristic_refine_keys(plan_item["keys"], plan_item["value"], failures)

    def _verify_and_refine_plan(
        self,
        steps: List[Dict[str, Any]],
        memory_plan: List[Dict[str, Any]],
        demand_dict: Dict[int, Dict[int, str]],
    ) -> List[Dict[str, Any]]:
        retriever = self._get_retriever()
        if retriever is None or not memory_plan:
            return memory_plan

        plan_by_step = {
            item["step_id"]: copy.deepcopy(item) for item in memory_plan
        }
        last_failures = {}

        for attempt in range(1, self.max_refinement_rounds + 1):
            current_plan = [plan_by_step[step_id] for step_id in sorted(plan_by_step)]
            failures = self._collect_retrieval_failures(steps, current_plan, demand_dict)
            last_failures = failures
            if not failures:
                break

            changed = False
            for source_step, miss_list in failures.items():
                refined_keys = self._refine_keys(plan_by_step[source_step], miss_list)
                if refined_keys != plan_by_step[source_step]["keys"]:
                    plan_by_step[source_step]["keys"] = refined_keys
                    changed = True

            if not changed:
                logger.warning(
                    "Retrieval refinement stalled after attempt %s with unresolved misses.",
                    attempt,
                )
                break

        final_plan = [plan_by_step[step_id] for step_id in sorted(plan_by_step)]
        if last_failures:
            for source_step, miss_list in last_failures.items():
                final_plan_entry = plan_by_step[source_step]
                final_plan_entry.setdefault("verification", {})
                final_plan_entry["verification"]["unresolved_failures"] = miss_list
                logger.warning(
                    "Memory step %s still misses retrieval for %s demanding steps.",
                    source_step,
                    len(miss_list),
                )

        return final_plan

    def _retrieve_context(
        self,
        observation: str,
        memory_store: List[MemoryItem],
    ) -> List[Dict[str, Any]]:
        retriever = self._get_retriever()
        if retriever is None or not memory_store:
            return []

        store = MemoryStore()
        for memory in memory_store:
            store.add(copy.deepcopy(memory))

        try:
            results = retriever.retrieve(
                observation,
                store,
                k=self.retrieval_k,
                return_scores=True,
            )
        except Exception as exc:
            logger.debug("Retrieval context lookup failed: %s", exc)
            return []

        context = []
        for memory, score in results:
            context.append(
                {
                    "step_id": memory.step_id,
                    "keys": memory.keys,
                    "value_preview": self._summarize_observation(memory.value, 160),
                    "score": round(float(score), 4),
                }
            )
        return context

    def _assemble_sft_samples(
        self,
        trajectory: WebShopTrajectory,
        steps: List[Dict[str, Any]],
        memory_plan: List[Dict[str, Any]],
        task_requirements: List[str],
        demand_metadata: Dict[str, Any],
    ) -> List[SFTSample]:
        plan_by_step = {item["step_id"]: item for item in memory_plan}
        accumulated_memories: List[MemoryItem] = []
        memory_budget = max(0, int(len(steps) * self.budget_ratio))
        samples = []

        for step_id, step in enumerate(steps):
            local_history = extract_local_history(trajectory, step_id, history_length=3)
            current_store = [copy.deepcopy(memory) for memory in accumulated_memories]
            retrieval_context = self._retrieve_context(step["observation"], current_store)
            plan_item = plan_by_step.get(step_id)

            if plan_item is not None:
                contribution_score = len(plan_item.get("demanded_by", [])) / max(
                    1, len(steps) - step_id - 1
                )
                target_memory = MemoryItem(
                    write=True,
                    keys=plan_item["keys"],
                    value=plan_item["value"],
                    timestamp=step["timestamp"],
                    step_id=step_id,
                    metadata={
                        "method": "demand_aware",
                        "reasoning": plan_item.get("reasoning", ""),
                        "demanded_by": plan_item.get("demanded_by", []),
                        "source_demands": plan_item.get("source_demands", {}),
                        "verification": plan_item.get("verification", {}),
                    },
                )
                hindsight_label = plan_item.get("reasoning", "demand_aware")
            else:
                contribution_score = 0.0
                target_memory = MemoryItem(
                    write=False,
                    keys=[],
                    value="",
                    timestamp=step["timestamp"],
                    step_id=step_id,
                    metadata={"method": "demand_aware"},
                )
                hindsight_label = "no_future_demand"

            sample = SFTSample(
                sample_id=f"{trajectory.episode_id}_step{step_id}",
                trajectory_id=trajectory.episode_id,
                step_id=step_id,
                observation=step["observation"],
                local_history=local_history,
                memory_store=current_store,
                budget_remaining=max(0, memory_budget - len(current_store)),
                episode_progress=step_id / max(1, len(steps)),
                target_memory=target_memory,
                future_task=steps[step_id + 1]["observation"] if step_id + 1 < len(steps) else "",
                hindsight_label=hindsight_label,
                contribution_score=min(1.0, contribution_score),
                metadata={
                    "instruction": trajectory.instruction,
                    "final_success": trajectory.is_success,
                    "task_requirements": task_requirements,
                    "retrieval_context": retrieval_context,
                    "memory_budget": memory_budget,
                    "labeling_method": "demand_aware",
                    "demand_metadata": demand_metadata,
                },
            )
            samples.append(sample)

            if target_memory.write:
                accumulated_memories.append(copy.deepcopy(target_memory))

        return samples

    def label_trajectory(self, trajectory: WebShopTrajectory) -> List[SFTSample]:
        steps = extract_observations_and_actions(trajectory)
        if not steps:
            logger.warning("No steps found in trajectory %s", trajectory.episode_id)
            return []

        try:
            demand_dict, task_requirements, analysis_metadata = self._analyze_demands(
                trajectory,
                steps,
            )
            memory_plan = self._generate_memory_plan(trajectory, steps, demand_dict)
            memory_plan = self._verify_and_refine_plan(steps, memory_plan, demand_dict)

            samples = self._assemble_sft_samples(
                trajectory=trajectory,
                steps=steps,
                memory_plan=memory_plan,
                task_requirements=task_requirements,
                demand_metadata={
                    "analysis": analysis_metadata,
                    "demand_dict": {
                        str(source_step): {
                            str(future_step): demand_text
                            for future_step, demand_text in future_map.items()
                        }
                        for source_step, future_map in demand_dict.items()
                    },
                },
            )

            logger.info(
                "Demand-aware labeled trajectory %s: %s samples, %s writes",
                trajectory.episode_id,
                len(samples),
                sum(1 for sample in samples if sample.target_memory.write),
            )
            return samples
        except Exception as exc:
            if self.strict:
                raise RuntimeError(
                    f"[STRICT] Demand-aware labeling failed for trajectory "
                    f"{trajectory.episode_id}: {exc}\n{traceback.format_exc()}"
                ) from exc
            logger.error(
                "Demand-aware labeling failed for trajectory %s: %s\n%s",
                trajectory.episode_id,
                exc,
                traceback.format_exc(),
            )
            return self.heuristic_labeler.label_trajectory(trajectory)

    def label_trajectories(
        self,
        trajectories: List[WebShopTrajectory],
        max_trajectories: Optional[int] = None,
    ) -> List[SFTSample]:
        if max_trajectories is not None:
            trajectories = trajectories[:max_trajectories]

        all_samples = []
        for idx, trajectory in enumerate(trajectories):
            logger.info(
                "Demand-aware labeling trajectory %s/%s: %s",
                idx + 1,
                len(trajectories),
                trajectory.episode_id,
            )
            try:
                all_samples.extend(self.label_trajectory(trajectory))
            except Exception as exc:
                logger.error(
                    "Failed to demand-aware label trajectory %s: %s",
                    trajectory.episode_id,
                    exc,
                )

        logger.info(
            "Demand-aware labeled %s trajectories: %s samples, %s writes",
            len(trajectories),
            len(all_samples),
            sum(1 for sample in all_samples if sample.target_memory.write),
        )
        return all_samples
