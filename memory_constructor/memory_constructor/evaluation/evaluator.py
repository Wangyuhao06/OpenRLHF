"""
Evaluator for memory constructor models.

This module provides evaluation logic for running episodes and collecting metrics.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from memory_constructor.models.retriever import HybridRetriever
from memory_constructor.models.agent import GPT5Agent
from memory_constructor.data.memory_store import MemoryStore
from memory_constructor.evaluation.metrics import MetricsCalculator, EvaluationMetrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for memory constructor models."""

    def __init__(
        self,
        model_path: str,
        retriever: Optional[HybridRetriever] = None,
        agent: Optional[GPT5Agent] = None,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model checkpoint
            retriever: Retriever for memory retrieval (default: HybridRetriever)
            agent: Downstream agent (default: GPT5Agent)
            device: Device to run model on
        """
        self.device = device

        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.model.eval()

        # Initialize retriever
        self.retriever = retriever or HybridRetriever()

        # Initialize agent (optional, for full episode evaluation)
        self.agent = agent

        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()

    def evaluate_episodes(
        self,
        episodes: List[Dict[str, Any]],
        max_episodes: Optional[int] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate on a list of episodes.

        Args:
            episodes: List of episode data
            max_episodes: Maximum number of episodes to evaluate

        Returns:
            EvaluationMetrics object
        """
        if max_episodes:
            episodes = episodes[:max_episodes]

        logger.info(f"Evaluating on {len(episodes)} episodes...")

        results = []
        for episode in tqdm(episodes, desc="Evaluating episodes"):
            result = self._evaluate_episode(episode)
            results.append(result)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate(results)

        return metrics

    def _evaluate_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single episode.

        Args:
            episode: Episode data containing trajectory

        Returns:
            Episode result with metrics
        """
        # Initialize memory store
        memory_store = MemoryStore()

        # Track episode data
        memories_written = []
        memories_retrieved = []
        computational_cost = 0.0

        # Run through trajectory
        trajectory = episode["trajectory"]
        for step_idx, step in enumerate(trajectory):
            observation = step["observation"]
            local_history = step.get("local_history", [])

            # Generate memory constructor action
            constructor_action = self._generate_constructor_action(
                observation=observation,
                local_history=local_history,
                memory_store=memory_store,
            )

            # Update memory store
            if constructor_action["write"]:
                raw_keys = constructor_action["keys"]
                flat_keys = []
                for k in raw_keys:
                    if isinstance(k, list):
                        flat_keys.extend(str(x) for x in k)
                    else:
                        flat_keys.append(str(k))
                memory_item = {
                    "write": True,
                    "keys": flat_keys,
                    "value": str(constructor_action["value"]),
                    "timestamp": float(step_idx),
                    "step_id": step_idx,
                    "metadata": {},
                }
                memory_store.add(memory_item)
                memories_written.append(memory_item)

                # Computational cost for writing
                computational_cost += 0.05

            # Retrieve memories (if agent is available)
            if self.agent:
                query = self.agent.generate_query(observation, local_history)
                retrieved = self.retriever.retrieve(query, memory_store, k=3)
                memories_retrieved.append(retrieved)

                # Computational cost for retrieval
                computational_cost += 0.01 * len(retrieved)

        # Calculate task reward (from episode data)
        task_reward = episode.get("reward", 0.0)
        task_success = episode.get("success", False)

        # Get ground truth memories (if available)
        ground_truth_memories = episode.get("ground_truth_memories", [])

        return {
            "task_reward": task_reward,
            "task_success": task_success,
            "memories_written": memories_written,
            "memories_retrieved": memories_retrieved,
            "ground_truth_memories": ground_truth_memories,
            "computational_cost": computational_cost,
        }

    def _generate_constructor_action(
        self,
        observation: str,
        local_history: List[str],
        memory_store: MemoryStore,
    ) -> Dict[str, Any]:
        """
        Generate memory constructor action.

        Args:
            observation: Current observation
            local_history: Recent action-observation history
            memory_store: Current memory store

        Returns:
            Constructor action dict with keys: write, keys, value
        """
        # Format prompt
        prompt = self._format_prompt(observation, local_history, memory_store)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,  # Greedy decoding for evaluation
                do_sample=False,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        # Parse JSON
        try:
            action = json.loads(generated_text.strip())
            return {
                "write": action.get("write", False),
                "keys": action.get("keys", []),
                "value": action.get("value", ""),
            }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse action: {generated_text}")
            return {"write": False, "keys": [], "value": ""}

    def _format_prompt(
        self,
        observation: str,
        local_history: List[str],
        memory_store: MemoryStore,
    ) -> str:
        """Format prompt for memory constructor."""
        prompt = "You are a memory constructor. Decide whether to write a memory.\n\n"

        # Current observation
        prompt += f"Current observation:\n{observation}\n\n"

        # Local history
        if local_history:
            prompt += "Recent history:\n"
            for item in local_history[-3:]:
                prompt += f"- {item}\n"
            prompt += "\n"

        # Current memories
        if memory_store.memories:
            prompt += "Current memories:\n"
            for mem in memory_store.memories[-5:]:
                from memory_constructor.data.schemas import MemoryItem as _MI
                raw_keys = mem.keys if isinstance(mem, _MI) else mem["keys"]
                # Flatten any nested lists and convert to str
                flat_keys = []
                for k in raw_keys:
                    if isinstance(k, list):
                        flat_keys.extend(str(x) for x in k)
                    else:
                        flat_keys.append(str(k))
                keys_str = ", ".join(flat_keys)
                mem_value = mem.value if isinstance(mem, _MI) else mem["value"]
                prompt += f"- [{keys_str}]: {mem_value}\n"
            prompt += "\n"

        # Instruction
        prompt += (
            "Output JSON with: {\"write\": bool, \"keys\": [str], \"value\": str}\n"
        )

        return prompt

    def evaluate_from_file(
        self,
        data_path: str,
        max_episodes: Optional[int] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate from a JSONL file.

        Args:
            data_path: Path to JSONL file with episodes
            max_episodes: Maximum number of episodes to evaluate

        Returns:
            EvaluationMetrics object
        """
        # Load data — support both flat SFTSample JSONL and pre-grouped episode dicts
        raw_records = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        # If records are flat SFTSample format (have trajectory_id but no trajectory key),
        # group them into episode dicts keyed by trajectory_id
        if raw_records and "trajectory_id" in raw_records[0] and "trajectory" not in raw_records[0]:
            traj_map: Dict[str, List[Any]] = {}
            for rec in raw_records:
                tid = rec["trajectory_id"]
                if tid not in traj_map:
                    traj_map[tid] = []
                traj_map[tid].append(rec)
            episodes = []
            for tid, steps in traj_map.items():
                steps.sort(key=lambda s: s.get("step_id", 0))
                episodes.append({
                    "trajectory_id": tid,
                    "trajectory": steps,
                    "metadata": steps[0].get("metadata", {}),
                })
        else:
            episodes = raw_records

        return self.evaluate_episodes(episodes, max_episodes)

    def save_results(
        self,
        metrics: EvaluationMetrics,
        output_path: str,
    ):
        """
        Save evaluation results to file.

        Args:
            metrics: Evaluation metrics
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"Results saved to {output_path}")
