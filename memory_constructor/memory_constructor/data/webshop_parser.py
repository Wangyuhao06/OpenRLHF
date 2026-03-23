"""
WebShop trajectory parser.

This module parses raw WebShop trajectory files (JSONL format) into
structured dataclasses for training and evaluation.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from memory_constructor.data.schemas import WebShopTrajectory, WebShopNode

logger = logging.getLogger(__name__)


class WebShopParser:
    """Parser for WebShop trajectory files."""

    def __init__(self, use_only_successful: bool = True):
        """
        Initialize parser.

        Args:
            use_only_successful: If True, only parse successful trajectories
        """
        self.use_only_successful = use_only_successful
        logger.info(
            f"Initialized WebShopParser (use_only_successful={use_only_successful})"
        )

    def parse_file(self, file_path: str) -> List[WebShopTrajectory]:
        """
        Parse a single JSONL trajectory file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of WebShopTrajectory objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")

        trajectories = []
        num_skipped = 0

        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    trajectory = self._parse_trajectory(data)

                    # Filter by success if needed
                    if self.use_only_successful and not trajectory.is_success:
                        num_skipped += 1
                        continue

                    trajectories.append(trajectory)

                except Exception as e:
                    logger.warning(
                        f"Failed to parse line {line_num} in {file_path}: {e}"
                    )
                    continue

        logger.info(
            f"Parsed {len(trajectories)} trajectories from {file_path} "
            f"(skipped {num_skipped} unsuccessful)"
        )

        return trajectories

    def parse_files(self, file_paths: List[str]) -> List[WebShopTrajectory]:
        """
        Parse multiple trajectory files.

        Args:
            file_paths: List of paths to JSONL files

        Returns:
            List of WebShopTrajectory objects from all files
        """
        all_trajectories = []

        for file_path in file_paths:
            try:
                trajectories = self.parse_file(file_path)
                all_trajectories.extend(trajectories)
            except Exception as e:
                logger.error(f"Failed to parse file {file_path}: {e}")
                continue

        logger.info(f"Parsed {len(all_trajectories)} total trajectories")

        return all_trajectories

    def _parse_trajectory(self, data: Dict[str, Any]) -> WebShopTrajectory:
        """
        Parse a single trajectory from JSON data.

        Args:
            data: Dictionary from JSONL line

        Returns:
            WebShopTrajectory object
        """
        # Parse trajectory nodes
        nodes = []
        for node_data in data.get("trajectory", []):
            node = self._parse_node(node_data)
            nodes.append(node)

        # Create trajectory object
        trajectory = WebShopTrajectory(
            episode_id=data.get("episode_id", ""),
            task_id=data.get("task_id", ""),
            task_idx=data.get("task_idx", 0),
            instruction=data.get("instruction", ""),
            steps=data.get("steps", 0),
            final_reward=data.get("final_reward", 0.0),
            is_success=data.get("is_success", False),
            is_partial=data.get("is_partial", False),
            termination=data.get("termination", ""),
            format_errors=data.get("format_errors", 0),
            final_ctx_tokens=data.get("final_ctx_tokens", 0),
            trajectory=nodes,
            metadata=data.get("metadata", {}),
        )

        return trajectory

    def _parse_node(self, node_data: Dict[str, Any]) -> WebShopNode:
        """
        Parse a single trajectory node.

        Args:
            node_data: Dictionary for a single node

        Returns:
            WebShopNode object
        """
        node = WebShopNode(
            node_type=node_data.get("node_type", ""),
            payload=node_data.get("payload", {}),
            call_id=node_data.get("call_id", ""),
            context_tokens=node_data.get("context_tokens", 0),
            t=node_data.get("t", 0.0),
            failure_flag=node_data.get("failure_flag"),
            failure_reason=node_data.get("failure_reason"),
            ref_ids=node_data.get("ref_ids"),
        )

        return node

    def split_trajectories(
        self,
        trajectories: List[WebShopTrajectory],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[WebShopTrajectory], List[WebShopTrajectory], List[WebShopTrajectory]]:
        """
        Split trajectories into train/val/test sets.

        Args:
            trajectories: List of trajectories to split
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_trajectories, val_trajectories, test_trajectories)
        """
        import random

        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # Shuffle with seed
        random.seed(seed)
        shuffled = trajectories.copy()
        random.shuffle(shuffled)

        # Calculate split indices
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split
        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        logger.info(
            f"Split {n} trajectories into train={len(train)}, "
            f"val={len(val)}, test={len(test)}"
        )

        return train, val, test

    def save_trajectories(
        self, trajectories: List[WebShopTrajectory], output_path: str
    ):
        """
        Save trajectories to JSONL file.

        Args:
            trajectories: List of trajectories to save
            output_path: Path to output JSONL file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for traj in trajectories:
                # Convert to dict
                traj_dict = asdict(traj)
                # Write as JSON line
                f.write(json.dumps(traj_dict) + "\n")

        logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")

    def get_statistics(self, trajectories: List[WebShopTrajectory]) -> Dict[str, Any]:
        """
        Compute statistics for a list of trajectories.

        Args:
            trajectories: List of trajectories

        Returns:
            Dictionary with statistics
        """
        if not trajectories:
            return {}

        num_trajectories = len(trajectories)
        num_successful = sum(1 for t in trajectories if t.is_success)
        num_partial = sum(1 for t in trajectories if t.is_partial)

        avg_steps = sum(t.steps for t in trajectories) / num_trajectories
        avg_reward = sum(t.final_reward for t in trajectories) / num_trajectories
        avg_tokens = sum(t.final_ctx_tokens for t in trajectories) / num_trajectories

        # Count node types
        node_type_counts = {}
        for traj in trajectories:
            for node in traj.trajectory:
                node_type = node.node_type
                node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

        stats = {
            "num_trajectories": num_trajectories,
            "num_successful": num_successful,
            "num_partial": num_partial,
            "success_rate": num_successful / num_trajectories,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "avg_tokens": avg_tokens,
            "node_type_counts": node_type_counts,
        }

        return stats


def extract_observations_and_actions(
    trajectory: WebShopTrajectory,
) -> List[Dict[str, Any]]:
    """
    Extract observation-action pairs from a trajectory.

    This is useful for creating training samples where we need to know
    what the agent observed and what action it took at each step.

    Args:
        trajectory: WebShopTrajectory object

    Returns:
        List of dictionaries with:
            - step_id: Step number
            - observation: Observation text (from OBS_TOOL node)
            - thought: Thought text (from THOUGHT node, if present)
            - action: Action text (from ACT_TOOL node)
            - timestamp: Timestamp
            - done: Whether this is the final step
    """
    steps = []
    current_obs = None
    current_thought = None
    step_id = 0

    for i, node in enumerate(trajectory.trajectory):
        if node.node_type == "OBS_TOOL":
            # New observation
            current_obs = node.payload.get("raw_text") or node.payload.get("observation", "")

        elif node.node_type == "THOUGHT":
            # Thought (optional)
            current_thought = node.payload.get("raw_text") or node.payload.get("thought", "")

        elif node.node_type == "ACT_TOOL":
            # Action - create a step
            if current_obs is not None:
                action = node.payload.get("action_str") or node.payload.get("action", "")

                # Check if this is the last step
                done = (i == len(trajectory.trajectory) - 1)

                steps.append({
                    "step_id": step_id,
                    "observation": current_obs,
                    "thought": current_thought,
                    "action": action,
                    "timestamp": node.t,
                    "done": done,
                })

                step_id += 1

                # Reset thought for next step
                current_thought = None

    return steps


def extract_local_history(
    trajectory: WebShopTrajectory,
    current_step_id: int,
    history_length: int = 3,
) -> List[str]:
    """
    Extract local history (last N steps) for a given step.

    Args:
        trajectory: WebShopTrajectory object
        current_step_id: Current step ID
        history_length: Number of previous steps to include

    Returns:
        List of history strings (most recent last)
    """
    steps = extract_observations_and_actions(trajectory)

    # Get previous steps
    start_idx = max(0, current_step_id - history_length)
    history_steps = steps[start_idx:current_step_id]

    # Format as strings
    history = []
    for step in history_steps:
        history_str = f"Step {step['step_id']}: "
        if step['thought']:
            history_str += f"Thought: {step['thought']} | "
        history_str += f"Action: {step['action']} | Obs: {step['observation'][:100]}..."
        history.append(history_str)

    return history
