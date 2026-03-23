"""
Candidate scorer for best-of-n training.

This module scores candidate memories using a combination of:
1. Hindsight score (what was actually retrieved)
2. Counterfactual score (what would have been retrieved)
3. Other quality metrics (compactness, redundancy, faithfulness)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import numpy as np

from memory_constructor.data.schemas import (
    CandidateMemoryList,
    MemoryItem,
    WebShopTrajectory,
)
from memory_constructor.data.webshop_parser import WebShopParser
from memory_constructor.models.retriever import HybridRetriever
from memory_constructor.data.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class CandidateScorer:
    """
    Candidate scorer using counterfactual and hindsight evaluation.

    Scores candidates based on:
    - Hindsight: What was actually retrieved in the trajectory
    - Counterfactual: What would have been retrieved if this memory existed
    - Compactness: Token efficiency
    - Redundancy: Overlap with existing memories
    - Faithfulness: Grounding in observation
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        hindsight_weight: float = 0.5,
        counterfactual_weight: float = 0.5,
        retrieval_usefulness_weight: float = 0.4,
        task_success_weight: float = 0.3,
        compactness_weight: float = 0.1,
        redundancy_weight: float = 0.1,
        faithfulness_weight: float = 0.1,
    ):
        """
        Initialize candidate scorer.

        Args:
            retriever: HybridRetriever instance (creates default if None)
            hindsight_weight: Weight for hindsight score
            counterfactual_weight: Weight for counterfactual score
            retrieval_usefulness_weight: Weight for retrieval usefulness
            task_success_weight: Weight for task success
            compactness_weight: Weight for compactness
            redundancy_weight: Weight for redundancy
            faithfulness_weight: Weight for faithfulness
        """
        self.retriever = retriever or HybridRetriever()
        self.hindsight_weight = hindsight_weight
        self.counterfactual_weight = counterfactual_weight
        self.retrieval_usefulness_weight = retrieval_usefulness_weight
        self.task_success_weight = task_success_weight
        self.compactness_weight = compactness_weight
        self.redundancy_weight = redundancy_weight
        self.faithfulness_weight = faithfulness_weight

        logger.info(
            f"Initialized CandidateScorer with hindsight_weight={hindsight_weight}, "
            f"counterfactual_weight={counterfactual_weight}"
        )

    def _compute_counterfactual_score(
        self,
        candidate: Dict[str, Any],
        future_observations: List[str],
        existing_memories: List[MemoryItem],
    ) -> float:
        """
        Compute counterfactual retrieval score.

        Simulates future retrieval queries and checks if this candidate
        would have been retrieved.

        Args:
            candidate: Candidate memory
            future_observations: List of future observations
            existing_memories: Existing memories in store

        Returns:
            Counterfactual score (0-1)
        """
        if not candidate.memory_item.write:
            return 0.0

        # Create temporary memory store with existing + candidate
        temp_store = MemoryStore(max_capacity=100)

        # Add existing memories
        for mem in existing_memories:
            temp_store.append(mem)

        # Add candidate
        candidate_mem = MemoryItem(
            write=True,
            keys=candidate.memory_item.keys,
            value=candidate.memory_item.value,
            timestamp=candidate.memory_item.timestamp,
            step_id=candidate.memory_item.step_id,
            metadata={},
        )
        temp_store.append(candidate_mem)

        # Simulate retrieval for each future observation
        retrieval_scores = []

        for obs in future_observations[:10]:  # Limit to first 10 future steps
            # Use observation as query (simplified)
            query = obs[:200]  # Truncate

            # Retrieve
            results = self.retriever.retrieve(
                query, temp_store, k=3, return_scores=True
            )

            # Check if candidate was retrieved
            candidate_retrieved = False
            candidate_score = 0.0

            for mem, score in results:
                if mem.step_id == candidate_mem.step_id:
                    candidate_retrieved = True
                    candidate_score = score
                    break

            retrieval_scores.append(candidate_score)

        # Average retrieval score
        if retrieval_scores:
            return float(np.mean(retrieval_scores))
        else:
            return 0.0

    def _compute_compactness_score(
        self, candidate: Dict[str, Any], max_value_tokens: int = 256
    ) -> float:
        """
        Compute compactness score.

        Args:
            candidate: Candidate memory
            max_value_tokens: Maximum value tokens

        Returns:
            Compactness score (0-1)
        """
        if not candidate.memory_item.write:
            return 1.0  # No write = maximum compactness

        value_length = len(candidate.memory_item.value.split())
        compactness = 1.0 - (value_length / max_value_tokens)

        return max(0.0, compactness)

    def _compute_redundancy_score(
        self, candidate: Dict[str, Any], existing_memories: List[MemoryItem]
    ) -> float:
        """
        Compute redundancy score.

        Args:
            candidate: Candidate memory
            existing_memories: Existing memories

        Returns:
            Redundancy score (0-1, higher = less redundant)
        """
        if not candidate.memory_item.write:
            return 1.0  # No write = no redundancy

        if not existing_memories:
            return 1.0  # No existing memories = no redundancy

        # Simple token overlap check
        candidate_tokens = set(candidate.memory_item.value.lower().split())

        max_overlap = 0.0
        for mem in existing_memories:
            mem_tokens = set(mem.value.lower().split())

            if len(candidate_tokens) == 0:
                overlap = 0.0
            else:
                overlap = len(candidate_tokens & mem_tokens) / len(candidate_tokens)

            max_overlap = max(max_overlap, overlap)

        # Return 1 - max_overlap (higher score = less redundant)
        return 1.0 - max_overlap

    def _compute_faithfulness_score(
        self, candidate: Dict[str, Any], observation: str
    ) -> float:
        """
        Compute faithfulness score.

        Checks if memory value is grounded in observation.

        Args:
            candidate: Candidate memory
            observation: Original observation

        Returns:
            Faithfulness score (0-1)
        """
        if not candidate.memory_item.write:
            return 1.0  # No write = no hallucination risk

        # Simple token overlap check
        value_tokens = set(candidate.memory_item.value.lower().split())
        obs_tokens = set(observation.lower().split())

        if len(value_tokens) == 0:
            return 0.0

        # Compute overlap
        overlap = len(value_tokens & obs_tokens) / len(value_tokens)

        return overlap

    def score_candidate(
        self,
        candidate: Dict[str, Any],
        candidate_list: CandidateMemoryList,
        future_observations: List[str],
    ) -> Dict[str, Any]:
        """
        Score a single candidate.

        Args:
            candidate: Candidate to score
            candidate_list: Parent candidate list (for context)
            future_observations: Future observations for counterfactual

        Returns:
            Updated candidate with scores
        """
        # Compute individual scores
        counterfactual_score = self._compute_counterfactual_score(
            candidate, future_observations, candidate_list.memory_store
        )

        compactness_score = self._compute_compactness_score(candidate)

        redundancy_score = self._compute_redundancy_score(
            candidate, candidate_list.memory_store
        )

        faithfulness_score = self._compute_faithfulness_score(
            candidate, candidate_list.observation
        )

        # Hindsight score (placeholder - would need full trajectory)
        hindsight_score = 0.0  # TODO: Implement with full trajectory

        # Task success score (placeholder - expensive to compute)
        task_success_score = 0.5  # Neutral

        # Compute retrieval usefulness (weighted merge of hindsight and counterfactual)
        retrieval_usefulness = (
            self.hindsight_weight * hindsight_score
            + self.counterfactual_weight * counterfactual_score
        )

        # Compute total score
        total_score = (
            self.retrieval_usefulness_weight * retrieval_usefulness
            + self.task_success_weight * task_success_score
            + self.compactness_weight * compactness_score
            + self.redundancy_weight * redundancy_score
            + self.faithfulness_weight * faithfulness_score
        )

        # Update candidate
        candidate.hindsight_score = hindsight_score
        candidate.counterfactual_score = counterfactual_score
        candidate.task_success_score = task_success_score
        candidate.retrieval_usefulness = retrieval_usefulness
        candidate.compactness_score = compactness_score
        candidate.redundancy_score = redundancy_score
        candidate.faithfulness_score = faithfulness_score
        candidate.total_score = total_score

        return candidate

    def score_candidate_list(
        self,
        candidate_list: CandidateMemoryList,
        future_observations: List[str],
    ) -> CandidateMemoryList:
        """
        Score all candidates in a list and select best.

        Args:
            candidate_list: Candidate list to score
            future_observations: Future observations

        Returns:
            Updated candidate list with scores and best selection
        """
        # Score each candidate
        scored_candidates = []

        for candidate in candidate_list.candidates:
            scored_candidate = self.score_candidate(
                candidate, candidate_list, future_observations
            )
            scored_candidates.append(scored_candidate)

        # Select best candidate
        best_idx = max(
            range(len(scored_candidates)),
            key=lambda i: scored_candidates[i].total_score,
        )

        # Update candidate list
        candidate_list.candidates = scored_candidates
        candidate_list.best_candidate_idx = best_idx
        candidate_list.selection_method = "weighted_merge"

        return candidate_list

    def score_from_file(
        self,
        input_path: str,
        output_path: str,
        trajectory_file: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Score candidates from file and save results.

        Args:
            input_path: Path to input JSONL with CandidateMemoryList objects
            output_path: Path to output JSONL for scored candidates
            trajectory_file: Optional trajectory file for hindsight scoring
            max_samples: Maximum samples to process
        """
        # Load candidate lists
        logger.info(f"Loading candidate lists from {input_path}...")
        candidate_lists = []

        with open(input_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and len(candidate_lists) >= max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    candidate_list = CandidateMemoryList.from_dict(data)
                    candidate_lists.append(candidate_list)
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(candidate_lists)} candidate lists")

        # Score each list
        logger.info("Scoring candidates...")
        scored_lists = []

        for i, candidate_list in enumerate(candidate_lists):
            # Get future observations (simplified - use empty list)
            future_observations = []  # TODO: Load from trajectory file

            # Score
            scored_list = self.score_candidate_list(
                candidate_list, future_observations
            )
            scored_lists.append(scored_list)

            if (i + 1) % 100 == 0:
                logger.info(f"Scored {i + 1}/{len(candidate_lists)} lists")

        # Save results
        logger.info(f"Saving scored candidates to {output_path}...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for scored_list in scored_lists:
                # Convert to dict
                scored_dict = asdict(scored_list)
                # Write as JSON line
                f.write(json.dumps(scored_dict) + "\n")

        logger.info(f"Saved {len(scored_lists)} scored lists to {output_path}")

        # Print statistics
        best_scores = [sl.candidates[sl.best_candidate_idx].total_score for sl in scored_lists]
        logger.info(f"Best score statistics:")
        logger.info(f"  Mean: {np.mean(best_scores):.3f}")
        logger.info(f"  Std: {np.std(best_scores):.3f}")
        logger.info(f"  Min: {np.min(best_scores):.3f}")
        logger.info(f"  Max: {np.max(best_scores):.3f}")
