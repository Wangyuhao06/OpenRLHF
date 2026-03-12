"""
Data schemas for Memory Constructor project.

This module defines all dataclasses used throughout the project for:
- Raw WebShop trajectories
- Memory items
- Training samples (SFT, best-of-n, RL)
- Evaluation records
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json


@dataclass
class WebShopNode:
    """Single node in a WebShop trajectory."""
    node_type: str  # "OBS_TOOL", "THOUGHT", "ACT_TOOL"
    payload: Dict[str, Any]
    call_id: str
    context_tokens: int
    t: float  # Timestamp
    failure_flag: Optional[bool] = None
    failure_reason: Optional[str] = None
    ref_ids: Optional[List[str]] = None


@dataclass
class WebShopTrajectory:
    """Complete WebShop trajectory from trajectory generation repo."""
    episode_id: str
    task_id: str
    task_idx: int
    instruction: str
    steps: int
    final_reward: float
    is_success: bool
    is_partial: bool
    termination: str
    format_errors: int
    final_ctx_tokens: int
    trajectory: List[WebShopNode]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebShopTrajectory":
        """Create from raw JSON dict."""
        trajectory_nodes = [
            WebShopNode(**node) for node in data.get("trajectory", [])
        ]
        return cls(
            episode_id=data["episode_id"],
            task_id=data["task_id"],
            task_idx=data["task_idx"],
            instruction=data["instruction"],
            steps=data["steps"],
            final_reward=data["final_reward"],
            is_success=data["is_success"],
            is_partial=data["is_partial"],
            termination=data["termination"],
            format_errors=data["format_errors"],
            final_ctx_tokens=data["final_ctx_tokens"],
            trajectory=trajectory_nodes,
            metadata=data.get("metadata", {}),
        )


@dataclass
class MemoryItem:
    """Single memory item with temporal ordering."""
    write: bool
    keys: List[str]
    value: str
    timestamp: float
    step_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Create from dict."""
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class SFTSample:
    """Training sample for SFT stage."""
    sample_id: str
    trajectory_id: str
    step_id: int

    # Input context
    observation: str
    local_history: List[str]
    memory_store: List[MemoryItem]
    budget_remaining: int
    episode_progress: float

    # Target output
    target_memory: MemoryItem

    # Metadata
    future_task: str
    hindsight_label: str
    contribution_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        data = asdict(self)
        # Convert MemoryItem objects to dicts
        data["memory_store"] = [m.to_dict() if isinstance(m, MemoryItem) else m for m in self.memory_store]
        data["target_memory"] = self.target_memory.to_dict() if isinstance(self.target_memory, MemoryItem) else self.target_memory
        return data


@dataclass
class CandidateMemory:
    """Single candidate memory with scores."""
    candidate_id: int
    memory_item: MemoryItem

    # Scores (weighted merge of hindsight and counterfactual)
    hindsight_score: float
    counterfactual_score: float
    task_success_score: float
    retrieval_usefulness: float
    compactness_score: float
    redundancy_score: float
    faithfulness_score: float

    # Combined score
    total_score: float

    # Debugging info
    future_retrieval_hits: int
    future_impact_steps: List[int]
    sampled_perspectives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        data = asdict(self)
        data["memory_item"] = self.memory_item.to_dict() if isinstance(self.memory_item, MemoryItem) else self.memory_item
        return data


@dataclass
class CandidateMemoryList:
    """List of candidate memories for best-of-n selection."""
    sample_id: str
    trajectory_id: str
    step_id: int

    # Input context (same as SFT)
    observation: str
    local_history: List[str]
    memory_store: List[MemoryItem]
    budget_remaining: int
    episode_progress: float

    # Sampled candidates
    candidates: List[CandidateMemory]

    # Best candidate selection
    best_candidate_idx: int
    selection_method: str  # "hindsight", "counterfactual", "weighted_merge"

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        data = asdict(self)
        data["memory_store"] = [m.to_dict() if isinstance(m, MemoryItem) else m for m in self.memory_store]
        data["candidates"] = [c.to_dict() if isinstance(c, CandidateMemory) else c for c in self.candidates]
        return data


@dataclass
class RLStep:
    """Single step in RL rollout."""
    step_id: int
    timestamp: float

    # State
    observation: str
    local_history: List[str]
    memory_store: List[MemoryItem]
    budget_remaining: int

    # Constructor action
    constructor_output: MemoryItem
    constructor_log_prob: float

    # Agent query generation
    agent_query: str
    retrieved_memories: List[MemoryItem]
    retrieval_scores: List[float]

    # Agent action
    agent_action: str
    agent_log_prob: float

    # Reward
    step_reward: float
    auxiliary_rewards: Dict[str, float]

    # Value estimates
    value: float
    advantage: float


@dataclass
class RLRolloutSample:
    """Complete RL rollout sample."""
    rollout_id: str
    episode_id: str

    # Trajectory
    steps: List[RLStep]

    # Episode-level info
    episode_success: bool
    episode_reward: float
    total_memories_written: int
    budget_exhausted: bool

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeEvaluation:
    """Evaluation results for a single episode."""
    episode_id: str
    task: str
    success: bool
    total_reward: float

    # Memory statistics
    num_memories_written: int
    num_no_writes: int
    write_ratio: float

    # Retrieval statistics
    num_retrievals: int
    num_retrieval_hits: int
    retrieval_hit_rate: float
    avg_retrieval_score: float

    # Memory quality
    avg_num_keys_per_memory: float
    avg_value_length: float
    memory_redundancy: float
    temporal_distribution: Dict[str, float]  # early/mid/late

    # Detailed trace
    steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationRecord:
    """Complete evaluation record."""
    eval_id: str
    model_checkpoint: str
    dataset: str

    # Episode-level metrics
    episodes: List[EpisodeEvaluation]

    # Aggregate metrics
    task_success_rate: float
    avg_memories_written: float
    avg_retrieval_hit_rate: float
    avg_memory_redundancy: float
    avg_value_length: float
    budget_exhaustion_rate: float

    # Baseline comparisons
    baseline_comparisons: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return asdict(self)

    def save_json(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
