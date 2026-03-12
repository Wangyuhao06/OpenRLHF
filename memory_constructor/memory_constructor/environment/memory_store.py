"""
Append-only memory store for Memory Constructor.

This module implements the memory store that:
- Stores memory items in append-only fashion
- Tracks budget usage
- Provides serialization for checkpointing
- Supports temporal ordering
"""

from typing import List, Dict, Any, Optional
import json
from dataclasses import asdict

from ..data.schemas import MemoryItem


class MemoryBudgetExceeded(Exception):
    """Raised when trying to write beyond memory budget."""
    pass


class MemoryStore:
    """Append-only memory store with budget tracking."""

    def __init__(self, max_capacity: int):
        """
        Initialize memory store.

        Args:
            max_capacity: Maximum number of memory items allowed
        """
        self.memories: List[MemoryItem] = []
        self.max_capacity = max_capacity

    def append(self, memory_item: MemoryItem) -> None:
        """
        Append a memory item to the store.

        Args:
            memory_item: Memory item to append

        Raises:
            MemoryBudgetExceeded: If budget is exhausted
        """
        if len(self.memories) >= self.max_capacity:
            raise MemoryBudgetExceeded(
                f"Memory budget exhausted: {len(self.memories)}/{self.max_capacity}"
            )
        self.memories.append(memory_item)

    def get_all(self) -> List[MemoryItem]:
        """Get all memories in the store."""
        return self.memories.copy()

    def get_budget_remaining(self) -> int:
        """Get remaining memory budget."""
        return self.max_capacity - len(self.memories)

    def get_budget_used(self) -> int:
        """Get used memory budget."""
        return len(self.memories)

    def is_budget_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return len(self.memories) >= self.max_capacity

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = []

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dict for serialization.

        Returns:
            Dict with memories and budget info
        """
        return {
            "memories": [m.to_dict() for m in self.memories],
            "budget_remaining": self.get_budget_remaining(),
            "budget_used": self.get_budget_used(),
            "max_capacity": self.max_capacity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryStore":
        """
        Create from dict.

        Args:
            data: Dict with memories and budget info

        Returns:
            MemoryStore instance
        """
        store = cls(max_capacity=data["max_capacity"])
        store.memories = [MemoryItem.from_dict(m) for m in data["memories"]]
        return store

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryStore":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_memories_by_timerange(
        self, start_time: float, end_time: float
    ) -> List[MemoryItem]:
        """
        Get memories within a time range.

        Args:
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            List of memories in the time range
        """
        return [
            m for m in self.memories
            if start_time <= m.timestamp <= end_time
        ]

    def get_memories_by_step_range(
        self, start_step: int, end_step: int
    ) -> List[MemoryItem]:
        """
        Get memories within a step range.

        Args:
            start_step: Start step ID
            end_step: End step ID

        Returns:
            List of memories in the step range
        """
        return [
            m for m in self.memories
            if start_step <= m.step_id <= end_step
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.

        Returns:
            Dict with statistics
        """
        if not self.memories:
            return {
                "total_memories": 0,
                "avg_num_keys": 0.0,
                "avg_value_length": 0.0,
                "budget_usage": 0.0,
            }

        total_keys = sum(len(m.keys) for m in self.memories)
        total_value_length = sum(len(m.value.split()) for m in self.memories)

        return {
            "total_memories": len(self.memories),
            "avg_num_keys": total_keys / len(self.memories),
            "avg_value_length": total_value_length / len(self.memories),
            "budget_usage": len(self.memories) / self.max_capacity,
            "budget_remaining": self.get_budget_remaining(),
        }

    def __len__(self) -> int:
        """Get number of memories in store."""
        return len(self.memories)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MemoryStore(memories={len(self.memories)}, "
            f"capacity={self.max_capacity}, "
            f"remaining={self.get_budget_remaining()})"
        )
