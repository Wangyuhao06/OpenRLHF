"""
Memory store for managing constructed memories.

This module provides a simple in-memory store for managing memories
during episode execution.
"""

from typing import List, Dict, Any, Optional
from memory_constructor.data.schemas import MemoryItem


class MemoryStore:
    """In-memory store for managing memories."""

    def __init__(self):
        """Initialize empty memory store."""
        self.memories: List[MemoryItem] = []

    def add(self, memory: Dict[str, Any]) -> None:
        """
        Add a memory to the store.

        Args:
            memory: Memory dict with keys, value, step, etc.
        """
        if isinstance(memory, dict):
            memory_item = MemoryItem.from_dict(memory)
        else:
            memory_item = memory

        self.memories.append(memory_item)

    def get_all(self) -> List[MemoryItem]:
        """Get all memories."""
        return self.memories

    def get_by_keys(self, keys: List[str]) -> List[MemoryItem]:
        """
        Get memories that match any of the given keys.

        Args:
            keys: List of keys to search for

        Returns:
            List of matching memories
        """
        matching = []
        for memory in self.memories:
            if any(k in memory.keys for k in keys):
                matching.append(memory)
        return matching

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = []

    def __len__(self) -> int:
        """Get number of memories."""
        return len(self.memories)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"memories": [m.to_dict() for m in self.memories]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryStore":
        """Create from dictionary."""
        store = cls()
        for mem_dict in data.get("memories", []):
            store.add(mem_dict)
        return store
