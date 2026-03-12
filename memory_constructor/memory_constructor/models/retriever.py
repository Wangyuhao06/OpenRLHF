"""
Hybrid retriever combining BM25 and dense embedding retrieval.

This module implements three retrieval strategies:
1. BM25Retriever: Keyword-based retrieval using BM25 algorithm
2. DenseRetriever: Semantic retrieval using sentence embeddings
3. HybridRetriever: Combines both with configurable weights
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch

from memory_constructor.data.memory_store import MemoryStore
from memory_constructor.data.schemas import MemoryItem

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25-based keyword retrieval on memory keys."""

    def __init__(self, tokenizer_fn=None):
        """
        Initialize BM25 retriever.

        Args:
            tokenizer_fn: Optional tokenizer function. If None, uses simple whitespace split.
        """
        self.tokenizer_fn = tokenizer_fn or (lambda x: x.lower().split())
        self.bm25 = None
        self.memory_items = []

    def _build_index(self, memory_store: MemoryStore):
        """Build BM25 index from memory store."""
        self.memory_items = memory_store.get_all()

        if not self.memory_items:
            self.bm25 = None
            return

        # Tokenize all keys for each memory
        corpus = []
        for mem in self.memory_items:
            # Combine all keys into a single document
            keys_text = " ".join(mem.keys)
            tokenized = self.tokenizer_fn(keys_text)
            corpus.append(tokenized)

        self.bm25 = BM25Okapi(corpus)

    def retrieve(
        self, query: str, memory_store: MemoryStore, k: int = 3
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Retrieve top-k memories using BM25.

        Args:
            query: Query string
            memory_store: Memory store to search
            k: Number of memories to retrieve

        Returns:
            List of (memory_item, score) tuples, sorted by score descending
        """
        # Rebuild index (in case memory store changed)
        self._build_index(memory_store)

        if self.bm25 is None or not self.memory_items:
            return []

        # Tokenize query
        tokenized_query = self.tokenizer_fn(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]

        # Return memory items with scores
        results = [
            (self.memory_items[idx], float(scores[idx])) for idx in top_k_indices
        ]

        return results


class DenseRetriever:
    """Dense embedding-based semantic retrieval."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize dense retriever.

        Args:
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.memory_items = []
        self.embeddings = None

    def _build_index(self, memory_store: MemoryStore):
        """Build embedding index from memory store."""
        self.memory_items = memory_store.get_all()

        if not self.memory_items:
            self.embeddings = None
            return

        # Create text for each memory: keys + value preview (first 100 chars)
        texts = []
        for mem in self.memory_items:
            keys_text = " ".join(mem.keys)
            value_preview = mem.value[:100] if len(mem.value) > 100 else mem.value
            combined_text = f"{keys_text} {value_preview}"
            texts.append(combined_text)

        # Encode all texts
        self.embeddings = self.model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )

    def retrieve(
        self, query: str, memory_store: MemoryStore, k: int = 3
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Retrieve top-k memories using semantic similarity.

        Args:
            query: Query string
            memory_store: Memory store to search
            k: Number of memories to retrieve

        Returns:
            List of (memory_item, score) tuples, sorted by score descending
        """
        # Rebuild index (in case memory store changed)
        self._build_index(memory_store)

        if self.embeddings is None or not self.memory_items:
            return []

        # Encode query
        query_embedding = self.model.encode(
            query, convert_to_tensor=True, show_progress_bar=False
        )

        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), self.embeddings
        )

        # Get top-k indices
        top_k_values, top_k_indices = torch.topk(similarities, min(k, len(similarities)))

        # Return memory items with scores
        results = [
            (self.memory_items[idx.item()], float(top_k_values[i].item()))
            for i, idx in enumerate(top_k_indices)
        ]

        return results


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and dense retrieval.

    Combines keyword-based BM25 retrieval with semantic dense retrieval
    using configurable weights. Supports temporal ordering of memories.
    """

    def __init__(
        self,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        retrieval_k: int = 3,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_fn=None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_weight: Weight for BM25 scores (default: 0.5)
            dense_weight: Weight for dense scores (default: 0.5)
            retrieval_k: Number of memories to retrieve (default: 3)
            dense_model: Sentence transformer model name
            tokenizer_fn: Optional tokenizer for BM25
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.k = retrieval_k

        self.bm25_retriever = BM25Retriever(tokenizer_fn=tokenizer_fn)
        self.dense_retriever = DenseRetriever(model_name=dense_model)

        logger.info(
            f"Initialized HybridRetriever with bm25_weight={bm25_weight}, "
            f"dense_weight={dense_weight}, k={retrieval_k}"
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HybridRetriever":
        """
        Create retriever from config dict.

        Args:
            config: Configuration dictionary with keys:
                - bm25_weight (float)
                - dense_weight (float)
                - retrieval_k (int)
                - dense_model (str, optional)

        Returns:
            HybridRetriever instance
        """
        return cls(
            bm25_weight=config.get("bm25_weight", 0.5),
            dense_weight=config.get("dense_weight", 0.5),
            retrieval_k=config.get("retrieval_k", 3),
            dense_model=config.get(
                "dense_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
        )

    def _normalize_scores(
        self, results: List[Tuple[MemoryItem, float]]
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            results: List of (memory_item, score) tuples

        Returns:
            List of (memory_item, normalized_score) tuples
        """
        if not results:
            return []

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same, return uniform scores
            return [(mem, 1.0) for mem, _ in results]

        normalized = [
            (mem, (score - min_score) / (max_score - min_score))
            for mem, score in results
        ]

        return normalized

    def _combine_scores(
        self,
        bm25_results: List[Tuple[MemoryItem, float]],
        dense_results: List[Tuple[MemoryItem, float]],
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Combine BM25 and dense retrieval scores.

        Args:
            bm25_results: BM25 retrieval results
            dense_results: Dense retrieval results

        Returns:
            Combined results sorted by score descending
        """
        # Normalize scores
        bm25_normalized = self._normalize_scores(bm25_results)
        dense_normalized = self._normalize_scores(dense_results)

        # Create score dictionaries (using memory step_id as key)
        bm25_scores = {mem.step_id: score for mem, score in bm25_normalized}
        dense_scores = {mem.step_id: score for mem, score in dense_normalized}

        # Collect all unique memories
        all_memories = {}
        for mem, _ in bm25_results + dense_results:
            all_memories[mem.step_id] = mem

        # Combine scores
        combined = []
        for step_id, mem in all_memories.items():
            bm25_score = bm25_scores.get(step_id, 0.0)
            dense_score = dense_scores.get(step_id, 0.0)

            combined_score = (
                self.bm25_weight * bm25_score + self.dense_weight * dense_score
            )

            combined.append((mem, combined_score))

        # Sort by combined score descending
        combined.sort(key=lambda x: x[1], reverse=True)

        return combined

    def retrieve(
        self,
        query: str,
        memory_store: MemoryStore,
        k: Optional[int] = None,
        return_scores: bool = False,
    ) -> List[MemoryItem]:
        """
        Retrieve top-k memories using hybrid retrieval.

        Args:
            query: Query string
            memory_store: Memory store to search
            k: Number of memories to retrieve (uses self.k if None)
            return_scores: If True, return (memory, score) tuples instead of just memories

        Returns:
            List of MemoryItem objects (or (MemoryItem, score) tuples if return_scores=True),
            sorted by relevance score descending
        """
        k = k or self.k

        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, memory_store, k=k * 2)
        dense_results = self.dense_retriever.retrieve(query, memory_store, k=k * 2)

        # Combine scores
        combined_results = self._combine_scores(bm25_results, dense_results)

        # Take top-k
        top_k_results = combined_results[:k]

        logger.debug(
            f"Retrieved {len(top_k_results)} memories for query: '{query[:50]}...'"
        )

        if return_scores:
            return top_k_results
        else:
            return [mem for mem, _ in top_k_results]

    def retrieve_with_stats(
        self, query: str, memory_store: MemoryStore, k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve memories and return detailed statistics.

        Args:
            query: Query string
            memory_store: Memory store to search
            k: Number of memories to retrieve

        Returns:
            Dictionary with:
                - memories: List of retrieved MemoryItem objects
                - scores: List of combined scores
                - bm25_scores: List of BM25 scores
                - dense_scores: List of dense scores
                - num_retrieved: Number of memories retrieved
        """
        k = k or self.k

        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, memory_store, k=k * 2)
        dense_results = self.dense_retriever.retrieve(query, memory_store, k=k * 2)

        # Normalize scores
        bm25_normalized = self._normalize_scores(bm25_results)
        dense_normalized = self._normalize_scores(dense_results)

        # Combine scores
        combined_results = self._combine_scores(bm25_results, dense_results)

        # Take top-k
        top_k_results = combined_results[:k]

        # Extract detailed scores
        memories = [mem for mem, _ in top_k_results]
        combined_scores = [score for _, score in top_k_results]

        bm25_score_dict = {mem.step_id: score for mem, score in bm25_normalized}
        dense_score_dict = {mem.step_id: score for mem, score in dense_normalized}

        bm25_scores = [bm25_score_dict.get(mem.step_id, 0.0) for mem in memories]
        dense_scores = [dense_score_dict.get(mem.step_id, 0.0) for mem in memories]

        return {
            "memories": memories,
            "scores": combined_scores,
            "bm25_scores": bm25_scores,
            "dense_scores": dense_scores,
            "num_retrieved": len(memories),
        }
