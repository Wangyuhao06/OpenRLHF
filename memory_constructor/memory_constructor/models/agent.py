"""
GPT-5 Agent Wrapper for WebShop environment.

This module provides a wrapper around the GPT-5 API to act as the fixed
downstream agent in the memory constructor system. The agent receives
observations and retrieved memories, then generates actions.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from memory_constructor.data.schemas import MemoryItem

logger = logging.getLogger(__name__)


class GPT5Agent:
    """
    Fixed agent using GPT-5 API.

    This agent acts in the WebShop environment using retrieved memories
    to inform its decisions. It is kept fixed during memory constructor training.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        temperature: float = 0.7,
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize GPT-5 agent.

        Args:
            api_key: OpenAI API key. If None, loads from environment or .env file
            model: Model name (default: "gpt-5")
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response (default: 256)
            system_prompt: Custom system prompt. If None, uses default WebShop prompt
            base_url: Custom API base URL (optional)
        """
        # Load API key
        if api_key is None:
            load_dotenv()  # Load from .env file
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key not provided and not found in environment. "
                    "Please set OPENAI_API_KEY or OPENAI_KEY in .env file or pass api_key parameter."
                )

        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Default system prompt for WebShop
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        logger.info(
            f"Initialized GPT5Agent with model={model}, temperature={temperature}"
        )

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for WebShop agent."""
        return """You are a helpful shopping assistant in the WebShop environment. Your goal is to help users find and purchase products that match their requirements.

Available actions:
- search[query]: Search for products using a query
- click[button]: Click on a button or link (e.g., "Next >", "< Prev", "Buy Now")
- click[product_id]: Click on a specific product to view details

Guidelines:
- Carefully read the user's requirements and any relevant memories
- Search for products that match the requirements
- Compare options and select the best match
- Pay attention to specific attributes like color, size, price, features
- Use memories to recall important information from earlier in the session

Format your response as a single action, e.g.:
search[red shoes size 10]
click[B07XYZ123]
click[Buy Now]
"""

    def _format_prompt(
        self, observation: str, retrieved_memories: List[MemoryItem], instruction: Optional[str] = None
    ) -> str:
        """
        Format input prompt with observation and memories.

        Args:
            observation: Current observation from environment
            retrieved_memories: List of retrieved memory items
            instruction: Optional task instruction

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        # Add instruction if provided
        if instruction:
            prompt_parts.append(f"Task: {instruction}\n")

        # Add retrieved memories
        if retrieved_memories:
            prompt_parts.append("Relevant memories from earlier in the session:")
            for i, mem in enumerate(retrieved_memories, 1):
                prompt_parts.append(f"{i}. {mem.value}")
            prompt_parts.append("")  # Empty line

        # Add current observation
        prompt_parts.append(f"Current observation:\n{observation}\n")

        # Add action prompt
        prompt_parts.append("What action should you take? (Respond with a single action)")

        return "\n".join(prompt_parts)

    def act(
        self,
        observation: str,
        retrieved_memories: List[MemoryItem],
        instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate action given observation and retrieved memories.

        Args:
            observation: Current observation from environment
            retrieved_memories: List of retrieved memory items
            instruction: Optional task instruction
            **kwargs: Additional arguments (e.g., temperature override)

        Returns:
            Action string
        """
        # Format prompt
        prompt = self._format_prompt(observation, retrieved_memories, instruction)

        # Override temperature if provided
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            # Call GPT-5 API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            action = response.choices[0].message.content.strip()

            logger.debug(f"Generated action: {action}")

            return action

        except Exception as e:
            logger.error(f"Error calling GPT-5 API: {e}")
            # Return a safe default action
            return "search[product]"

    def generate_query(
        self,
        observation: str,
        instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a compact query for memory retrieval.

        This method prompts GPT-5 to format the current observation and task
        into a compact sentence suitable for memory retrieval, ensuring no
        key information is missed.

        Args:
            observation: Current observation from environment
            instruction: Optional task instruction
            **kwargs: Additional arguments

        Returns:
            Compact query string
        """
        # Create prompt for query generation
        query_prompt_parts = []

        if instruction:
            query_prompt_parts.append(f"Task: {instruction}\n")

        query_prompt_parts.append(f"Current observation:\n{observation}\n")

        query_prompt_parts.append(
            "Generate a compact query (1-2 sentences) to search for relevant memories. "
            "Include key information like product attributes, requirements, or constraints. "
            "Do not include unnecessary details."
        )

        query_prompt = "\n".join(query_prompt_parts)

        try:
            # Call GPT-5 API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates concise search queries.",
                    },
                    {"role": "user", "content": query_prompt},
                ],
                temperature=0.3,  # Lower temperature for more focused queries
                max_tokens=100,
            )

            query = response.choices[0].message.content.strip()

            logger.debug(f"Generated query: {query}")

            return query

        except Exception as e:
            logger.error(f"Error generating query: {e}")
            # Fallback: use observation directly (truncated)
            return observation[:200]

    def act_with_query_generation(
        self,
        observation: str,
        memory_store,
        retriever,
        instruction: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate query, retrieve memories, and act.

        This is a convenience method that combines query generation,
        memory retrieval, and action generation in one call.

        Args:
            observation: Current observation
            memory_store: MemoryStore instance
            retriever: Retriever instance (e.g., HybridRetriever)
            instruction: Optional task instruction
            **kwargs: Additional arguments

        Returns:
            Dictionary with:
                - action: Generated action string
                - query: Generated query string
                - retrieved_memories: List of retrieved MemoryItem objects
        """
        # Generate query
        query = self.generate_query(observation, instruction, **kwargs)

        # Retrieve memories
        retrieved_memories = retriever.retrieve(query, memory_store)

        # Generate action
        action = self.act(observation, retrieved_memories, instruction, **kwargs)

        return {
            "action": action,
            "query": query,
            "retrieved_memories": retrieved_memories,
        }


class MockAgent:
    """
    Mock agent for testing without API calls.

    This agent returns simple rule-based actions for testing purposes.
    """

    def __init__(self):
        """Initialize mock agent."""
        self.call_count = 0
        logger.info("Initialized MockAgent")

    def act(
        self,
        observation: str,
        retrieved_memories: List[MemoryItem],
        instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate mock action.

        Args:
            observation: Current observation
            retrieved_memories: Retrieved memories (unused)
            instruction: Task instruction (unused)
            **kwargs: Additional arguments (unused)

        Returns:
            Mock action string
        """
        self.call_count += 1

        # Simple rule-based logic
        if "search" in observation.lower() or self.call_count == 1:
            return "search[product]"
        elif "buy now" in observation.lower():
            return "click[Buy Now]"
        elif "next" in observation.lower():
            return "click[Next >]"
        else:
            return "search[item]"

    def generate_query(
        self,
        observation: str,
        instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate mock query.

        Args:
            observation: Current observation
            instruction: Task instruction
            **kwargs: Additional arguments

        Returns:
            Mock query string
        """
        # Extract first few words from observation
        words = observation.split()[:10]
        return " ".join(words)

    def act_with_query_generation(
        self,
        observation: str,
        memory_store,
        retriever,
        instruction: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Mock version of act_with_query_generation.

        Args:
            observation: Current observation
            memory_store: MemoryStore instance
            retriever: Retriever instance
            instruction: Task instruction
            **kwargs: Additional arguments

        Returns:
            Dictionary with action, query, and retrieved_memories
        """
        query = self.generate_query(observation, instruction, **kwargs)
        retrieved_memories = retriever.retrieve(query, memory_store)
        action = self.act(observation, retrieved_memories, instruction, **kwargs)

        return {
            "action": action,
            "query": query,
            "retrieved_memories": retrieved_memories,
        }
