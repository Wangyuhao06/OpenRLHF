"""
Memory Constructor AgentInstance for OpenRLHF RL Training
=========================================================
Multi-turn agent instance where the trainable model (memory constructor)
decides per-step whether to write memory slots, while a frozen LLM agent
navigates WebShop using RAG-retrieved memories as context.

Environment variables:
    WEBSHOP_SERVER_URL   - WebShop pool server URL (default: http://localhost:6001)
    FROZEN_AGENT_URL     - OpenAI-compatible API for frozen agent (required)
    FROZEN_AGENT_MODEL   - Model name for frozen agent (default: Qwen/Qwen3-8B)
    FROZEN_AGENT_API_KEY - API key for frozen agent (default: "EMPTY")
    MAX_STEPS            - Max steps per episode (default: 40)
    MEMORY_BUDGET        - Max memory slots per episode (default: 40)

Usage:
    --agent_func_path memory_constructor/scripts/agent_func_memory_constructor.py
"""

import math
import os
import re
import sys
import json
import time
import logging
from typing import Any, Dict, List, Optional

import aiohttp
import torch

# Add memory_constructor project root so memory_constructor.* imports work
from pathlib import Path

_MC_ROOT = Path(__file__).resolve().parent.parent
if str(_MC_ROOT) not in sys.path:
    sys.path.insert(0, str(_MC_ROOT))

from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

from memory_constructor.environment.memory_store import MemoryStore, MemoryBudgetExceeded
from memory_constructor.data.schemas import MemoryItem
from memory_constructor.utils.json_utils import parse_json_robust, validate_memory_item

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Configuration (via environment variables)
# ---------------------------------------------------------------------------
WEBSHOP_SERVER_URL = os.environ.get("WEBSHOP_SERVER_URL", "http://localhost:6001")
FROZEN_AGENT_URL = os.environ.get("FROZEN_AGENT_URL", "http://localhost:8000/v1")
FROZEN_AGENT_MODEL = os.environ.get("FROZEN_AGENT_MODEL", "Qwen/Qwen3-8B")
FROZEN_AGENT_API_KEY = os.environ.get("FROZEN_AGENT_API_KEY", "EMPTY")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "40"))
MEMORY_BUDGET = int(os.environ.get("MEMORY_BUDGET", "40"))

# Reward hyperparameters
RWD_NO_WRITE_PENALTY = float(os.environ.get("RWD_NO_WRITE_PENALTY", "-0.1"))
RWD_WRITE_COST = float(os.environ.get("RWD_WRITE_COST", "-0.05"))
RWD_FORMAT_ERROR = float(os.environ.get("RWD_FORMAT_ERROR", "-0.6"))
RWD_LONG_MEMORY_COEF = float(os.environ.get("RWD_LONG_MEMORY_COEF", "-0.1"))
RWD_LONG_MEMORY_THRESHOLD = int(os.environ.get("RWD_LONG_MEMORY_THRESHOLD", "128"))
RWD_RETRIEVAL_HIT = float(os.environ.get("RWD_RETRIEVAL_HIT", "0.2"))
RWD_RETRIEVAL_HIT_CAP = int(os.environ.get("RWD_RETRIEVAL_HIT_CAP", "3"))
RWD_NEVER_RETRIEVED = float(os.environ.get("RWD_NEVER_RETRIEVED", "-0.3"))
RWD_TASK_A = float(os.environ.get("RWD_TASK_A", "0.5"))
RWD_TASK_B = float(os.environ.get("RWD_TASK_B", "0.5"))

# ---------------------------------------------------------------------------
# System prompt — extracted from sft_dataset.py:75-111
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a memory constructor for a shopping agent. You decide what to store
in the agent's memory to help it complete its task.

Decide: should you write a memory now? Consider:
- Does this observation contain NEW information not already in memory?
- Would this information help the agent in future steps?
- Is this information retrievable from existing memory?

Respond in JSON:
{
    "write": true/false,
    "keys": ["key1", ...],
    "value": "..."
}"""

# ---------------------------------------------------------------------------
# Frozen agent prompt
# ---------------------------------------------------------------------------
FROZEN_AGENT_SYSTEM = """\
You are a shopping agent operating in a text-only WebShop environment.
Your job: follow the instruction, browse the shop, and buy exactly the right product.

## Action Space
- search[<keywords>]       — submit a keyword search
- click[<exact text>]      — click an item or button using its EXACT text from the page

## Output Format
Thought: <one-sentence reasoning>
Action: <exactly one action, no extra text>

## Rules
- Output ONLY the Thought + Action block. No extra commentary.
- NEVER repeat an action that already returned the same page.
- When you see "Thank you for shopping", the episode is over — output nothing.
- If you are unsure, try a more specific search rather than clicking randomly."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_frozen_agent_action(text: str) -> str:
    """Extract action string from frozen agent's ReAct output."""
    a_match = re.search(r"Action:\s*(.*)", text, re.IGNORECASE)
    if not a_match:
        return ""
    action = a_match.group(1).strip().split("\n")[0].strip().strip("\"'")
    # Normalize known navigation targets
    click_match = re.match(r"click\[(.+)\]", action, re.IGNORECASE)
    if click_match:
        inner_lower = click_match.group(1).lower()
        for kw in ("buy now", "back to search", "next >", "< prev"):
            if inner_lower == kw:
                return f"click[{kw}]"
    return action


def _truncate(text: str, max_words: int = 200) -> str:
    """Truncate text to max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _format_memory_store(memories: List[MemoryItem]) -> str:
    """Format memory store for prompt — matches sft_dataset.py:_format_prompt line 185-193."""
    if not memories:
        return "(Empty)"
    lines = []
    for i, mem in enumerate(memories, 1):
        lines.append(f"{i}. Keys: {', '.join(mem.keys)} | Value: {mem.value[:160]}")
    return "\n".join(lines)


def _format_retrieval_context(
    retrieved: List[MemoryItem], scores: Optional[List[float]] = None
) -> str:
    """Format retrieval context — matches sft_dataset.py:_format_retrieval_context."""
    if not retrieved:
        return "(Nothing retrievable yet)"
    lines = []
    for i, mem in enumerate(retrieved):
        keys = ", ".join(mem.keys) or "(no keys)"
        score_text = f" | score={scores[i]:.3f}" if scores and i < len(scores) else ""
        value_preview = mem.value[:80] if mem.value else ""
        lines.append(
            f"- Step {mem.step_id}: keys={keys}{score_text} | value={value_preview}"
        )
    return "\n".join(lines)


def _format_user_message(
    instruction: str,
    task_requirements: str,
    observation: str,
    local_history: List[str],
    memory_store: List[MemoryItem],
    retrieval_context: str,
    budget_remaining: int,
    episode_progress: float,
    max_obs_words: int = 200,
) -> str:
    """Format user message — matches sft_dataset.py prompt template fields exactly."""
    history_text = "\n".join(local_history) if local_history else "(No previous steps)"
    memory_text = _format_memory_store(memory_store)

    return (
        f"Agent Task:\n{instruction}\n\n"
        f"Key Requirements:\n{task_requirements}\n\n"
        f"Current Observation:\n{_truncate(observation, max_obs_words)}\n\n"
        f"Local History (last 3 steps):\n{history_text}\n\n"
        f"Current Memory Store:\n{memory_text}\n\n"
        f"Currently Retrievable (top matches for this observation):\n{retrieval_context}\n\n"
        f"Budget Remaining: {budget_remaining}\n"
        f"Episode Progress: {episode_progress:.1%}"
    )


# ---------------------------------------------------------------------------
# Lazy retriever (loaded once per worker, shared across episodes)
# ---------------------------------------------------------------------------
_retriever = None


def _get_retriever():
    """Get or create the shared HybridRetriever instance."""
    global _retriever
    if _retriever is None:
        from memory_constructor.models.retriever import HybridRetriever

        _retriever = HybridRetriever(
            bm25_weight=0.5,
            dense_weight=0.5,
            retrieval_k=3,  # default, overridden per call
        )
        logger.info("Initialized shared HybridRetriever")
    return _retriever


# ---------------------------------------------------------------------------
# AgentInstance
# ---------------------------------------------------------------------------

class AgentInstance(AgentInstanceBase):
    """Memory constructor agent instance for OpenRLHF multi-turn RL training."""

    # Max total words across ALL turns — prevents OOM during backward pass.
    # With ~1.3 tokens/word, 3000 words ≈ 4000 tokens. Keeps sequences
    # within generate_max_len=4096 so ZeRO-3 backward doesn't deadlock.
    MAX_TOTAL_WORDS = int(os.environ.get("MAX_TOTAL_WORDS", "1500"))
    # Max words per observation in user message (truncated for sequence control)
    MAX_OBS_WORDS = int(os.environ.get("MAX_OBS_WORDS", "200"))

    def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.max_steps = MAX_STEPS
        self.session_id = None
        self.instruction = ""
        self.task_requirements = "(Not available)"
        self.current_obs = ""  # Current WebShop observation (for retrieval queries)
        self._http_session = None
        self._cumulative_words = 0  # Track total words across all turns

        # Memory state
        self.memory_store = None
        self.local_history: List[str] = []
        # Tracks retrieval counts per memory index: {memory_idx: count}
        self.retrieval_tracker: Dict[int, int] = {}
        # Consecutive format errors — force done after too many
        self.consecutive_format_errors = 0
        self.MAX_CONSECUTIVE_FORMAT_ERRORS = 3

    async def _get_http_session(self):
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def _close_http_session(self):
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------
    async def reset(self, states: dict, **kwargs):
        """Create a new WebShop session and return initial observation for constructor."""
        label = states.get("label", "")

        # Parse task_idx from label
        task_idx = None
        if label and label.strip().isdigit():
            task_idx = int(label.strip())

        # Create WebShop session
        session = await self._get_http_session()
        payload = {}
        if task_idx is not None:
            payload["task_idx"] = task_idx

        async with session.post(
            f"{WEBSHOP_SERVER_URL}/create_session", json=payload
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        self.session_id = data["session_id"]
        obs = data["obs"]
        self.instruction = data.get("instruction_text", "")
        self.current_obs = obs  # Track current WebShop observation
        self.step_idx = 0

        # Initialize memory state
        self.memory_store = MemoryStore(max_capacity=MEMORY_BUDGET)
        self.local_history = []
        self.retrieval_tracker = {}

        # Format initial prompt (SFT-aligned, with Qwen3 chat template)
        user_msg = _format_user_message(
            instruction=self.instruction,
            task_requirements=self.task_requirements,
            observation=obs,
            local_history=[],
            memory_store=[],
            retrieval_context="(Nothing retrievable yet)",
            budget_remaining=MEMORY_BUDGET,
            episode_progress=0.0,
            max_obs_words=self.MAX_OBS_WORDS,
        )

        initial_prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n</think>\n"
        )

        self._cumulative_words = len(initial_prompt.split())

        return {"observation": initial_prompt}

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------
    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """Process constructor output, run frozen agent, return feedback."""
        action_text = states["action_text"]

        step_reward = 0.0
        format_valid = True

        # 1. Parse constructor JSON output
        # Strip Qwen3 <think>...</think> blocks before checking format
        clean_text = re.sub(r'<think>.*?</think>', '', action_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r'<think>.*$', '', clean_text, flags=re.DOTALL).strip()
        constructor_action = parse_json_robust(action_text)
        if constructor_action == {"write": False, "keys": [], "value": ""}:
            # Check if this was a parse failure (vs genuine no-write)
            if clean_text and not clean_text.startswith("{"):
                format_valid = False
                step_reward += RWD_FORMAT_ERROR
                self.consecutive_format_errors += 1
            else:
                self.consecutive_format_errors = 0
        else:
            self.consecutive_format_errors = 0

        # Early termination on too many consecutive format errors
        if self.consecutive_format_errors >= self.MAX_CONSECUTIVE_FORMAT_ERRORS:
            logger.warning(f"Forcing done: {self.consecutive_format_errors} consecutive format errors")
            total_reward = step_reward + self._compute_episode_rewards(
                webshop_reward=0.0, is_success=False)
            environment_feedback = f"<|im_end|>\n<|im_start|>user\nEpisode ended.<|im_end|>"
            if self.session_id:
                try:
                    session = await self._get_http_session()
                    async with session.post(
                        f"{WEBSHOP_SERVER_URL}/close_session",
                        json={"session_id": self.session_id},
                    ) as resp:
                        pass
                except Exception:
                    pass
            return {
                "environment_feedback": environment_feedback,
                "rewards": torch.tensor(total_reward, dtype=torch.float),
                "done": True,
            }
        constructor_action = validate_memory_item(
            constructor_action,
            max_key_tokens=64,
            max_value_tokens=256,
            max_num_keys=4,
        )

        # 2. Write memory if applicable
        memory_written = False
        write_decision = constructor_action.get("write", False)

        if write_decision:
            try:
                mem_item = MemoryItem(
                    write=True,
                    keys=constructor_action["keys"],
                    value=constructor_action["value"],
                    timestamp=time.time(),
                    step_id=self.step_idx,
                    metadata={},
                )
                self.memory_store.append(mem_item)
                memory_written = True
                # Init retrieval tracker for this memory
                mem_idx = len(self.memory_store.memories) - 1
                self.retrieval_tracker[mem_idx] = 0
            except MemoryBudgetExceeded:
                logger.debug("Memory budget exceeded, skipping write")
                memory_written = False

        # Per-step reward: write vs no-write
        if memory_written:
            step_reward += RWD_WRITE_COST
            # Long memory penalty
            value_words = len(constructor_action.get("value", "").split())
            if value_words > RWD_LONG_MEMORY_THRESHOLD:
                excess = value_words / RWD_LONG_MEMORY_THRESHOLD - 1
                step_reward += RWD_LONG_MEMORY_COEF * excess
        elif format_valid:
            step_reward += RWD_NO_WRITE_PENALTY

        # 3. Retrieve memories using current WebShop observation as query
        retriever = _get_retriever()
        query_text = _truncate(self.current_obs, max_words=200)

        dynamic_k = max(1, math.ceil((self.step_idx + 1) / 2))
        retrieved_memories = []
        retrieval_scores = []

        if len(self.memory_store) > 0:
            results = retriever.retrieve(
                query_text, self.memory_store, k=dynamic_k, return_scores=True
            )
            retrieved_memories = [mem for mem, _ in results]
            retrieval_scores = [score for _, score in results]

            # Update retrieval tracker
            all_memories = self.memory_store.get_all()
            for mem in retrieved_memories:
                for idx, stored_mem in enumerate(all_memories):
                    if stored_mem.step_id == mem.step_id:
                        if idx in self.retrieval_tracker:
                            self.retrieval_tracker[idx] += 1
                        break

        # 4. Frozen agent acts in WebShop
        agent_action = await self._frozen_agent_act(
            observation=self.current_obs,
            retrieved_memories=retrieved_memories,
        )

        # 5. WebShop step
        obs = ""
        webshop_reward = 0.0
        done = False

        if not agent_action:
            obs = "Error: Frozen agent returned no action."
            done = True
        elif self.session_id is None:
            obs = "Error: No active session."
            done = True
        else:
            try:
                session = await self._get_http_session()
                payload = {"session_id": self.session_id, "action": agent_action}
                async with session.post(
                    f"{WEBSHOP_SERVER_URL}/step", json=payload
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                obs = data["obs"]
                webshop_reward = data["reward"]
                done = data["done"]
                self.current_obs = obs  # Update current observation
            except Exception as e:
                logger.warning(f"WebShop step failed: {e}")
                obs = f"Error: Environment step failed ({e})"
                done = True

        self.step_idx += 1

        # 6. Update local history (keep last 3)
        history_entry = f"Step {self.step_idx}: Agent={agent_action[:80] if agent_action else 'none'}"
        self.local_history.append(history_entry)
        if len(self.local_history) > 3:
            self.local_history = self.local_history[-3:]

        # Force done at max steps
        if self.step_idx >= self.max_steps:
            done = True

        # Force done if cumulative sequence length would exceed limit
        if not done:
            # Estimate words for next turn's environment_feedback
            est_next_words = len(obs.split()[:self.MAX_OBS_WORDS]) + 150  # obs + boilerplate
            if self._cumulative_words + est_next_words > self.MAX_TOTAL_WORDS:
                logger.info(
                    f"Forcing done: cumulative {self._cumulative_words} + next ~{est_next_words} "
                    f"would exceed MAX_TOTAL_WORDS={self.MAX_TOTAL_WORDS}"
                )
                done = True

        # 7. Compute episode-end rewards if done
        episode_reward = 0.0
        if done:
            episode_reward = self._compute_episode_rewards(
                webshop_reward=webshop_reward,
                is_success=webshop_reward > 0,
            )

        total_reward = step_reward + episode_reward

        # 8. Format environment feedback (SFT-aligned)
        if done:
            environment_feedback = f"<|im_end|>\n<|im_start|>user\nEpisode ended.<|im_end|>"
        else:
            # Retrieve context for the NEW observation
            new_retrieval_context = "(Nothing retrievable yet)"
            if len(self.memory_store) > 0:
                new_k = max(1, math.ceil((self.step_idx + 1) / 2))
                new_results = retriever.retrieve(
                    _truncate(obs, 200), self.memory_store, k=new_k, return_scores=True
                )
                new_mems = [m for m, _ in new_results]
                new_scores = [s for _, s in new_results]
                new_retrieval_context = _format_retrieval_context(new_mems, new_scores)

            user_msg = _format_user_message(
                instruction=self.instruction,
                task_requirements=self.task_requirements,
                observation=obs,
                local_history=self.local_history,
                memory_store=self.memory_store.get_all(),
                retrieval_context=new_retrieval_context,
                budget_remaining=self.memory_store.get_budget_remaining(),
                episode_progress=self.step_idx / self.max_steps,
                max_obs_words=self.MAX_OBS_WORDS,
            )
            environment_feedback = (
                f"<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n</think>\n"
            )
            self._cumulative_words += len(environment_feedback.split())

        # If LLM output already contains <|im_end|>, don't prepend another one
        if "<|im_end|>" in action_text:
            environment_feedback = environment_feedback.lstrip("<|im_end|>\n")

        # Clean up on done
        if done:
            await self._try_close_session()

        return {
            "rewards": torch.tensor(total_reward, dtype=torch.float),
            "scores": torch.tensor(total_reward, dtype=torch.float),
            "environment_feedback": environment_feedback,
            "done": done,
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": {
                "webshop_reward": torch.tensor(webshop_reward),
                "step_reward": torch.tensor(step_reward),
                "episode_reward": torch.tensor(episode_reward),
                "turn_count": torch.tensor(self.step_idx),
                "num_memories": torch.tensor(len(self.memory_store)),
                "memory_written": torch.tensor(float(memory_written)),
            },
        }

    # ------------------------------------------------------------------
    # Episode-end reward computation
    # ------------------------------------------------------------------
    def _compute_episode_rewards(
        self, webshop_reward: float, is_success: bool
    ) -> float:
        """Compute end-of-episode rewards: retrieval hits + task reward."""
        reward = 0.0
        total_memories = len(self.memory_store)

        # Retrieval hit / never-retrieved rewards
        for mem_idx, count in self.retrieval_tracker.items():
            if count > 0:
                # Proportional reward capped at k
                reward += RWD_RETRIEVAL_HIT * min(count, RWD_RETRIEVAL_HIT_CAP)
            else:
                # Never retrieved penalty
                reward += RWD_NEVER_RETRIEVED

        # Task reward: is_success * (rwd_a * total_mem/total_steps + rwd_b)
        if is_success:
            mem_ratio = total_memories / max(self.step_idx, 1)
            reward += RWD_TASK_A * mem_ratio + RWD_TASK_B

        return reward

    # ------------------------------------------------------------------
    # Frozen agent interaction
    # ------------------------------------------------------------------
    async def _frozen_agent_act(
        self,
        observation: str,
        retrieved_memories: List[MemoryItem],
    ) -> str:
        """Call frozen agent API to get a WebShop action."""
        # Format memory context
        if retrieved_memories:
            mem_lines = []
            for i, mem in enumerate(retrieved_memories, 1):
                keys_str = ", ".join(mem.keys)
                mem_lines.append(f"  {i}. [{keys_str}] {mem.value[:120]}")
            memory_context = "Retrieved Memories:\n" + "\n".join(mem_lines)
        else:
            memory_context = "Retrieved Memories: (none)"

        user_content = (
            f"Task: {self.instruction}\n\n"
            f"{memory_context}\n\n"
            f"Current Page:\n{_truncate(observation, 500)}"
        )

        try:
            session = await self._get_http_session()
            payload = {
                "model": FROZEN_AGENT_MODEL,
                "messages": [
                    {"role": "system", "content": FROZEN_AGENT_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 150,
                "temperature": 0.7,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {FROZEN_AGENT_API_KEY}",
            }
            async with session.post(
                f"{FROZEN_AGENT_URL}/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            response_text = data["choices"][0]["message"]["content"]
            action = _parse_frozen_agent_action(response_text)

            if not action:
                # Fallback: search with instruction keywords on first step
                if self.step_idx == 0:
                    action = f"search[{self.instruction[:50]}]"
                else:
                    action = "click[buy now]"
                logger.debug(f"Frozen agent parse failed, fallback: {action}")

            return action

        except Exception as e:
            logger.warning(f"Frozen agent call failed: {e}")
            # Fallback action
            if self.step_idx == 0:
                return f"search[{self.instruction[:50]}]"
            return "click[buy now]"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    async def _try_close_session(self):
        """Best-effort session cleanup."""
        if self.session_id is None:
            return
        try:
            session = await self._get_http_session()
            payload = {"session_id": self.session_id}
            async with session.post(
                f"{WEBSHOP_SERVER_URL}/close_session", json=payload
            ) as resp:
                pass
        except Exception:
            pass
        finally:
            self.session_id = None
            await self._close_http_session()


# ---------------------------------------------------------------------------
# AgentExecutor (required by OpenRLHF)
# ---------------------------------------------------------------------------

class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
