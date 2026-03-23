"""
WebShop AgentInstance for OpenRLHF
==================================
Multi-turn agent instance that connects to a WebShop HTTP server
for online trajectory generation during RL training.

Environment variables:
    WEBSHOP_SERVER_URL  - WebShop pool server URL (default: http://localhost:6001)
    WEBSHOP_MAX_STEPS   - Max steps per episode (default: 15)

Usage:
    # In training script:
    --agent_func_path examples/python/agent_func_webshop.py
"""

import os
import re
import logging
from typing import Any, Dict

import aiohttp
import torch

from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

WEBSHOP_SERVER_URL = os.environ.get("WEBSHOP_SERVER_URL", "http://localhost:6001")
WEBSHOP_MAX_STEPS = int(os.environ.get("WEBSHOP_MAX_STEPS", "15"))

SYSTEM_PROMPT = """\
You are a shopping agent operating in a text-only WebShop environment.
Your job: follow the instruction, browse the shop, and buy exactly the right product.

## Action Space
- search[<keywords>]       — submit a keyword search
- click[<exact text>]      — click an item or button using its EXACT text from the page
                             (product IDs like B09QQWRW2M, options like "large", nav like "buy now")

## Output Format  ← YOU MUST FOLLOW THIS EXACTLY, EVERY TURN
Thought: <one-sentence reasoning>
Action: <exactly one action, no extra text>

## Rules
- Output ONLY the Thought + Action block. No extra commentary.
- NEVER repeat an action that already returned the same page.
- When you see "Thank you for shopping", the episode is over — output nothing.
- If you are unsure, try a more specific search rather than clicking randomly."""


def parse_action(text: str) -> str:
    """Extract action string from LLM output (Thought: ... Action: ...)."""
    a_match = re.search(r"Action:\s*(.*)", text, re.IGNORECASE)
    if not a_match:
        return ""
    action = a_match.group(1).strip().split("\n")[0].strip().strip("\"'")
    # Normalize case for click actions with known navigation targets
    click_match = re.match(r"click\[(.+)\]", action, re.IGNORECASE)
    if click_match:
        inner = click_match.group(1)
        inner_lower = inner.lower()
        for kw in ("buy now", "back to search", "next >", "< prev"):
            if inner_lower == kw:
                return f"click[{kw}]"
    return action


class AgentInstance(AgentInstanceBase):
    async def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.max_steps = WEBSHOP_MAX_STEPS
        self.session_id = None
        self._http_session = None

    async def _get_http_session(self):
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def _close_http_session(self):
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

    async def reset(self, states: dict, **kwargs):
        """Create a new WebShop session and return formatted initial observation."""
        label = states.get("label", "")

        # Use label as task_idx if it's a valid integer
        task_idx = None
        if label and label.strip().isdigit():
            task_idx = int(label.strip())

        # Create session via HTTP
        session = await self._get_http_session()
        payload = {}
        if task_idx is not None:
            payload["task_idx"] = task_idx

        async with session.post(f"{WEBSHOP_SERVER_URL}/create_session", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        self.session_id = data["session_id"]
        obs = data["obs"]
        instruction_text = data.get("instruction_text", "")
        self.step_idx = 0

        # Format with Qwen chat template
        initial_prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Instruction: {instruction_text}\n\n"
            f"Observation:\n{obs}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return {"observation": initial_prompt}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """Parse action from LLM output, execute in WebShop, return feedback."""
        action_text = states["action_text"]

        action = parse_action(action_text)

        reward = 0.0
        done = False
        obs = ""

        if not action:
            # No valid action parsed
            obs = "Error: Could not parse a valid action. Please respond with:\nThought: <reasoning>\nAction: <action>"
        elif self.session_id is None:
            obs = "Error: No active session."
            done = True
        else:
            # Execute action in WebShop
            try:
                session = await self._get_http_session()
                payload = {"session_id": self.session_id, "action": action}
                async with session.post(f"{WEBSHOP_SERVER_URL}/step", json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                obs = data["obs"]
                reward = data["reward"]
                done = data["done"]
            except Exception as e:
                logger.warning(f"WebShop step failed: {e}")
                obs = f"Error: Environment step failed ({e})"
                done = True

        self.step_idx += 1

        # Force done if max steps reached
        if self.step_idx >= self.max_steps:
            done = True

        # Format environment feedback with chat template
        if done:
            environment_feedback = f"<|im_end|>\n<|im_start|>user\nObservation:\n{obs}<|im_end|>"
        else:
            environment_feedback = (
                f"<|im_end|>\n<|im_start|>user\n"
                f"Observation:\n{obs}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        # If LLM output already contains <|im_end|>, don't prepend another one
        if "<|im_end|>" in action_text:
            environment_feedback = environment_feedback.lstrip("<|im_end|>\n")

        # Clean up on done
        if done:
            await self._try_close_session()

        return {
            "rewards": torch.tensor(reward, dtype=torch.float),
            "scores": torch.tensor(reward, dtype=torch.float),
            "environment_feedback": environment_feedback,
            "done": done,
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": {
                "webshop_reward": torch.tensor(reward),
                "turn_count": torch.tensor(self.step_idx),
            },
        }

    async def _try_close_session(self):
        """Best-effort session cleanup."""
        if self.session_id is None:
            return
        try:
            session = await self._get_http_session()
            payload = {"session_id": self.session_id}
            async with session.post(f"{WEBSHOP_SERVER_URL}/close_session", json=payload) as resp:
                pass
        except Exception:
            pass
        finally:
            self.session_id = None
            await self._close_http_session()


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)