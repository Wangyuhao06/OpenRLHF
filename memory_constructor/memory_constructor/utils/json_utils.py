"""
JSON utilities for robust parsing and validation.

This module provides utilities for:
- Robust JSON parsing with fallback
- Validation against schemas
- Truncation enforcement for memory items
"""

import json
import re
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def _fix_json_string(text: str) -> str:
    """Fix common LLM JSON generation issues."""
    # Fix invalid escape sequences (LLMs generate \$ \| \- etc.)
    text = re.sub(r'\\([^"\\\/bfnrtu])', r'\1', text)
    # Fix unescaped newlines inside strings: find string content and escape newlines
    # This is a simplified approach - replace literal newlines within JSON string values
    text = re.sub(r'(?<=": ")([^"]*)\n([^"]*")', lambda m: m.group(0).replace('\n', '\\n'), text)
    return text


def parse_json_robust(text: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Robustly parse JSON from text with fallback.

    Args:
        text: Text containing JSON
        fallback: Fallback dict if parsing fails

    Returns:
        Parsed JSON dict or fallback
    """
    if fallback is None:
        fallback = {"write": False, "keys": [], "value": ""}

    # Strip Qwen3 <think>...</think> blocks (model outputs thinking before JSON)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Also strip incomplete thinking blocks (model may still be thinking at end)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL).strip()
    # Strip bare </think> tag (when prompt already included <think>\n</think>)
    text = re.sub(r'</think>', '', text).strip()

    # Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass

    # Try to extract JSON object by brace counting (handles nested braces in values)
    start = text.find('{')
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if not in_string:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            # Try fixing common LLM JSON issues
                            fixed = _fix_json_string(candidate)
                            try:
                                parsed = json.loads(fixed)
                                if isinstance(parsed, dict):
                                    return parsed
                            except json.JSONDecodeError:
                                pass
                            break

    # Legacy regex fallback for simple cases
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    logger.warning(f"Failed to parse JSON from text: {text[:100]}...")
    return fallback


def validate_memory_item(
    memory_item: Dict[str, Any],
    max_key_tokens: int = 256,
    max_value_tokens: int = 512,
    max_num_keys: int = 8,
) -> Dict[str, Any]:
    """
    Validate and fix memory item structure.

    Args:
        memory_item: Memory item dict
        max_key_tokens: Maximum tokens per key
        max_value_tokens: Maximum tokens for value
        max_num_keys: Maximum number of keys

    Returns:
        Validated and fixed memory item
    """
    # Ensure required fields
    if "write" not in memory_item:
        memory_item["write"] = False

    if "keys" not in memory_item:
        memory_item["keys"] = []

    if "value" not in memory_item:
        memory_item["value"] = ""

    # Convert to correct types
    memory_item["write"] = bool(memory_item["write"])

    if not isinstance(memory_item["keys"], list):
        memory_item["keys"] = []

    if not isinstance(memory_item["value"], str):
        memory_item["value"] = str(memory_item["value"])

    # Truncate keys
    memory_item["keys"] = memory_item["keys"][:max_num_keys]
    memory_item["keys"] = [
        truncate_text(key, max_key_tokens) for key in memory_item["keys"]
    ]

    # Truncate value
    memory_item["value"] = truncate_text(memory_item["value"], max_value_tokens)

    return memory_item


def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncate text to maximum number of tokens (words).

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens

    Returns:
        Truncated text
    """
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def enforce_memory_constraints(
    memory_item: Dict[str, Any],
    max_key_tokens: int = 256,
    max_value_tokens: int = 512,
    max_num_keys: int = 8,
    log_truncation: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Enforce memory constraints and log truncation statistics.

    Args:
        memory_item: Memory item dict
        max_key_tokens: Maximum tokens per key
        max_value_tokens: Maximum tokens for value
        max_num_keys: Maximum number of keys
        log_truncation: Whether to log truncation stats

    Returns:
        Tuple of (validated memory item, truncation stats)
    """
    stats = {
        "keys_truncated": 0,
        "value_truncated": False,
        "num_keys_truncated": False,
        "original_num_keys": len(memory_item.get("keys", [])),
        "original_value_length": len(memory_item.get("value", "").split()),
    }

    # Validate first
    memory_item = validate_memory_item(
        memory_item, max_key_tokens, max_value_tokens, max_num_keys
    )

    # Check truncation
    if len(memory_item["keys"]) < stats["original_num_keys"]:
        stats["num_keys_truncated"] = True

    for i, key in enumerate(memory_item["keys"]):
        original_length = len(memory_item.get("keys", [])[i].split()) if i < stats["original_num_keys"] else 0
        if len(key.split()) < original_length:
            stats["keys_truncated"] += 1

    if len(memory_item["value"].split()) < stats["original_value_length"]:
        stats["value_truncated"] = True

    if log_truncation and (stats["keys_truncated"] > 0 or stats["value_truncated"] or stats["num_keys_truncated"]):
        logger.info(f"Truncation stats: {stats}")

    return memory_item, stats


def format_memory_prompt(
    observation: str,
    local_history: List[str],
    memory_store: List[Dict[str, Any]],
    budget_remaining: int,
    episode_progress: float,
    max_history_length: int = 3,
) -> str:
    """
    Format prompt for memory constructor.

    Args:
        observation: Current observation
        local_history: Recent history
        memory_store: Current memories
        budget_remaining: Remaining budget
        episode_progress: Progress through episode (0-1)
        max_history_length: Maximum history items to include

    Returns:
        Formatted prompt string
    """
    # Truncate history
    history = local_history[-max_history_length:] if local_history else []

    # Format memories
    memory_text = ""
    if memory_store:
        memory_text = "\n".join([
            f"Memory {i+1}: keys={m.get('keys', [])}, value={m.get('value', '')[:100]}..."
            for i, m in enumerate(memory_store[-5:])  # Show last 5 memories
        ])
    else:
        memory_text = "No memories yet."

    prompt = f"""You are a memory constructor for a WebShop agent. Your task is to decide whether to write a memory item and what to write.

Current Observation:
{observation}

Recent History:
{chr(10).join(history) if history else "No history yet."}

Current Memory Store:
{memory_text}

Budget Remaining: {budget_remaining}
Episode Progress: {episode_progress:.1%}

Output a JSON object with the following structure:
{{
  "write": true/false,
  "keys": ["key1", "key2", ...],
  "value": "compressed memory content"
}}

If write=false, set keys=[] and value="".
If write=true, provide multiple relevant keys and a compressed value.

JSON Output:"""

    return prompt


def parse_constructor_output(text: str) -> Dict[str, Any]:
    """
    Parse constructor output with robust fallback.

    Args:
        text: Constructor output text

    Returns:
        Parsed memory item dict
    """
    memory_item = parse_json_robust(text)
    memory_item = validate_memory_item(memory_item)
    return memory_item
