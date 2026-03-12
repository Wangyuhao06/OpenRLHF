"""
Candidate sampler for best-of-n training.

This module uses vLLM to efficiently generate multiple candidate memories
per step from the SFT-trained constructor model.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from vllm import LLM, SamplingParams

from memory_constructor.data.schemas import SFTSample, CandidateSample
from memory_constructor.utils.json_utils import parse_json_robust, validate_memory_item

logger = logging.getLogger(__name__)


class CandidateSampler:
    """
    Candidate sampler using vLLM for fast batch inference.

    Generates N candidate memories per step using nucleus sampling
    for diversity.
    """

    def __init__(
        self,
        model_path: str,
        num_candidates: int = 4,
        temperature: float = 0.9,
        top_p: float = 0.9,
        max_tokens: int = 512,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize candidate sampler.

        Args:
            model_path: Path to SFT-trained model checkpoint
            num_candidates: Number of candidates to generate per sample
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0-1)
        """
        self.model_path = model_path
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Initialize vLLM
        logger.info(f"Loading model from {model_path}...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=num_candidates,  # Generate N candidates per prompt
        )

        logger.info(
            f"Initialized CandidateSampler with {num_candidates} candidates per sample"
        )

    def _format_prompt(self, sample: SFTSample) -> str:
        """
        Format prompt for candidate generation.

        Args:
            sample: SFTSample object

        Returns:
            Formatted prompt string
        """
        # Format local history
        if sample.local_history:
            history_text = "\n".join(sample.local_history)
        else:
            history_text = "(No previous steps)"

        # Format memory store
        if sample.memory_store:
            memory_lines = []
            for i, mem in enumerate(sample.memory_store, 1):
                memory_lines.append(
                    f"{i}. Keys: {', '.join(mem.keys)} | Value: {mem.value[:100]}..."
                )
            memory_text = "\n".join(memory_lines)
        else:
            memory_text = "(Empty)"

        # Create prompt
        prompt = f"""You are a memory constructor for a shopping agent. Your task is to decide whether to write a memory at this step, and if so, what keys and value to use.

Task: {sample.metadata.get('instruction', '')}

Current Observation:
{sample.observation}

Local History (last 3 steps):
{history_text}

Current Memory Store:
{memory_text}

Budget Remaining: {sample.budget_remaining}
Episode Progress: {sample.episode_progress:.1%}

Should you write a memory at this step? If yes, provide:
1. Keys: Searchable keywords (max 4 keys, each max 64 tokens)
2. Value: Compressed, faithful summary (max 256 tokens)

Respond in JSON format:
{{
    "write": true/false,
    "keys": ["key1", "key2", ...],
    "value": "compressed memory content"
}}"""

        return prompt

    def sample_candidates(
        self, samples: List[SFTSample]
    ) -> List[CandidateSample]:
        """
        Generate candidates for a list of samples.

        Args:
            samples: List of SFTSample objects

        Returns:
            List of CandidateSample objects
        """
        logger.info(f"Generating candidates for {len(samples)} samples...")

        # Format prompts
        prompts = [self._format_prompt(sample) for sample in samples]

        # Generate with vLLM (batch inference)
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Parse outputs
        candidate_lists = []

        for i, (sample, output) in enumerate(zip(samples, outputs)):
            # Parse each candidate
            candidates = []

            for j, gen_output in enumerate(output.outputs):
                candidate_text = gen_output.text.strip()

                # Parse JSON
                candidate_dict = parse_json_robust(candidate_text)
                candidate_dict = validate_memory_item(
                    candidate_dict,
                    max_key_tokens=64,
                    max_value_tokens=256,
                    max_num_keys=4,
                )

                # Create Dict[str, Any] object (scores will be filled by scorer)
                candidate = Dict[str, Any](
                    candidate_id=j,
                    memory_item={
                        "write": candidate_dict.get("write", False),
                        "keys": candidate_dict.get("keys", []),
                        "value": candidate_dict.get("value", ""),
                        "timestamp": sample.target_memory.timestamp,
                        "step_id": sample.step_id,
                        "metadata": {},
                    },
                    hindsight_score=0.0,  # To be filled by scorer
                    counterfactual_score=0.0,
                    task_success_score=0.0,
                    retrieval_usefulness=0.0,
                    compactness_score=0.0,
                    redundancy_score=0.0,
                    faithfulness_score=0.0,
                    total_score=0.0,
                    future_retrieval_hits=0,
                    future_impact_steps=[],
                    sampled_perspectives=[],
                )

                candidates.append(candidate)

            # Always add no_write as a candidate
            no_write_candidate = Dict[str, Any](
                candidate_id=len(candidates),
                memory_item={
                    "write": False,
                    "keys": [],
                    "value": "",
                    "timestamp": sample.target_memory.timestamp,
                    "step_id": sample.step_id,
                    "metadata": {"type": "no_write"},
                },
                hindsight_score=0.0,
                counterfactual_score=0.0,
                task_success_score=0.0,
                retrieval_usefulness=0.0,
                compactness_score=1.0,  # Saves budget
                redundancy_score=1.0,  # No redundancy
                faithfulness_score=1.0,  # No hallucination risk
                total_score=0.0,
                future_retrieval_hits=0,
                future_impact_steps=[],
                sampled_perspectives=["no_write"],
            )
            candidates.append(no_write_candidate)

            # Create CandidateSample
            candidate_list = CandidateSample(
                sample_id=sample.sample_id,
                trajectory_id=sample.trajectory_id,
                step_id=sample.step_id,
                observation=sample.observation,
                local_history=sample.local_history,
                memory_store=sample.memory_store,
                budget_remaining=sample.budget_remaining,
                episode_progress=sample.episode_progress,
                candidates=candidates,
                best_candidate_idx=-1,  # To be filled by scorer
                selection_method="",
                metadata=sample.metadata,
            )

            candidate_lists.append(candidate_list)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(samples)} samples")

        logger.info(f"Generated {len(candidate_lists)} candidate lists")

        return candidate_lists

    def sample_from_file(
        self, input_path: str, output_path: str, max_samples: Optional[int] = None
    ):
        """
        Generate candidates from input file and save to output file.

        Args:
            input_path: Path to input JSONL file with SFTSample objects
            output_path: Path to output JSONL file for CandidateSample objects
            max_samples: Maximum number of samples to process (for testing)
        """
        # Load samples
        logger.info(f"Loading samples from {input_path}...")
        samples = []

        with open(input_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and len(samples) >= max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    sample = SFTSample.from_dict(data)
                    samples.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(samples)} samples")

        # Generate candidates
        candidate_lists = self.sample_candidates(samples)

        # Save to file
        logger.info(f"Saving candidates to {output_path}...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for candidate_list in candidate_lists:
                # Convert to dict
                candidate_dict = asdict(candidate_list)
                # Write as JSON line
                f.write(json.dumps(candidate_dict) + "\n")

        logger.info(f"Saved {len(candidate_lists)} candidate lists to {output_path}")
