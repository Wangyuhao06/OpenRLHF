#!/usr/bin/env python3
"""
Evaluate memory constructor SFT model vs base model.

This script:
1. Loads test samples (demand-aware labeled)
2. Runs both base model and trained model on the same prompts
3. Compares memory decisions against ground truth labels
4. Computes write decision accuracy, precision, recall, F1
5. Computes retrieval hit rate for model-generated memories
"""

import argparse
import copy
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_constructor.data.memory_store import MemoryStore
from memory_constructor.data.schemas import MemoryItem, SFTSample
from memory_constructor.utils.json_utils import parse_json_robust
from memory_constructor.utils.logging_utils import LOG_FORMAT, setup_file_logging

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Same prompt template as sft_dataset.py for consistency
PROMPT_TEMPLATE = """You are a memory constructor for a shopping agent. You decide what to store
in the agent's memory to help it complete its task.

Agent Task:
{instruction}

Key Requirements:
{task_requirements}

Current Observation:
{observation}

Local History (last 3 steps):
{local_history}

Current Memory Store:
{memory_store}

Currently Retrievable (top matches for this observation):
{retrieval_context}

Budget Remaining: {budget_remaining}
Episode Progress: {episode_progress:.1%}

Decide: should you write a memory now? Consider:
- Does this observation contain NEW information not already in memory?
- Would this information help the agent in future steps?
- Is this information retrievable from existing memory?

Respond in JSON:
{{
    "write": true/false,
    "keys": ["key1", ...],
    "value": "..."
}}"""


def load_test_samples(data_path: str) -> List[SFTSample]:
    """Load SFTSample objects from JSONL."""
    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(SFTSample.from_dict(data))
    return samples


def format_prompt(sample: SFTSample) -> str:
    """Format prompt identically to MemoryConstructorSFTDataset."""
    # Local history
    if sample.local_history:
        history_text = "\n".join(sample.local_history)
    else:
        history_text = "(No previous steps)"

    # Memory store
    if sample.memory_store:
        memory_lines = []
        for i, mem in enumerate(sample.memory_store, 1):
            memory_lines.append(
                f"{i}. Keys: {', '.join(mem.keys)} | Value: {mem.value[:160]}"
            )
        memory_text = "\n".join(memory_lines)
    else:
        memory_text = "(Empty)"

    # Task requirements
    task_requirements = sample.metadata.get("task_requirements", [])
    if isinstance(task_requirements, list):
        lines = [f"- {item}" for item in task_requirements if str(item).strip()]
        task_req_text = "\n".join(lines) if lines else "(Not available)"
    elif isinstance(task_requirements, str) and task_requirements.strip():
        task_req_text = task_requirements
    else:
        task_req_text = "(Not available)"

    # Retrieval context
    retrieval_context = sample.metadata.get("retrieval_context", [])
    if isinstance(retrieval_context, list) and retrieval_context:
        rc_lines = []
        for item in retrieval_context:
            if not isinstance(item, dict):
                continue
            keys = ", ".join(item.get("keys", [])) or "(no keys)"
            score = item.get("score")
            score_text = f" | score={score}" if score is not None else ""
            value_preview = item.get("value_preview", "")
            rc_lines.append(
                f"- Step {item.get('step_id', '?')}: keys={keys}{score_text} | value={value_preview}"
            )
        retrieval_text = "\n".join(rc_lines) if rc_lines else "(Nothing retrievable yet)"
    else:
        retrieval_text = "(Nothing retrievable yet)"

    return PROMPT_TEMPLATE.format(
        instruction=sample.metadata.get("instruction", ""),
        task_requirements=task_req_text,
        observation=sample.observation,
        local_history=history_text,
        memory_store=memory_text,
        retrieval_context=retrieval_text,
        budget_remaining=sample.budget_remaining,
        episode_progress=sample.episode_progress,
    )


def run_model_on_samples(
    model,
    tokenizer,
    samples: List[SFTSample],
    max_length: int = 2048,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Run model on each sample and parse its output."""
    predictions = []
    model.eval()

    for sample in tqdm(samples, desc="Running model"):
        prompt = format_prompt(sample)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        parsed = parse_json_robust(
            generated_text,
            fallback={"write": False, "keys": [], "value": ""},
        )

        predictions.append({
            "sample_id": sample.sample_id,
            "trajectory_id": sample.trajectory_id,
            "step_id": sample.step_id,
            "pred_write": bool(parsed.get("write", False)),
            "pred_keys": parsed.get("keys", []),
            "pred_value": parsed.get("value", ""),
            "gt_write": sample.target_memory.write,
            "gt_keys": sample.target_memory.keys,
            "gt_value": sample.target_memory.value,
            "raw_output": generated_text[:500],
        })

    return predictions


def compute_write_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute write decision metrics: accuracy, precision, recall, F1."""
    tp = fp = tn = fn = 0
    for p in predictions:
        if p["pred_write"] and p["gt_write"]:
            tp += 1
        elif p["pred_write"] and not p["gt_write"]:
            fp += 1
        elif not p["pred_write"] and not p["gt_write"]:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "write_rate_pred": (tp + fp) / max(total, 1),
        "write_rate_gt": (tp + fn) / max(total, 1),
    }


def compute_key_similarity(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute key similarity when both predict and ground truth are write=True."""
    jaccard_scores = []
    for p in predictions:
        if p["pred_write"] and p["gt_write"]:
            pred_keys = set(k.lower() for k in p["pred_keys"])
            gt_keys = set(k.lower() for k in p["gt_keys"])
            if pred_keys or gt_keys:
                union = pred_keys | gt_keys
                intersection = pred_keys & gt_keys
                jaccard_scores.append(len(intersection) / len(union))

    if not jaccard_scores:
        return {"avg_key_jaccard": 0.0, "num_both_write": 0}

    return {
        "avg_key_jaccard": sum(jaccard_scores) / len(jaccard_scores),
        "num_both_write": len(jaccard_scores),
    }


def compute_retrieval_hit_rate(
    predictions: List[Dict[str, Any]],
    samples: List[SFTSample],
    retrieval_k: int = 3,
) -> Dict[str, float]:
    """
    Compute retrieval hit rate for model-predicted memories.

    Simulates the memory store using model predictions, then checks if
    predicted memories are retrievable at demanding steps.
    """
    from memory_constructor.models.retriever import HybridRetriever

    retriever = HybridRetriever(retrieval_k=retrieval_k)

    # Group by trajectory
    traj_preds = defaultdict(list)
    traj_samples = defaultdict(list)
    for p, s in zip(predictions, samples):
        traj_preds[p["trajectory_id"]].append(p)
        traj_samples[p["trajectory_id"]].append(s)

    total_queries = 0
    total_hits = 0

    for traj_id in traj_preds:
        preds = sorted(traj_preds[traj_id], key=lambda x: x["step_id"])
        samps = sorted(traj_samples[traj_id], key=lambda x: x.step_id)

        # Build accumulated memory store from predictions
        accumulated = []
        for pred in preds:
            if pred["pred_write"] and pred["pred_keys"]:
                mem = MemoryItem(
                    write=True,
                    keys=pred["pred_keys"],
                    value=pred["pred_value"],
                    timestamp=float(pred["step_id"]),
                    step_id=pred["step_id"],
                    metadata={},
                )
                accumulated.append(mem)

        # Test retrieval at each step that has memories before it
        for i, (pred, samp) in enumerate(zip(preds, samps)):
            prior_mems = [m for m in accumulated if m.step_id < pred["step_id"]]
            if not prior_mems:
                continue

            store = MemoryStore()
            for m in prior_mems:
                store.add(copy.deepcopy(m))

            results = retriever.retrieve(
                samp.observation, store, k=retrieval_k, return_scores=True,
            )
            retrieved_ids = {m.step_id for m, _ in results}

            # Check if any prior memory that was supposed to be relevant is retrieved
            for m in prior_mems:
                demanded_by = m.metadata.get("demanded_by", [])
                if pred["step_id"] in demanded_by:
                    total_queries += 1
                    if m.step_id in retrieved_ids:
                        total_hits += 1

    hit_rate = total_hits / max(total_queries, 1)
    return {
        "retrieval_queries": total_queries,
        "retrieval_hits": total_hits,
        "retrieval_hit_rate": hit_rate,
    }


def evaluate_model(
    model_path: str,
    test_samples: List[SFTSample],
    model_name: str,
    max_length: int = 2048,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Full evaluation of a single model."""
    logger.info(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Running {model_name} on {len(test_samples)} samples...")
    predictions = run_model_on_samples(
        model, tokenizer, test_samples, max_length=max_length, device=device,
    )

    write_metrics = compute_write_metrics(predictions)
    key_sim = compute_key_similarity(predictions)

    # Free GPU memory before retrieval computation
    del model
    torch.cuda.empty_cache()

    logger.info("=" * 70)
    logger.info(f"  {model_name} Results")
    logger.info("=" * 70)
    logger.info(f"  Write Decision:")
    logger.info(f"    Accuracy:  {write_metrics['accuracy']:.2%}")
    logger.info(f"    Precision: {write_metrics['precision']:.2%}")
    logger.info(f"    Recall:    {write_metrics['recall']:.2%}")
    logger.info(f"    F1:        {write_metrics['f1']:.2%}")
    logger.info(f"    TP={write_metrics['tp']} FP={write_metrics['fp']} TN={write_metrics['tn']} FN={write_metrics['fn']}")
    logger.info(f"    Pred write rate: {write_metrics['write_rate_pred']:.2%}, GT write rate: {write_metrics['write_rate_gt']:.2%}")
    logger.info(f"  Key Similarity:")
    logger.info(f"    Avg Jaccard: {key_sim['avg_key_jaccard']:.4f} (n={key_sim['num_both_write']})")
    logger.info("=" * 70)

    return {
        "model_name": model_name,
        "model_path": model_path,
        "num_samples": len(test_samples),
        "write_metrics": write_metrics,
        "key_similarity": key_sim,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model vs base model")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--trained_model", type=str, required=True, help="Trained model path")
    parser.add_argument("--test_data", type=str, required=True, help="Test JSONL path")
    parser.add_argument("--output_file", type=str, default="results/sft_eval.json")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    setup_file_logging("evaluate_sft", project_root)

    logger.info("=" * 80)
    logger.info("Memory Constructor SFT Evaluation")
    logger.info("=" * 80)

    # Load test data
    logger.info(f"Loading test data from {args.test_data}...")
    test_samples = load_test_samples(args.test_data)
    if args.max_samples:
        test_samples = test_samples[:args.max_samples]
    logger.info(f"Loaded {len(test_samples)} test samples")

    gt_writes = sum(1 for s in test_samples if s.target_memory.write)
    logger.info(f"Ground truth: {gt_writes} writes, {len(test_samples) - gt_writes} no-writes")

    # Evaluate base model
    logger.info("\n[1/2] Evaluating base model...")
    base_results = evaluate_model(
        args.base_model, test_samples, "Base Model",
        max_length=args.max_length, device=args.device,
    )

    # Evaluate trained model
    logger.info("\n[2/2] Evaluating trained model...")
    trained_results = evaluate_model(
        args.trained_model, test_samples, "Trained Model",
        max_length=args.max_length, device=args.device,
    )

    # Print comparison
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Base vs Trained")
    logger.info("=" * 80)
    bw = base_results["write_metrics"]
    tw = trained_results["write_metrics"]
    for metric in ["accuracy", "precision", "recall", "f1"]:
        delta = tw[metric] - bw[metric]
        arrow = "+" if delta >= 0 else ""
        logger.info(
            f"  {metric:>10s}: Base={bw[metric]:.2%}  Trained={tw[metric]:.2%}  ({arrow}{delta:.2%})"
        )
    bk = base_results["key_similarity"]
    tk = trained_results["key_similarity"]
    delta_j = tk["avg_key_jaccard"] - bk["avg_key_jaccard"]
    arrow_j = "+" if delta_j >= 0 else ""
    logger.info(
        f"  {'key_jaccard':>10s}: Base={bk['avg_key_jaccard']:.4f}  Trained={tk['avg_key_jaccard']:.4f}  ({arrow_j}{delta_j:.4f})"
    )
    logger.info("=" * 80)

    # Save results (without raw predictions to keep file small)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "base_model": {k: v for k, v in base_results.items() if k != "predictions"},
        "trained_model": {k: v for k, v in trained_results.items() if k != "predictions"},
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
