#!/usr/bin/env python3
"""
Script 07: Train memory constructor with RL (REINFORCE baseline).

Launches OpenRLHF's PPO/REINFORCE training with the memory constructor
AgentInstance. Requires:
  1. WebShop pool server running (see interaction_env/.../scripts/)
  2. Frozen agent vLLM server running (Qwen3-8B on separate machine)
  3. Ray cluster started

Usage:
    # Set environment variables first:
    export WEBSHOP_SERVER_URL="http://localhost:6001"
    export FROZEN_AGENT_URL="http://<separate-machine>:8000/v1"
    export FROZEN_AGENT_MODEL="Qwen/Qwen3-8B"

    # Run via Ray:
    ray job submit --address="http://127.0.0.1:8265" -- \\
        python memory_constructor/scripts/07_train_rl.py

    # Or directly:
    python memory_constructor/scripts/07_train_rl.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # OpenRLHF root
MC_ROOT = Path(__file__).resolve().parent.parent  # memory_constructor root
AGENT_FUNC_PATH = MC_ROOT / "scripts" / "agent_func_memory_constructor.py"
DEFAULT_PRETRAIN = MC_ROOT / "checkpoints" / "sft_qwen3_8b_v2" / "checkpoints" / "global_step280_hf"
DEFAULT_PROMPT_DATA = MC_ROOT / "data" / "webshop" / "rl_prompts.jsonl"
DEFAULT_SAVE_PATH = MC_ROOT / "checkpoints" / "rl_qwen3_8b"


def parse_args():
    parser = argparse.ArgumentParser(description="Train memory constructor with RL")

    # Model & data
    parser.add_argument("--pretrain", type=str, default=str(DEFAULT_PRETRAIN))
    parser.add_argument("--prompt_data", type=str, default=str(DEFAULT_PROMPT_DATA))
    parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE_PATH))

    # RL hyperparameters
    parser.add_argument("--advantage_estimator", type=str, default="reinforce_baseline")
    parser.add_argument("--n_samples_per_prompt", type=int, default=4)
    parser.add_argument("--rollout_batch_size", type=int, default=16)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=2048)
    parser.add_argument("--generate_max_len", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=4000)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--actor_learning_rate", type=float, default=3e-8)
    parser.add_argument("--eps_clip", type=float, default=0.1)
    parser.add_argument("--max_norm", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)

    # GPU layout
    parser.add_argument("--actor_num_gpus", type=int, default=6)
    parser.add_argument("--ref_num_gpus", type=int, default=6)
    parser.add_argument("--vllm_num_engines", type=int, default=3)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=2)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.5)

    # ZeRO stage (2 = partitioned grads+optimizer; use 2 for single-GPU to avoid unnecessary overhead)
    parser.add_argument("--zero_stage", type=int, default=2)

    # Checkpointing
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--max_ckpt_num", type=int, default=5)

    # Logging
    parser.add_argument("--tensorboard_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate prerequisites
    if not Path(args.pretrain).exists():
        print(f"ERROR: Pretrain model not found: {args.pretrain}")
        sys.exit(1)
    if not AGENT_FUNC_PATH.exists():
        print(f"ERROR: Agent func not found: {AGENT_FUNC_PATH}")
        sys.exit(1)
    if not Path(args.prompt_data).exists():
        print(f"ERROR: Prompt data not found: {args.prompt_data}")
        print("Run 07a_prepare_rl_prompts.py first.")
        sys.exit(1)

    # Check env vars
    for var in ["WEBSHOP_SERVER_URL", "FROZEN_AGENT_URL"]:
        if var not in os.environ:
            print(f"WARNING: {var} not set, using default.")

    tensorboard_dir = args.tensorboard_dir or str(
        MC_ROOT / "results" / "rl_runs"
    )
    ckpt_path = str(Path(args.save_path) / "ckpt")

    cmd = [
        sys.executable, "-m", "openrlhf.cli.train_ppo_ray",
        # Model
        "--pretrain", args.pretrain,
        "--load_checkpoint",
        "--save_path", args.save_path,
        "--ckpt_path", ckpt_path,
        "--save_hf_ckpt",
        "--max_ckpt_num", str(args.max_ckpt_num),
        "--save_steps", str(args.save_steps),
        # Agent function
        "--agent_func_path", str(AGENT_FUNC_PATH),
        # Data
        "--prompt_data", args.prompt_data,
        "--input_key", "prompt",
        "--label_key", "label",
        "--prompt_max_len", str(args.prompt_max_len),
        "--generate_max_len", str(args.generate_max_len),
        # NOTE: --packing_samples removed — causes NaN in bf16 log_softmax on long packed sequences
        # RL config
        "--advantage_estimator", args.advantage_estimator,
        "--n_samples_per_prompt", str(args.n_samples_per_prompt),
        "--rollout_batch_size", str(args.rollout_batch_size),
        "--train_batch_size", str(args.train_batch_size),
        "--micro_train_batch_size", str(args.micro_train_batch_size),
        "--micro_rollout_batch_size", str(args.micro_rollout_batch_size),
        "--max_samples", str(args.max_samples),
        "--max_epochs", str(args.max_epochs),
        "--num_episodes", str(args.num_episodes),
        # Engine — synchronous mode, NO colocate to avoid all NCCL conflicts
        # Each group (actor, ref, vLLM) gets its own GPUs
        "--actor_num_nodes", "1",
        "--actor_num_gpus_per_node", str(args.actor_num_gpus),
        "--ref_num_nodes", "1",
        "--ref_num_gpus_per_node", str(args.ref_num_gpus),
        "--vllm_num_engines", str(args.vllm_num_engines),
        "--vllm_tensor_parallel_size", str(args.vllm_tensor_parallel_size),
        "--vllm_gpu_memory_utilization", str(args.vllm_gpu_memory_utilization),
        "--vllm_sync_backend", "nccl",
        "--enforce_eager",
        # Training config — ZeRO-2 for single GPU (avoids AllGather deadlocks)
        "--zero_stage", str(args.zero_stage),
        "--adam_offload",
        "--gradient_checkpointing",
        "--param_dtype", "bf16",
        "--ref_reward_offload",
        "--init_kl_coef", "0.01",
        "--use_kl_loss",
        "--eps_clip", str(args.eps_clip),
        "--max_norm", str(args.max_norm),
        "--actor_learning_rate", str(args.actor_learning_rate),
        "--train_max_tokens_per_gpu", "2048",
        "--repetition_penalty", str(args.repetition_penalty),
        # Logging
        "--use_tensorboard", tensorboard_dir,
        "--logging_steps", "1",
        "--eval_steps", "-1",
    ]

    print("=" * 80)
    print("Memory Constructor RL Training")
    print("=" * 80)
    print(f"Pretrain:      {args.pretrain}")
    print(f"Agent func:    {AGENT_FUNC_PATH}")
    print(f"Prompt data:   {args.prompt_data}")
    print(f"Save path:     {args.save_path}")
    print(f"WebShop URL:   {os.environ.get('WEBSHOP_SERVER_URL', '(default)')}")
    print(f"Frozen agent:  {os.environ.get('FROZEN_AGENT_URL', '(default)')}")
    print(f"Frozen model:  {os.environ.get('FROZEN_AGENT_MODEL', '(default)')}")
    print(f"Advantage:     {args.advantage_estimator}")
    print(f"ZeRO stage:    {args.zero_stage}")
    print(f"Batch sizes:   rollout={args.rollout_batch_size}, train={args.train_batch_size}")
    print(f"GPU layout:    actor={args.actor_num_gpus}, ref={args.ref_num_gpus}, "
          f"vllm={args.vllm_num_engines}xTP{args.vllm_tensor_parallel_size}")
    print("=" * 80)
    print(f"\nCommand:\n{' '.join(cmd)}\n")

    # Execute
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
