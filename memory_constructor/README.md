# Memory Constructor for WebShop

A trainable memory constructor that learns to compress and store information for a fixed downstream agent in the WebShop environment. The constructor observes agent-environment interactions and decides **when** to write memory, **what keys** to index by, and **what value** to store — forming an append-only, retrieval-augmented memory system.

## Architecture Overview

```
                    ┌──────────────────────────┐
                    │     WebShop Environment   │
                    └────────┬─────────────────┘
                             │ observation
                             ▼
┌─────────────────────────────────────────────────────────┐
│                Memory Constructor (Trainable)            │
│  Input:  observation, local_history, memory_store,       │
│          retrieval_context, budget, progress              │
│  Output: {"write": bool, "keys": [...], "value": "..."}  │
└──────────────────────┬──────────────────────────────────┘
                       │ write to / retrieve from
                       ▼
              ┌─────────────────┐
              │  Memory Store   │  (append-only, budget-limited)
              │  + Retriever    │  (BM25 + Dense hybrid)
              └────────┬────────┘
                       │ top-k memories
                       ▼
              ┌─────────────────┐
              │  Frozen Agent   │  (Qwen3-8B, via vLLM API)
              │  takes action   │
              └─────────────────┘
```

## Project Structure

```
memory_constructor/
├── configs/
│   ├── model.yaml                # Model hyperparameters (base_model, max_length, dtype)
│   ├── training.yaml             # Training settings (batch size, LR, epochs)
│   └── data.yaml                 # Data paths and preprocessing parameters
├── data/webshop/
│   ├── raw/                      # Raw trajectory JSONL files
│   ├── processed/                # Hindsight-labeled SFTSample train/val/test splits
│   ├── processed_demand_aware/   # Demand-aware variant
│   ├── candidates/               # Best-of-N generated & scored candidates
│   └── rl_prompts.jsonl          # RL task prompt dataset
├── checkpoints/
│   ├── sft_qwen3_8b_v2/          # SFT checkpoint (Qwen3-8B, step 280)
│   ├── sft_demand_aware_qwen3_8b/# Demand-aware SFT variant
│   └── rl_qwen3_8b/              # RL-trained checkpoint
├── results/                      # Evaluation results & TensorBoard logs
├── memory_constructor/           # Python package (see below)
├── scripts/                      # Pipeline scripts (see below)
└── requirements.txt
```

## Training Pipeline

The pipeline has three progressive stages: **SFT → Best-of-N → RL**.

```
Raw Trajectories ──► 01_preprocess ──► SFT Samples
                                           │
                                     02_train_sft ──► SFT Checkpoint
                                           │
                    03_generate_candidates ─┤
                                           │
                    04_score_candidates ────┤
                                           │
                    05_train_best_of_n ─────► BoN Checkpoint
                                           │
                    07a_prepare_rl_prompts ─┤
                                           │
                    07_train_rl ────────────► RL Checkpoint
                                           │
                    06_evaluate ────────────► Metrics
```

### Prerequisites

```bash
# 1. OpenRLHF venv (managed by uv)
source /path/to/OpenRLHF/.venv/bin/activate

# 2. HF mirror (if huggingface.co is unreachable)
export HF_ENDPOINT=https://hf-mirror.com

# 3. Install retrieval dependencies
pip install rank-bm25 sentence-transformers faiss-cpu

# 4. For RL training: WebShop server + Java for Lucene
export JAVA_HOME=/path/to/jdk
```

---

### Step 1: Preprocess WebShop Trajectories (`01_preprocess_webshop.py`)

Parse raw agent trajectories and apply **hindsight labeling** to generate supervised training samples.

```bash
python scripts/01_preprocess_webshop.py \
    --input_files /path/to/trajectories/*.jsonl \
    --output_dir data/webshop/processed \
    --labeling_method demand_aware \
    --contribution_threshold 0.3
```

**Input**: Raw trajectory JSONL files (agent interactions with WebShop)
**Output**: `train.jsonl`, `val.jsonl`, `test.jsonl` (SFTSample format) + `metadata.json`

**Labeling methods**:
| Method | Description |
|--------|-------------|
| `heuristic` | Rule-based (fast, low quality) |
| `llm_judge` | LLM decides per-step whether to write memory |
| `demand_aware` | 4-phase: backward demand analysis → global plan → retrieval verification → assembly |

**Key arguments**:
- `--labeling_method {heuristic, llm_judge, demand_aware}`
- `--contribution_threshold` — minimum contribution score to write (default: 0.3)
- `--budget_ratio` — memory budget = steps × ratio (default: 2.0)
- `--max_value_tokens`, `--max_key_tokens`, `--max_num_keys` — memory size limits
- `--train_ratio / --val_ratio / --test_ratio` — split fractions (default: 0.7/0.15/0.15)

**Dependencies**: `WebShopParser`, `HindsightLabeler` / `DemandAwareHindsightLabeler`, `HybridRetriever`

---

### Step 2: Train SFT (`02_train_sft.py`)

Supervised fine-tuning on labeled samples using OpenRLHF's distributed trainer with DeepSpeed.

```bash
python scripts/02_train_sft.py \
    --pretrain Qwen/Qwen3-8B \
    --train_data data/webshop/processed/train.jsonl \
    --output_dir checkpoints/sft \
    --batch_size 64 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --zero_stage 2
```

**Input**: `train.jsonl` from Step 1, base model (e.g., `Qwen/Qwen3-8B`)
**Output**: Checkpoints at `checkpoints/sft/checkpoints/global_step<N>_hf/`

**Key arguments**:
- `--pretrain` — base model name or path
- `--zero_stage {0,1,2,3}` — DeepSpeed ZeRO stage (use 3 for 8B on 32GB GPUs)
- `--write_loss_weight` — upweight write=True samples (combats no-write bias)
- `--oversample_write` — repeat write=True samples N times
- `--adam_offload` — offload optimizer to CPU (saves GPU memory)
- `--save_steps` — checkpoint frequency

**Dependencies**: OpenRLHF (`SFTTrainer`, `Actor`, `get_strategy`), DeepSpeed, `MemoryConstructorSFTDataset`

---

### Step 3: Generate Candidates (`03_generate_candidates.py`)

Generate N diverse candidate memories per sample using vLLM for fast batch inference.

```bash
python scripts/03_generate_candidates.py \
    --model_path checkpoints/sft/checkpoints/global_step280_hf \
    --input_file data/webshop/processed/train.jsonl \
    --output_file data/webshop/candidates/train_candidates.jsonl \
    --num_candidates 4 \
    --temperature 0.9
```

**Input**: SFT checkpoint + training samples
**Output**: `train_candidates.jsonl` (CandidateMemoryList format)

**Key arguments**:
- `--num_candidates` — candidates per sample (default: 4)
- `--temperature`, `--top_p` — sampling parameters
- `--tensor_parallel_size` — number of GPUs for vLLM
- `--gpu_memory_utilization` — vLLM GPU usage fraction

**Dependencies**: `CandidateSampler`, vLLM

---

### Step 4: Score Candidates (`04_score_candidates.py`)

Score candidates using counterfactual simulation + hindsight evaluation + quality metrics.

```bash
python scripts/04_score_candidates.py \
    --input_file data/webshop/candidates/train_candidates.jsonl \
    --output_file data/webshop/candidates/train_scored.jsonl
```

**Input**: `train_candidates.jsonl` from Step 3
**Output**: `train_scored.jsonl` with scores and `best_candidate_idx`

**Scoring components** (weighted combination):
| Component | Weight | Description |
|-----------|--------|-------------|
| Counterfactual | 0.5 | Simulates future retrieval — would this memory be found? |
| Hindsight | 0.5 | Was this memory actually retrieved in the trajectory? |
| Compactness | 0.1 | Shorter values preferred |
| Redundancy | 0.1 | Penalizes overlap with existing memories |
| Faithfulness | 0.1 | Grounded in the actual observation |

**Dependencies**: `CandidateScorer`, `HybridRetriever`

---

### Step 5: Train Best-of-N (`05_train_best_of_n.py`)

Train on the highest-scored candidate per sample.

```bash
python scripts/05_train_best_of_n.py \
    --model_path checkpoints/sft/checkpoints/global_step280_hf \
    --train_file data/webshop/candidates/train_scored.jsonl \
    --output_dir checkpoints/best_of_n \
    --learning_rate 5e-6
```

**Input**: Scored candidates from Step 4, SFT checkpoint as base
**Output**: Best-of-N trained checkpoint

**Dependencies**: `BestOfNTrainer`, optional DeepSpeed

---

### Step 6: Evaluate (`06_evaluate.py`)

Evaluate a trained checkpoint on test data.

```bash
python scripts/06_evaluate.py \
    --model_path checkpoints/sft_qwen3_8b_v2/checkpoints/global_step280_hf \
    --test_file data/webshop/processed/test.jsonl \
    --output_file results/evaluation.json
```

**Input**: Trained checkpoint + test samples
**Output**: `evaluation.json` with metrics

**Metrics**: accuracy, precision, recall, F1, key Jaccard, write rate, retrieval quality

**Dependencies**: `Evaluator`, `MetricsCalculator`, `HybridRetriever`

---

### Step 7: RL Training (`07_train_rl.py` + `07a_prepare_rl_prompts.py`)

Online RL training with dynamic WebShop interaction. The memory constructor is trained with REINFORCE while a frozen agent uses its memories to act.

#### 7a. Prepare RL Prompts

```bash
python scripts/07a_prepare_rl_prompts.py \
    --output data/webshop/rl_prompts.jsonl \
    --num_tasks 500
```

**Output**: `rl_prompts.jsonl` — `{"prompt": "Task <idx>", "label": "<idx>"}` per line

#### 7b. Launch RL Training

The all-in-one launcher starts WebShop server, frozen agent vLLM, and training:

```bash
bash scripts/launch_rl_training.sh          # full run
bash scripts/launch_rl_training.sh --test   # small test run
```

Or manually:

```bash
# 1. Start WebShop pool server (requires Java + Lucene)
python interaction_env/.../runner/webshop_server_pool.py --port 6001

# 2. Start frozen agent vLLM (on separate GPU)
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --port 8000

# 3. Set environment variables
export WEBSHOP_SERVER_URL=http://localhost:6001
export FROZEN_AGENT_URL=http://localhost:8000/v1
export FROZEN_AGENT_MODEL=Qwen/Qwen3-8B
export FROZEN_AGENT_API_KEY=EMPTY

# 4. Run training
python scripts/07_train_rl.py \
    --pretrain checkpoints/sft_qwen3_8b_v2/checkpoints/global_step280_hf \
    --actor_num_gpus 4 --vllm_num_engines 1 \
    --rollout_batch_size 12 --train_batch_size 12
```

**Input**: SFT checkpoint, RL prompts, running WebShop server, running frozen agent
**Output**: RL checkpoint at `checkpoints/rl_qwen3_8b/`, TensorBoard logs at `results/rl_runs/`

**Reward design**:

| Signal | Value | Trigger |
|--------|-------|---------|
| No-write penalty | -0.1 | `write=false` (biases toward writing) |
| Write cost | -0.05 | `write=true` |
| Format error | -3 | Invalid JSON output |
| Long memory | -0.1 × max(0, len/128 - 1) | Value exceeds 128 tokens |
| Retrieval hit (episode end) | +0.2 × min(count, 3) per memory | Memory was retrieved ≥1 times |
| Never retrieved (episode end) | -0.3 per memory | Written but never used |
| Task success (episode end) | is_success × (0.5 × mem/steps + 0.5) | WebShop reward > 0 |

**Key training parameters**:
- `--advantage_estimator reinforce_baseline` (no critic model needed)
- `--n_samples_per_prompt 4` (4 rollouts per task for variance reduction)
- `--zero_stage 3` + `--adam_offload` (full-param 8B on 32GB GPUs)
- `--repetition_penalty 1.2` (prevents token repetition collapse)
- `--init_kl_coef 0.01` (KL penalty vs SFT policy, via reward signal)
- `--async_train --colocate_all_models` (actor & vLLM on separate GPU placement groups)

**GPU layout** (auto-detected by `launch_rl_training.sh`):
| Clean GPUs | Actor | vLLM Engines | Total |
|------------|-------|-------------|-------|
| ≥ 7 | 5 | 2 | 7 |
| ≥ 5 | 4 | 1 | 5 |
| ≥ 3 | 2 | 1 | 3 |

Note: Frozen agent vLLM occupies 1 additional GPU (GPU 4 by default).

**Dependencies**: OpenRLHF (`train_ppo_ray`), `agent_func_memory_constructor.py`, vLLM, WebShop server, Ray, DeepSpeed

---

## Package Modules

### `memory_constructor/data/`

| Module | Description |
|--------|-------------|
| `schemas.py` | Dataclasses: `WebShopTrajectory`, `MemoryItem`, `SFTSample`, `CandidateMemory`, `EvaluationRecord` |
| `webshop_parser.py` | `WebShopParser`: parses raw JSONL trajectories, extracts observations/actions/history |
| `sft_dataset.py` | `MemoryConstructorSFTDataset`: PyTorch Dataset, formats prompt template, tokenizes, returns `(input_ids, attn_mask, loss_mask)` |
| `hindsight_labeling.py` | `HindsightLabeler` (LLM-as-judge), `HeuristicLabeler` (rule-based), `DemandAwareHindsightLabeler` (4-phase with retrieval verification) |
| `memory_store.py` | `MemoryStore`: append-only buffer with budget enforcement |

### `memory_constructor/models/`

| Module | Description |
|--------|-------------|
| `retriever.py` | `BM25Retriever`, `DenseRetriever` (SentenceTransformer), `HybridRetriever` (weighted fusion, default 0.5/0.5) |
| `agent.py` | `GPT5Agent` (frozen agent wrapper, OpenAI-compatible API), `MockAgent` (testing) |

### `memory_constructor/training/`

| Module | Description |
|--------|-------------|
| `candidate_sampler.py` | `CandidateSampler`: vLLM-based batch generation of N candidates per sample |
| `candidate_scorer.py` | `CandidateScorer`: counterfactual + hindsight + quality scoring |
| `best_of_n_trainer.py` | `BestOfNTrainer`: train on highest-scored candidates |

### `memory_constructor/environment/`

| Module | Description |
|--------|-------------|
| `memory_store.py` | Environment-side memory store for RL episodes |
| `agent_instance.py` | `MemoryConstructorAgentInstance`: OpenRLHF `AgentInstanceBase` with `reset()` / `step()` for multi-turn RL |

### `memory_constructor/evaluation/`

| Module | Description |
|--------|-------------|
| `evaluator.py` | `Evaluator`: runs full episode evaluation, computes all metrics |
| `metrics.py` | `MetricsCalculator`, `EvaluationMetrics`: aggregation and reporting |

### `memory_constructor/utils/`

| Module | Description |
|--------|-------------|
| `json_utils.py` | `parse_json_robust()` (handles LLM output quirks, Qwen3 `<think>` blocks), `validate_memory_item()`, `truncate_by_tokens()` |
| `logging_utils.py` | `setup_file_logging()` with file rotation |

---

## Key Design Decisions

1. **Append-only memory** — No update/delete. Simplifies credit assignment and makes the write decision binary.
2. **Hybrid retrieval** — BM25 (keyword) + dense embeddings (semantic). Robust to both exact-match and paraphrase queries.
3. **Asymmetric write reward** — No-write penalty (-0.1) > write cost (-0.05). Directly combats the low write rate problem observed in SFT models.
4. **Dynamic retrieval k** — `k = ceil(step / 2)`. More memories become available and relevant as episodes progress.
5. **Full parameter training** — No LoRA. ZeRO-3 + Adam offload + gradient checkpointing enables 8B full-param training on 32GB GPUs.
6. **Chunked logits** — Float32 logits processed in 2048-token chunks to avoid OOM from `(batch, seq, 151936) × 4 bytes`.

## Dependencies

**Core**: PyTorch, Transformers, DeepSpeed, OpenRLHF (parent project)

**Retrieval**: `rank-bm25`, `sentence-transformers` (all-MiniLM-L6-v2), `faiss-cpu`

**RL Runtime**: vLLM (generation engine), Ray (distributed training), WebShop environment (Java/Lucene)

**Evaluation**: scikit-learn, rouge-score, nltk

See [requirements.txt](requirements.txt) for full list.

## Hardware Requirements

| Stage | Minimum GPUs | Recommended |
|-------|-------------|-------------|
| SFT (ZeRO-2) | 2 × 32GB | 4 × 32GB |
| SFT (ZeRO-3) | 4 × 32GB | 8 × 32GB |
| Best-of-N generation | 1 × 32GB | 2 × 32GB (vLLM TP) |
| RL training | 4 × 32GB (3 actor + 1 vLLM) + 1 × 32GB (frozen agent) | 8 × 32GB |

Docker users: set `NCCL_SHM_DISABLE=1` if `/dev/shm` < 1GB.
