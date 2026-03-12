# Memory Constructor for WebShop

A trainable memory constructor that learns to compress and store information for a fixed downstream agent in the WebShop environment.

## Project Structure

```
memory_constructor/
├── configs/                    # Configuration files
│   ├── model.yaml             # Model hyperparameters
│   ├── training.yaml          # Training settings
│   └── data.yaml              # Data paths and preprocessing
├── data/                      # Data storage
│   └── webshop/
│       ├── raw/               # Raw trajectories
│       ├── processed/         # Processed data
│       └── splits/            # Train/val/test splits
├── memory_constructor/        # Main package
│   ├── models/                # Model implementations
│   │   ├── retriever.py      # Hybrid retriever (BM25 + dense)
│   │   └── agent.py          # GPT-5 agent wrapper
│   ├── data/                  # Data processing
│   │   ├── schemas.py        # Data schemas
│   │   ├── webshop_parser.py # WebShop trajectory parser
│   │   ├── sft_dataset.py    # SFT dataset
│   │   └── candidate_dataset.py # Best-of-n dataset
│   ├── training/              # Training modules
│   │   ├── candidate_sampler.py # Candidate generation
│   │   ├── candidate_scorer.py  # Candidate scoring
│   │   ├── sft_trainer.py    # SFT training
│   │   └── rl_trainer.py     # RL training
│   ├── environment/           # Environment components
│   │   ├── memory_store.py   # Append-only memory store
│   │   └── agent_instance.py # AgentInstance for OpenRLHF
│   ├── evaluation/            # Evaluation framework
│   │   ├── metrics.py        # Evaluation metrics
│   │   ├── evaluator.py      # Main evaluator
│   │   └── baselines.py      # Baseline implementations
│   └── utils/                 # Utilities
│       ├── json_utils.py     # JSON parsing/validation
│       ├── logging_utils.py  # Logging
│       └── openrlhf_utils.py # OpenRLHF helpers
├── scripts/                   # Training/evaluation scripts
│   ├── 01_preprocess_webshop.py
│   ├── 02_train_sft.py
│   ├── 03_generate_candidates.py
│   ├── 04_score_candidates.py
│   ├── 05_train_bestofn.py
│   ├── 06_train_rl.py
│   └── 07_evaluate.py
├── experiments/               # Experiment logs
├── tests/                     # Unit tests
└── requirements.txt           # Dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Copy .env file with GPT-5 API key
cp /home/yuhao/code/.env .env
```

3. Verify OpenRLHF installation:
```bash
python -c "import openrlhf; print(openrlhf.__version__)"
```

## Training Pipeline

### Stage 1: SFT Training

Train the memory constructor with hindsight-labeled data:

```bash
python scripts/01_preprocess_webshop.py --config configs/data.yaml
python scripts/02_train_sft.py --config configs/training.yaml
```

### Stage 2: Best-of-N Training

Generate and score candidates, then train on best selections:

```bash
python scripts/03_generate_candidates.py --checkpoint checkpoints/sft_final
python scripts/04_score_candidates.py --config configs/training.yaml
python scripts/05_train_bestofn.py --config configs/training.yaml
```

### Stage 3: RL Training (Optional)

Train with online trajectory generation:

```bash
python scripts/06_train_rl.py --config configs/training.yaml
```

## Evaluation

```bash
python scripts/07_evaluate.py \
    --checkpoint checkpoints/bestofn_final \
    --config configs/training.yaml \
    --output results/evaluation.json
```

## Key Features

- **Append-only memory**: No update/merge/delete operations
- **Hybrid retrieval**: BM25 + dense embedding
- **Multi-turn training**: Uses OpenRLHF's AgentInstance
- **Weighted scoring**: Hindsight + counterfactual evaluation
- **Computational cost penalty**: Natural write/no_write balance
- **Temporal ordering**: Memories include timestamps

## Configuration

Edit `configs/*.yaml` files to adjust:
- Model size and hyperparameters
- Memory constraints (max_value_tokens, budget)
- Training settings (batch size, learning rate)
- Reward weights
- Data paths

## Monitoring

Training metrics are logged to:
- Console output
- TensorBoard logs in `logs/`
- Weights & Biases (if configured)

## Citation

If you use this code, please cite:
```
@misc{memory-constructor-2026,
  title={Trainable Memory Constructor for Long-Horizon Tasks},
  author={Your Name},
  year={2026}
}
```
