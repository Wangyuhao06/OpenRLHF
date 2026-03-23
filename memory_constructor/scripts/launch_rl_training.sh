#!/bin/bash
# ============================================================
# Launch RL Training for Memory Constructor
# ============================================================
# This script ensures all prerequisite services are running
# (WebShop server, frozen agent vLLM), then launches training.
#
# Usage:
#   bash memory_constructor/scripts/launch_rl_training.sh
#   bash memory_constructor/scripts/launch_rl_training.sh --test   # small test run
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MC_ROOT="$PROJECT_ROOT/memory_constructor"

# ---- NCCL fixes for small /dev/shm (Docker) ----
export NCCL_SHM_DISABLE=1
# NOTE: P2P (NVLink/PCIe) does NOT use /dev/shm, so leave it enabled for ZeRO-3 performance
# Clean stale NCCL/PSM shared memory segments
rm -f /dev/shm/nccl-* /dev/shm/psm_* 2>/dev/null || true

# ---- CUDA memory management ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Configuration ----
WEBSHOP_PORT=6001
WEBSHOP_NUM_PRODUCTS=1000
FROZEN_AGENT_PORT=8000
FROZEN_AGENT_GPU=4                    # GPU for frozen agent vLLM
FROZEN_AGENT_MODEL="Qwen/Qwen3-8B"

# Paths
WEBSHOP_PYTHON="$PROJECT_ROOT/.webshop_venv/bin/python"
WEBSHOP_PATH="/home/yuhao/home/yuhao/code/WebShop"
JAVA_HOME="/home/yuhao/.jdk/jdk-21.0.10+7"
TRAINING_PYTHON="$PROJECT_ROOT/.venv/bin/python"
TRAINING_VENV="$PROJECT_ROOT/.venv"
WEBSHOP_SERVER_SCRIPT="$PROJECT_ROOT/interaction_env/WebShop_VanillaReactAgent_TrajectoryGeneration/runner/webshop_server_pool.py"

LOG_DIR="$MC_ROOT/results"
mkdir -p "$LOG_DIR"

# ---- Determine training mode ----
TEST_MODE=false
if [[ "$1" == "--test" ]]; then
    TEST_MODE=true
    echo "[INFO] Running in TEST mode (small batch)"
fi

# ---- Helper functions ----
check_health() {
    local url="$1"
    curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null
}

wait_for_service() {
    local name="$1"
    local url="$2"
    local max_wait="${3:-300}"  # default 5 min
    local elapsed=0

    echo "[INFO] Waiting for $name at $url ..."
    while [ $elapsed -lt $max_wait ]; do
        if [ "$(check_health "$url")" = "200" ]; then
            echo "[OK] $name is ready."
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  ... still waiting ($elapsed/${max_wait}s)"
    done
    echo "[ERROR] $name did not become ready in ${max_wait}s"
    return 1
}

# ============================================================
# Step 1: Start WebShop Pool Server (if not running)
# ============================================================
echo ""
echo "========================================"
echo "Step 1: WebShop Pool Server"
echo "========================================"

WEBSHOP_URL="http://localhost:$WEBSHOP_PORT"
if [ "$(check_health "$WEBSHOP_URL/health")" = "200" ]; then
    echo "[OK] WebShop server already running at $WEBSHOP_URL"
else
    echo "[INFO] Starting WebShop pool server on port $WEBSHOP_PORT ..."

    export JAVA_HOME="$JAVA_HOME"
    export PATH="$JAVA_HOME/bin:$PATH"
    export WEBSHOP_PATH="$WEBSHOP_PATH"

    cd "$WEBSHOP_PATH"
    nohup "$WEBSHOP_PYTHON" "$WEBSHOP_SERVER_SCRIPT" \
        --port "$WEBSHOP_PORT" \
        --num_products "$WEBSHOP_NUM_PRODUCTS" \
        > "$LOG_DIR/webshop_server.log" 2>&1 &
    WEBSHOP_PID=$!
    echo "[INFO] WebShop server PID: $WEBSHOP_PID"
    cd "$PROJECT_ROOT"

    # Wait for WebShop to be ready (loading products + Lucene index takes ~60s)
    wait_for_service "WebShop" "$WEBSHOP_URL/health" 180
fi

# ============================================================
# Step 2: Frozen Agent vLLM Server (if not running)
# ============================================================
echo ""
echo "========================================"
echo "Step 2: Frozen Agent vLLM Server"
echo "========================================"

FROZEN_AGENT_URL="http://localhost:$FROZEN_AGENT_PORT/v1"
if curl -s "http://localhost:$FROZEN_AGENT_PORT/v1/models" 2>/dev/null | grep -q "model"; then
    echo "[OK] Frozen agent already running at $FROZEN_AGENT_URL"
else
    echo "[INFO] Starting frozen agent vLLM on GPU $FROZEN_AGENT_GPU ..."

    CUDA_VISIBLE_DEVICES=$FROZEN_AGENT_GPU \
    HF_ENDPOINT=https://hf-mirror.com \
    nohup "$TRAINING_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
        --model "$FROZEN_AGENT_MODEL" \
        --port "$FROZEN_AGENT_PORT" \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        > "$LOG_DIR/frozen_agent.log" 2>&1 &
    FROZEN_PID=$!
    echo "[INFO] Frozen agent PID: $FROZEN_PID"

    wait_for_service "Frozen Agent" "http://localhost:$FROZEN_AGENT_PORT/v1/models" 300
fi

# ============================================================
# Step 3: Launch RL Training
# ============================================================
echo ""
echo "========================================"
echo "Step 3: RL Training"
echo "========================================"

# Determine available GPUs (exclude frozen agent GPU)
ALL_GPUS=(0 1 2 3 5 6 7)
CLEAN_GPUS=()
for gpu in "${ALL_GPUS[@]}"; do
    mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null)
    if [ "$mem_used" -lt 1000 ] 2>/dev/null; then
        CLEAN_GPUS+=($gpu)
    fi
done

echo "[INFO] Clean GPUs available: ${CLEAN_GPUS[*]}"

# Determine GPU layout based on available clean GPUs
# NOTE: With --async_train, actor PG and vLLM PG are SEPARATE, so:
#   total GPUs needed = ACTOR_GPUS + VLLM_ENGINES * VLLM_TP
# Preferred: 5 actor GPUs + 2 vLLM engines = 7 GPUs
# Medium:    4 actor GPUs + 1 vLLM engine  = 5 GPUs
# Fallback:  2 actor GPUs + 1 vLLM engine  = 3 GPUs
if [ ${#CLEAN_GPUS[@]} -ge 7 ]; then
    ACTOR_GPUS=5
    VLLM_ENGINES=2
    VLLM_TP=1
    TOTAL_NEEDED=7
    echo "[INFO] Full layout: 5 actor GPUs + 2 vLLM engines = 7 GPUs"
elif [ ${#CLEAN_GPUS[@]} -ge 5 ]; then
    ACTOR_GPUS=4
    VLLM_ENGINES=1
    VLLM_TP=1
    TOTAL_NEEDED=5
    echo "[INFO] Medium layout: 4 actor GPUs + 1 vLLM engine = 5 GPUs"
elif [ ${#CLEAN_GPUS[@]} -ge 3 ]; then
    ACTOR_GPUS=2
    VLLM_ENGINES=1
    VLLM_TP=1
    TOTAL_NEEDED=3
    echo "[INFO] Reduced layout: 2 actor GPUs + 1 vLLM engine = 3 GPUs"
else
    echo "[ERROR] Need at least 3 clean GPUs. Only found: ${CLEAN_GPUS[*]}"
    echo "[HINT] If GPUs have leaked memory from killed processes, restart the container."
    exit 1
fi

# Build CUDA_VISIBLE_DEVICES string
TRAIN_GPUS=""
for i in $(seq 0 $((TOTAL_NEEDED - 1))); do
    if [ -n "$TRAIN_GPUS" ]; then
        TRAIN_GPUS="$TRAIN_GPUS,${CLEAN_GPUS[$i]}"
    else
        TRAIN_GPUS="${CLEAN_GPUS[$i]}"
    fi
done

echo "[INFO] Training GPUs: $TRAIN_GPUS (actor=$ACTOR_GPUS, vllm=${VLLM_ENGINES}xTP${VLLM_TP})"

# Set training parameters based on mode and GPU count
if [ "$TEST_MODE" = true ]; then
    MAX_SAMPLES=24
    ROLLOUT_BATCH=8
    TRAIN_BATCH=8
    NUM_EPISODES=1
    SAVE_STEPS=100
    echo "[INFO] TEST mode: $MAX_SAMPLES samples, $NUM_EPISODES episode"
else
    MAX_SAMPLES=500
    # Scale batch sizes with GPU count (must be divisible by ACTOR_GPUS)
    if [ "$ACTOR_GPUS" -ge 5 ]; then
        ROLLOUT_BATCH=20
        TRAIN_BATCH=20
    elif [ "$ACTOR_GPUS" -ge 4 ]; then
        ROLLOUT_BATCH=12
        TRAIN_BATCH=12
    else
        ROLLOUT_BATCH=8
        TRAIN_BATCH=8
    fi
    NUM_EPISODES=3
    SAVE_STEPS=10
    echo "[INFO] FULL mode: $MAX_SAMPLES samples, $NUM_EPISODES episodes"
fi

# Export env vars for agent_func
export WEBSHOP_SERVER_URL="$WEBSHOP_URL"
export FROZEN_AGENT_URL="$FROZEN_AGENT_URL"
export FROZEN_AGENT_MODEL="$FROZEN_AGENT_MODEL"
export FROZEN_AGENT_API_KEY="EMPTY"
export HF_ENDPOINT="https://hf-mirror.com"

echo ""
echo "Environment:"
echo "  WEBSHOP_SERVER_URL=$WEBSHOP_SERVER_URL"
echo "  FROZEN_AGENT_URL=$FROZEN_AGENT_URL"
echo "  FROZEN_AGENT_MODEL=$FROZEN_AGENT_MODEL"
echo "  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS"
echo ""

# Launch training
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
"$TRAINING_PYTHON" "$MC_ROOT/scripts/07_train_rl.py" \
    --max_samples "$MAX_SAMPLES" \
    --rollout_batch_size "$ROLLOUT_BATCH" \
    --train_batch_size "$TRAIN_BATCH" \
    --num_episodes "$NUM_EPISODES" \
    --actor_num_gpus "$ACTOR_GPUS" \
    --ref_num_gpus "$ACTOR_GPUS" \
    --vllm_num_engines "$VLLM_ENGINES" \
    --vllm_tensor_parallel_size "$VLLM_TP" \
    --vllm_gpu_memory_utilization 0.9 \
    --save_steps "$SAVE_STEPS" \
    2>&1 | tee "$LOG_DIR/rl_training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "[DONE] Training finished with exit code $?"
