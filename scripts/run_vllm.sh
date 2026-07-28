#!/bin/bash
#
# Start vLLM Service
#
# Usage:
#   bash run_vllm.sh [GPU_ID] [PORT] [MODEL]
#
#   Examples:
#     bash run_vllm.sh
#     bash run_vllm.sh 0 8001 /path/to/model
#

echo "=========================================="
echo " Starting vLLM Service"
echo "=========================================="

# ==================== Configuration ====================
# GPU and port configuration (can be overridden via command line args)
GPU_ID=${1:-0}
PORT=${2:-8001}
MODEL_PATH=${3:-Qwen/Qwen2.5-7B-Instruct}

# vLLM configuration
DTYPE="float16"
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=128
GPU_MEMORY_UTILIZATION=0.85
SWAP_SPACE=8
SEED=42
# ==================================================

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  GPU: $GPU_ID"
echo "  Port: $PORT"
echo "  MAX_NUM_SEQS: $MAX_NUM_SEQS"
echo "  GPU_MEMORY_UTILIZATION: $GPU_MEMORY_UTILIZATION"

# CUDA configuration
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Start vLLM service
vllm serve "$MODEL_PATH" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --swap-space "$SWAP_SPACE" \
    --port "$PORT" \
    --seed "$SEED" \
    --disable-log-requests
