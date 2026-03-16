#!/bin/bash
#
# Start vLLM Service
#
# Usage:
#   bash run_vllm.sh [GPU_ID] [PORT]
#
#   Examples:
#     bash run_vllm.sh           # Default: GPU 0, Port 8001
#     bash run_vllm.sh 0 8001    # GPU 0, Port 8001
#     bash run_vllm.sh 1 8002    # GPU 1, Port 8002
#

echo "=========================================="
echo " Starting vLLM Service"
echo "=========================================="

# ==================== Configuration ====================
# Model path - modify to your model path
MODEL_PATH="/data/home/ranxinyu/common_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# GPU and port configuration (can be overridden via command line args)
GPU_ID=${1:-5}
PORT=${2:-8001}

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
