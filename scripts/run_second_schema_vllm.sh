#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-5}"
PORT="${2:-8003}"
BASE_MODEL="${3:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL="Qwen/Qwen2.5-7B-Instruct"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Serving preregistered Base model"
echo "  path: ${BASE_MODEL}"
echo "  served name: ${SERVED_MODEL}"
echo "  physical GPU: ${GPU_ID}"
echo "  port: ${PORT}"

vllm serve "${BASE_MODEL}" \
  --served-model-name "${SERVED_MODEL}" \
  --dtype float16 \
  --max-model-len 2048 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 \
  --swap-space 8 \
  --port "${PORT}" \
  --seed 42 \
  --disable-log-requests
