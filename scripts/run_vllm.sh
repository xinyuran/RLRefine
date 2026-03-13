#!/bin/bash
#
# 启动 vLLM 服务
#
# 使用方法：
#   bash run_vllm.sh [GPU_ID] [PORT]
#
#   示例：
#     bash run_vllm.sh           # 默认: GPU 0, 端口 8001
#     bash run_vllm.sh 0 8001    # GPU 0, 端口 8001
#     bash run_vllm.sh 1 8002    # GPU 1, 端口 8002
#

echo "=========================================="
echo " 启动 vLLM 服务"
echo "=========================================="

# ==================== 配置区域 ====================
# 模型路径 - 请修改为您的模型路径
MODEL_PATH="/data/home/ranxinyu/common_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# GPU 和端口配置（可通过命令行参数覆盖）
GPU_ID=${1:-5}
PORT=${2:-8001}

# vLLM 配置
DTYPE="float16"
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=128
GPU_MEMORY_UTILIZATION=0.85
SWAP_SPACE=8
SEED=42
# ==================================================

echo "配置："
echo "  模型: $MODEL_PATH"
echo "  GPU: $GPU_ID"
echo "  端口: $PORT"
echo "  MAX_NUM_SEQS: $MAX_NUM_SEQS"
echo "  GPU_MEMORY_UTILIZATION: $GPU_MEMORY_UTILIZATION"

# CUDA 配置
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 启动 vLLM 服务（确保 \ 后无空格，参数格式正确）
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