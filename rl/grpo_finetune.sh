#!/bin/bash
# ============================================================
# RLRefine GRPO Reinforcement Learning Script
# Use RL to refine LLM's reasoning process and information extraction capability
#
# ms-swift version: 3.11.2+
#
# Usage:
#   chmod +x grpo_finetune.sh
#   ./grpo_finetune.sh
#
# ============================================================
# Training Pipeline:
#
# Recommended flow: SFT -> DPO -> GRPO
#   1. After SFT training, run merge_lora.sh to merge weights
#   2. Use the merged model for DPO training
#   3. After DPO training, run merge_lora.sh again to merge weights
#   4. Update MODEL_PATH below to the DPO merged model path
#   5. Run GRPO training: ./grpo_finetune.sh
#
# Notes:
#   - GRPO requires vLLM for inference acceleration
#   - Requires large GPU memory
# ============================================================

# ===================== Configuration =====================
# GPU configuration - specify GPU device IDs
export CUDA_VISIBLE_DEVICES=0,1
# Number of processes per node (equal to number of GPUs)
export NPROC_PER_NODE=2
# Model path (recommended to use DPO merged model)
# Please update to the actual model path
MODEL_PATH="./output/dpo_merged_model"
# Dataset path (GRPO format, only prompts needed)
DATASET_PATH="./data/grpo_prompts.jsonl"
# Output directory
OUTPUT_DIR="./output/grpo"
# ===================== Training Parameters =====================
NUM_EPOCHS=1
BATCH_SIZE=2
GRADIENT_ACCUMULATION=8
LEARNING_RATE=1.5e-6
MAX_LENGTH=4096
MAX_COMPLETION_LENGTH=2048
# LoRA parameters
LORA_RANK=8
LORA_ALPHA=16
# GRPO-specific parameters
BETA=0.01
NUM_GENERATIONS=8
TEMPERATURE=1.9
# Reward function configuration
EXTERNAL_PLUGINS="./reward_builder.py"
REWARD_FUNCS="schema_based_reward"
# vLLM inference acceleration configuration
USE_VLLM=true
VLLM_MODE=colocate
VLLM_MAX_MODEL_LEN=4096
VLLM_GPU_MEMORY_UTILIZATION=0.5
# Save and logging
SAVE_STEPS=50
EVAL_STEPS=50
LOGGING_STEPS=50

# ============================================================
echo "============================================================"
echo "Starting GRPO Training"
echo "============================================================"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES}"
echo "Processes: ${NPROC_PER_NODE}"
echo "Model path: ${MODEL_PATH}"
echo "Dataset path: ${DATASET_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"
# Check if model path exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Warning: Model path does not exist: ${MODEL_PATH}"
    echo "Please update MODEL_PATH to the actual model path"
    exit 1
fi
# Check if dataset path exists
if [ ! -f "${DATASET_PATH}" ]; then
    echo "Warning: Dataset path does not exist: ${DATASET_PATH}"
    echo "Please update DATASET_PATH to the actual dataset path"
    exit 1
fi
# Check output directory
mkdir -p ${OUTPUT_DIR}
# Start training
nproc_per_node=${NPROC_PER_NODE} \
swift rl \
    --model_path ${MODEL_PATH} \
    --dataset ${DATASET_PATH} \
        --train_type grpo \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
        --learning_rate ${LEARNING_RATE} \
        --max_length ${MAX_LENGTH} \
        --max_completion_length ${MAX_COMPLETION_LENGTH} \
        --torch_dtype bfloat16 \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --beta ${BETA} \
        --num_generations ${NUM_GENERATIONS} \
        --temperature ${TEMPERATURE} \
        --external_plugins ${EXTERNAL_PLUGINS} \
        --reward_funcs ${REWARD_FUNCS} \
        --use_vllm ${USE_VLLM} \
        --vllm_mode ${VLLM_MODE} \
        --vllm_max_model_len ${VLLM_MAX_MODEL_LEN} \
        --vllm_gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
        --save_steps ${SAVE_STEPS} \
        --eval_steps ${EVAL_STEPS} \
        --logging_steps ${LOGGING_STEPS} \
        --output_dir ${OUTPUT_DIR} \
        --logging_dir ${OUTPUT_DIR}/logs \
        --report_to all
