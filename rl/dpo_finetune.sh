#!/bin/bash
# ============================================================
# RLRefine DPO Preference Alignment Script
# Use RL to refine LLM's reasoning process and information extraction capability
#
# ms-swift version: 3.11.2+
#
# Usage:
#   chmod +x dpo_finetune.sh
#   ./dpo_finetune.sh
#
# ============================================================
# Training Pipeline
#
# Option 1: Use merged model (recommended)
#   1. Run SFT training: ./sft_finetune.sh
#   2. Merge LoRA weights: ./merge_lora.sh
#   3. Update MODEL_PATH below to the merged model path
#   4. Run DPO training: ./dpo_finetune.sh
#
# Option 2: Use base model + SFT LoRA weights
#   1. Set MODEL_PATH to the base model path
#   2. Set SFT_ADAPTER_PATH to the SFT checkpoint path
#   3. Run DPO training: ./dpo_finetune.sh
# ============================================================
# ===================== Configuration =====================
# GPU configuration - specify GPU device IDs
export CUDA_VISIBLE_DEVICES=0,1
# Number of processes per node (equal to number of GPUs)
export NPROC_PER_NODE=2
# Model path (merged SFT model or base model)
MODEL_PATH="./output/sft_merged_model"
# SFT adapter path (required if using Option 2)
SFT_ADAPTER_PATH="./output/sft/checkpoint-xxx"
# Dataset path (DPO format)
DATASET_PATH="./data/dpo_dataset.jsonl"
# Output directory
OUTPUT_DIR="./output/dpo"
# ===================== Training Parameters =====================
NUM_EPOCHS=1
BATCH_SIZE=2
GRADIENT_ACCUMULATION=2
LEARNING_RATE=5e-7
MAX_LENGTH=4096
MAX_GRAD_NORM=1.0
# LoRA parameters
LORA_RANK=8
LORA_ALPHA=32
# DPO-specific parameters
BETA=0.2
# Save and logging
SAVE_STEPS=50
EVAL_STEPS=50
LOGGING_STEPS=50
# ============================================================
echo "============================================================"
echo "Starting DPO Training"
echo "============================================================"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES}"
echo "Processes: ${NPROC_PER_NODE}"
echo "Model path: ${MODEL_PATH}"
echo "Dataset path: ${DATASET_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Beta (KL penalty): ${BETA}"
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
swift dpo \
    --model_path ${MODEL_PATH} \
    --dataset ${DATASET_PATH} \
        --train_type dpo \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
        --learning_rate ${LEARNING_RATE} \
        --max_length ${MAX_LENGTH} \
        --max_grad_norm ${MAX_GRAD_NORM} \
        --torch_dtype bfloat16 \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --beta ${BETA} \
        --save_steps ${SAVE_STEPS} \
        --eval_steps ${EVAL_STEPS} \
        --logging_steps ${LOGGING_STEPS} \
        --output_dir ${OUTPUT_DIR} \
        --logging_dir ${OUTPUT_DIR}/logs \
        --report_to all
