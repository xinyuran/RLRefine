#!/bin/bash
# ============================================================
# ms-swift 3.x SFT Fine-tuning Script (Shell Version)
# For supervised fine-tuning of keyword extraction models
# 
# ms-swift version: 3.11.2+
# 
# Usage:
#   chmod +x sft_finetune.sh
#   ./sft_finetune.sh
#
# Note: If you encounter transformers version compatibility issues, run:
#   pip install transformers==4.46.0
# ============================================================

# ===================== Configuration =====================
# GPU configuration - specify GPU device IDs
export CUDA_VISIBLE_DEVICES=6,7

# Number of processes per node (equal to number of GPUs)
export NPROC_PER_NODE=2

# Model path
MODEL_PATH="/data/home/ranxinyu/common_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# Dataset path (JSONL file)
DATASET_PATH="/data/home/ranxinyu/project_rxy/llm_keyword_sft/sft_data_with_think_tags_jsonl_output/sft_all_merged.jsonl"

# Output directory
OUTPUT_DIR="./output/sft"

# ===================== Training Parameters =====================
NUM_EPOCHS=5
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=1e-5
MAX_LENGTH=8192

# LoRA parameters
LORA_RANK=16
LORA_ALPHA=64

# Save and logging
SAVE_STEPS=100
EVAL_STEPS=100
LOGGING_STEPS=20
EARLY_STOP_INTERVAL=3
# ===================================================

echo "============================================================"
echo "Starting SFT Fine-tuning Training"
echo "============================================================"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES}"
echo "Processes: ${NPROC_PER_NODE}"
echo "Model path: ${MODEL_PATH}"
echo "Dataset path: ${DATASET_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Max length: ${MAX_LENGTH}"
echo "============================================================"

swift sft \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_PATH}" \
    --train_type lora \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules all-linear \
    --max_length ${MAX_LENGTH} \
    --truncation_strategy delete \
    --torch_dtype bfloat16 \
    --gradient_checkpointing true \
    --warmup_ratio 0.05 \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --early_stop_interval ${EARLY_STOP_INTERVAL} \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --eval_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_total_limit 3 \
    --dataloader_num_workers 8 \
    --split_dataset_ratio 0.1

echo "============================================================"
echo "Training complete!"
echo "Model saved at: ${OUTPUT_DIR}"
echo "============================================================"
