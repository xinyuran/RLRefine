#!/bin/bash
# ============================================================
# ms-swift 3.x SFT 微调脚本 (Shell 版本)
# 用于关键词提取模型的监督微调
# 
# ms-swift 版本: 3.11.2+
# 
# 使用方法:
#   chmod +x sft_finetune.sh
#   ./sft_finetune.sh
#
# 注意: 如遇到 transformers 版本兼容性问题，请执行:
#   pip install transformers==4.46.0
# ============================================================

# ===================== 配置区域 =====================
# GPU配置 - 指定要使用的GPU编号
export CUDA_VISIBLE_DEVICES=6,7

# 每个节点的进程数 (等于GPU数量)
export NPROC_PER_NODE=2

# 模型路径
MODEL_PATH="/data/home/ranxinyu/common_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# 数据集路径 (JSONL文件)
DATASET_PATH="/data/home/ranxinyu/project_rxy/llm_keyword_sft/sft_data_with_think_tags_jsonl_output/sft_all_merged.jsonl"

# 输出目录
OUTPUT_DIR="./output/sft"

# ===================== 训练参数 =====================
NUM_EPOCHS=5
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=1e-5
MAX_LENGTH=8192

# LoRA参数
LORA_RANK=16
LORA_ALPHA=64

# 保存和日志
SAVE_STEPS=100
EVAL_STEPS=100
LOGGING_STEPS=20
EARLY_STOP_INTERVAL=3
# ===================================================

echo "============================================================"
echo "开始 SFT 微调训练"
echo "============================================================"
echo "GPU设备: ${CUDA_VISIBLE_DEVICES}"
echo "进程数: ${NPROC_PER_NODE}"
echo "模型路径: ${MODEL_PATH}"
echo "数据集路径: ${DATASET_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "最大长度: ${MAX_LENGTH}"
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
echo "训练完成!"
echo "模型保存在: ${OUTPUT_DIR}"
echo "============================================================"
