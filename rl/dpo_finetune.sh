#!/bin/bash
# ============================================================
# RLRefine DPO 偏好对齐脚本
# 用 RL 精炼 LLM 的思考过程与信息提取能力
#
# ms-swift 版本: 3.11.2+
#
# 使用方法:
#   chmod +x dpo_finetune.sh
#   ./dpo_finetune.sh
#
# ============================================================
# 训练流程说明
#
# 方式1: 使用合并后的模型 (推荐)
#   1. 运行 SFT 训练: ./sft_finetune.sh
#   2. 合并 LoRA 权重: ./merge_lora.sh
#   3. 修改下方 MODEL_PATH 为合并后的模型路径
#   4. 运行 DPO 训练: ./dpo_finetune.sh
#
# 方式2: 使用原始模型 + SFT LoRA 权重
#   1. 设置 MODEL_PATH 为原始模型路径
#   2. 设置 SFT_ADAPTER_PATH 为 SFT checkpoint 路径
#   3. 运行 DPO 训练: ./dpo_finetune.sh
# ============================================================
# ===================== 配置区域 =====================
# GPU配置 - 指定要使用的GPU编号
export CUDA_VISIBLE_DEVICES=0,1
# 每个节点的进程数 (等于GPU数量)
export NPROC_PER_NODE=2
# 模型路径 (合并后的 SFT 模型或原始模型)
MODEL_PATH="./output/sft_merged_model"
# SFT 适配器路径 (如果使用方式2，需要指定)
SFT_ADAPTER_PATH="./output/sft/checkpoint-xxx"
# 数据集路径 (DPO格式)
DATASET_PATH="./data/dpo_dataset.jsonl"
# 输出目录
OUTPUT_DIR="./output/dpo"
# ===================== 训练参数 =====================
NUM_EPOCHS=1
BATCH_SIZE=2
GRADIENT_ACCUMULATION=2
LEARNING_RATE=5e-7
MAX_LENGTH=4096
MAX_GRAD_NORM=1.0
# LoRA参数
LORA_RANK=8
LORA_ALPHA=32
# DPO特定参数
BETA=0.2
# 保存和日志
SAVE_STEPS=50
EVAL_STEPS=50
LOGGING_STEPS=50
# ============================================================
echo "============================================================"
echo "开始 DPO 训练"
echo "============================================================"
echo "GPU设备: ${CUDA_VISIBLE_DEVICES}"
echo "进程数: ${NPROC_PER_NODE}"
echo "模型路径: ${MODEL_PATH}"
echo "数据集路径: ${DATASET_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "Beta (KL惩罚): ${BETA}"
echo "============================================================"
# 检查模型路径是否存在
if [ ! -d "${MODEL_PATH}" ]; then
    echo "警告: 模型路径不存在: ${MODEL_PATH}"
    echo "请修改 MODEL_PATH 为实际的模型路径"
    exit 1
fi
# 检查数据集路径是否存在
if [ ! -f "${DATASET_PATH}" ]; then
    echo "警告: 数据集路径不存在: ${DATASET_PATH}"
    echo "请修改 DATASET_PATH 为实际的数据集路径"
    exit 1
fi
# 检查输出目录
mkdir -p ${OUTPUT_DIR}
# 启动训练
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
