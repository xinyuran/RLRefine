#!/bin/bash
# ============================================================
# StructAlign GRPO 强化学习脚本
# 通用结构化数据提取模型
#
# ms-swift 版本: 3.11.2+
#
# 使用方法:
#   chmod +x grpo_finetune.sh
#   ./grpo_finetune.sh
#
# ============================================================
# 训练流程说明:
#
# 推荐流程: SFT -> DPO -> GRPO
#   1. 完成 SFT 训练后，运行 merge_lora.sh 合并权重
#   2. 使用合并后的模型进行 DPO 训练
#   3. 完成 DPO 训练后，再次运行 merge_lora.sh 合并权重
#   4. 修改下方 MODEL_PATH 为 DPO 合并后的模型路径
#   5. 运行 GRPO 训练: ./grpo_finetune.sh
#
# 注意:
#   - GRPO 需要 vLLM 进行推理加速
#   - 需要较大的显存
# ============================================================

# ===================== 配置区域 =====================
# GPU配置 - 指定要使用的GPU编号
export CUDA_VISIBLE_DEVICES=0,1
# 每个节点的进程数 (等于GPU数量)
export NPROC_PER_NODE=2
# 模型路径 (建议使用 DPO 合并后的模型)
# 请修改为实际的模型路径
MODEL_PATH="./output/dpo_merged_model"
# 数据集路径 (GRPO格式，只需要 prompt)
DATASET_PATH="./data/grpo_prompts.jsonl"
# 输出目录
OUTPUT_DIR="./output/grpo"
# ===================== 训练参数 =====================
NUM_EPOCHS=1
BATCH_SIZE=2
GRADIENT_ACCUMULATION=8
LEARNING_RATE=1.5e-6
MAX_LENGTH=4096
MAX_COMPLETION_LENGTH=2048
# LoRA参数
LORA_RANK=8
LORA_ALPHA=16
# GRPO特定参数
BETA=0.01
NUM_GENERATIONS=8
TEMPERATURE=1.9
# 像函数配置
EXTERNAL_PLUGINS="./reward_builder.py"
REWARD_FUNCS="schema_based_reward"
# vLLM 推理加速配置
USE_VLLM=true
VLLM_MODE=colocate
VLLM_MAX_MODEL_LEN=4096
VLLM_GPU_MEMORY_UTILIZATION=0.5
# 保存和日志
SAVE_STEPS=50
EVAL_STEPS=50
LOGGING_STEPS=50

# ============================================================
echo "============================================================"
echo "开始 GRPO 训练"
echo "============================================================"
echo "GPU设备: ${CUDA_VISIBLE_DEVICES}"
echo "进程数: ${NPROC_PER_NODE}"
echo "模型路径: ${MODEL_PATH}"
echo "数据集路径: ${DATASET_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
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
