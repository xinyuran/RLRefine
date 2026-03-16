#!/bin/bash
# ============================================================
# ms-swift 3.x LoRA Weight Merging Script
# Merges trained LoRA adapter weights into the base model
#
# ms-swift version: 3.11.2+
#
# Usage:
#   chmod +x merge_lora.sh
#   ./merge_lora.sh
#
# ============================================================
# How it works:
#
# ms-swift saves training config to args.json in the checkpoint dir,
# including the original base model path. `swift export` reads this
# config automatically.
#
# If you need to explicitly specify the base model, use BASE_MODEL_PATH.
#
# The merged model is saved under the checkpoint dir as xxx-merged
# e.g.: output/sft/v2-xxx/checkpoint-200 -> output/sft/v2-xxx/checkpoint-200-merged
# ============================================================

# ===================== Configuration =====================
# GPU config
export CUDA_VISIBLE_DEVICES=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# LoRA checkpoint path (from SFT/DPO/GRPO training)
# Change this to your actual checkpoint path
ADAPTER_PATH="./output/grpo/your-run-id/checkpoint-xxx"

# Base model path (optional)
# Leave empty to auto-detect from checkpoint's args.json
BASE_MODEL_PATH=""
# BASE_MODEL_PATH="/path/to/your/base/model"

# Merge mode: "gpu" (fast, needs VRAM) or "cpu" (slow, no VRAM limit)
MERGE_MODE="gpu"
# ===================================================

# Validate adapter path
if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "Error: checkpoint path does not exist: ${ADAPTER_PATH}"
    echo "Please check the path"
    exit 1
fi

# Check args.json
if [ ! -f "${ADAPTER_PATH}/args.json" ]; then
    echo "Warning: ${ADAPTER_PATH}/args.json not found"
    echo "This may prevent auto-detection of the base model path"
    if [ -z "${BASE_MODEL_PATH}" ]; then
        echo "Please set BASE_MODEL_PATH"
        exit 1
    fi
fi

echo "============================================================"
echo "Starting LoRA weight merging"
echo "============================================================"
echo "Adapter path: ${ADAPTER_PATH}"
if [ -n "${BASE_MODEL_PATH}" ]; then
    echo "Base model path: ${BASE_MODEL_PATH}"
else
    echo "Base model path: (auto-detect from args.json)"
fi
echo "Output path: ${ADAPTER_PATH}-merged"
echo "Merge mode: ${MERGE_MODE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

if [ "${MERGE_MODE}" == "cpu" ]; then
    echo "Using CPU mode (slower but no VRAM limit)..."
    export CUDA_VISIBLE_DEVICES=""

    if [ -n "${BASE_MODEL_PATH}" ]; then
        swift export \
            --model "${BASE_MODEL_PATH}" \
            --adapters "${ADAPTER_PATH}" \
            --merge_lora true \
            --device_map cpu
    else
        swift export \
            --adapters "${ADAPTER_PATH}" \
            --merge_lora true \
            --device_map cpu
    fi
else
    echo "Using GPU mode..."
    if [ -n "${BASE_MODEL_PATH}" ]; then
        swift export \
            --model "${BASE_MODEL_PATH}" \
            --adapters "${ADAPTER_PATH}" \
            --merge_lora true \
            --device_map auto
    else
        swift export \
            --adapters "${ADAPTER_PATH}" \
            --merge_lora true \
            --device_map auto
    fi
fi

# Check result
if [ -d "${ADAPTER_PATH}-merged" ]; then
    echo "============================================================"
    echo "LoRA weight merging completed!"
    echo "============================================================"
    echo "Merged model saved at: ${ADAPTER_PATH}-merged"
    echo ""
    echo "You can now use the merged model for the next training stage:"
    echo "  MODEL_PATH=\"${ADAPTER_PATH}-merged\""
    echo "============================================================"
else
    echo "============================================================"
    echo "LoRA weight merging may have failed, check errors above"
    echo ""
    echo "If VRAM is insufficient, try:"
    echo "  1. Set MERGE_MODE=\"cpu\""
    echo "  2. Add more GPUs to CUDA_VISIBLE_DEVICES"
    echo "============================================================"
fi
