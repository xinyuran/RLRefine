#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_GRPO_FILE="${2:-data/canonical/keyword_v1/splits/grpo_train.jsonl}"
DRY_RUN_FILE="${3:-data/canonical/keyword_v1/dry_run/grpo_reward_dry_run.jsonl}"
OUTPUT_DIR="${4:-output/p0_ms_swift_reward_dry_run}"
LOG_FILE="${5:-new_plan/logs/p0_ms_swift_reward_dry_run.log}"
PLUGIN_FILE="rl/reward_builder.py"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE=1

mkdir -p "$(dirname "${LOG_FILE}")"
overall_status=0
missing_input=0

echo "{\"event\":\"p0_ms_swift_dry_run_start\",\"model_path\":\"${MODEL_PATH}\",\"train_grpo_file\":\"${TRAIN_GRPO_FILE}\",\"dry_run_file\":\"${DRY_RUN_FILE}\",\"output_dir\":\"${OUTPUT_DIR}\",\"cuda_visible_devices\":\"${CUDA_VISIBLE_DEVICES}\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"
python -c "import importlib.metadata as m; print('ms-swift', m.version('ms-swift')); print('torch', m.version('torch')); print('transformers', m.version('transformers'))" 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?

for input_spec in "model:${MODEL_PATH}" "train_grpo:${TRAIN_GRPO_FILE}" "plugin:${PLUGIN_FILE}"; do
  input_name="${input_spec%%:*}"
  input_path="${input_spec#*:}"
  if [ -e "${input_path}" ]; then
    echo "{\"event\":\"ms_swift_dry_run_input\",\"name\":\"${input_name}\",\"path\":\"${input_path}\",\"exists\":true}" | tee -a "${LOG_FILE}"
  else
    echo "{\"event\":\"ms_swift_dry_run_input\",\"name\":\"${input_name}\",\"path\":\"${input_path}\",\"exists\":false}" | tee -a "${LOG_FILE}"
    missing_input=1
  fi
done

if [ "${overall_status}" -eq 0 ] && [ "${missing_input}" -eq 0 ]; then
  python scripts/prepare_ms_swift_reward_dry_run.py \
    --input-file "${TRAIN_GRPO_FILE}" \
    --output-file "${DRY_RUN_FILE}" \
    --sample-count 2 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?
else
  overall_status=1
fi

if [ "${overall_status}" -eq 0 ]; then
  python -m unittest -v \
    tests.test_reward_builder \
    tests.test_data_canonicalization \
    tests.test_data_split \
    tests.test_annotation_packet \
    tests.test_annotation_validation \
    tests.test_gold_freeze 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?
fi

if [ "${overall_status}" -eq 0 ]; then
  swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --dataset "${DRY_RUN_FILE}" \
    --external_plugins "${PLUGIN_FILE}" \
    --reward_funcs schema_based_reward \
    --train_type lora \
    --lora_rank 4 \
    --lora_alpha 8 \
    --torch_dtype bfloat16 \
    --max_length 2048 \
    --max_completion_length 256 \
    --max_steps 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_generations 2 \
    --beta 0.0 \
    --temperature 0.7 \
    --use_vllm false \
    --split_dataset_ratio 0 \
    --dataset_num_proc 1 \
    --dataloader_num_workers 0 \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    --output_dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || overall_status=$?
fi

if [ "${overall_status}" -eq 0 ] && ! grep -q 'REWARD_INPUT_CONTRACT' "${LOG_FILE}"; then
  echo '{"event":"p0_ms_swift_dry_run_contract_check","status":"FAIL","reason":"missing_reward_input_contract_log"}' | tee -a "${LOG_FILE}"
  overall_status=1
fi

if [ "${overall_status}" -eq 0 ] && ! grep -q '"prompt_source": "messages_kwarg"' "${LOG_FILE}"; then
  echo '{"event":"p0_ms_swift_dry_run_contract_check","status":"FAIL","reason":"source_messages_not_connected"}' | tee -a "${LOG_FILE}"
  overall_status=1
fi

if [ "${overall_status}" -eq 0 ]; then
  echo '{"event":"p0_ms_swift_dry_run_contract_check","status":"PASS","prompt_source":"messages_kwarg"}' | tee -a "${LOG_FILE}"
fi

if [ "${overall_status}" -eq 0 ]; then
  echo '{"event":"p0_ms_swift_dry_run_complete","status":"PASS","max_steps":1}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p0_ms_swift_dry_run_complete\",\"status\":\"FAIL\",\"exit_code\":${overall_status}}" | tee -a "${LOG_FILE}"
fi

exit "${overall_status}"
