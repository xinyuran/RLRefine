#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
GOLD_FILE="${2:-data/canonical/keyword_v1/gold/v1/gold_test.jsonl}"
OUTPUT_DIR="${3:-reports/baselines/qwen2_5_7b_human_gold/v1}"
LOG_FILE="${4:-new_plan/logs/p1_final_base_gold.log}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$(dirname "${LOG_FILE}")"
status=0

echo "{\"event\":\"p1_final_base_gold_runner_start\",\"model_path\":\"${MODEL_PATH}\",\"gold_file\":\"${GOLD_FILE}\",\"output_dir\":\"${OUTPUT_DIR}\",\"prompt_variant\":\"b1_structured_v3\",\"cuda_visible_devices\":\"${CUDA_VISIBLE_DEVICES}\",\"server_repo\":\".\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

for required_path in "${MODEL_PATH}" "${GOLD_FILE}"; do
  if [ ! -e "${required_path}" ]; then
    echo "{\"event\":\"p1_final_base_gold_input\",\"status\":\"FAIL\",\"reason\":\"missing_path\",\"path\":\"${required_path}\"}" | tee -a "${LOG_FILE}"
    status=1
  fi
done

if [ "${status}" -eq 0 ]; then
  python -m unittest -v \
    tests.test_reward_builder \
    tests.test_data_canonicalization \
    tests.test_data_split \
    tests.test_annotation_packet \
    tests.test_annotation_validation \
    tests.test_gold_freeze \
    tests.test_keyword_schema_contract \
    tests.test_processor_contract \
    tests.test_keyword_evaluator \
    tests.test_baseline_inference \
    tests.test_baseline_diagnostics \
    tests.test_prompt_optimization_diagnostics 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -c 'import json, torch, transformers; print(json.dumps({"event":"p1_final_base_gold_environment","torch":torch.__version__,"transformers":transformers.__version__,"cuda_available":torch.cuda.is_available(),"cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, ensure_ascii=False))' 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.baseline_inference \
    --model-path "${MODEL_PATH}" \
    --gold-file "${GOLD_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --variants b1_structured_v3 \
    --max-input-tokens 4096 \
    --max-new-tokens 512 \
    --seed 42 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "p1_final_base_gold_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"p1_final_base_gold_log_check","status":"FAIL","reason":"missing_final_gold_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p1_final_base_gold_runner_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p1_final_base_gold_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
