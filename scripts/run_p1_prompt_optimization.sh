#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
DEV_FILE="${2:-data/canonical/keyword_v1/splits/grpo_dev.jsonl}"
BASELINE_DIR="${3:-reports/baselines/qwen2_5_7b_dev/v1}"
CANDIDATE_DIR="${4:-reports/baselines/qwen2_5_7b_dev/v2}"
LOG_FILE="${5:-new_plan/logs/p1_prompt_optimization.log}"
CANDIDATE_VARIANT="${CANDIDATE_VARIANT:-b2_compact_v1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$(dirname "${LOG_FILE}")"
status=0

echo "{\"event\":\"p1_prompt_optimization_runner_start\",\"model_path\":\"${MODEL_PATH}\",\"dev_file\":\"${DEV_FILE}\",\"baseline_dir\":\"${BASELINE_DIR}\",\"candidate_dir\":\"${CANDIDATE_DIR}\",\"candidate_variant\":\"${CANDIDATE_VARIANT}\",\"cuda_visible_devices\":\"${CUDA_VISIBLE_DEVICES}\",\"server_repo\":\".\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

for required_path in \
  "${MODEL_PATH}" \
  "${DEV_FILE}" \
  "${BASELINE_DIR}/run_manifest.json" \
  "${BASELINE_DIR}/dev_teacher_reference.jsonl" \
  "${BASELINE_DIR}/b1_structured_v3_predictions.jsonl"; do
  if [ ! -e "${required_path}" ]; then
    echo "{\"event\":\"p1_prompt_optimization_input\",\"status\":\"FAIL\",\"reason\":\"missing_path\",\"path\":\"${required_path}\"}" | tee -a "${LOG_FILE}"
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
  python -c 'import json, torch, transformers; print(json.dumps({"event":"p1_prompt_optimization_environment","torch":torch.__version__,"transformers":transformers.__version__,"cuda_available":torch.cuda.is_available(),"cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, ensure_ascii=False))' 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.baseline_inference \
    --model-path "${MODEL_PATH}" \
    --dev-file "${DEV_FILE}" \
    --output-dir "${CANDIDATE_DIR}" \
    --variants "${CANDIDATE_VARIANT}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.prompt_optimization_diagnostics \
    --baseline-dir "${BASELINE_DIR}" \
    --candidate-dir "${CANDIDATE_DIR}" \
    --candidate-variant "${CANDIDATE_VARIANT}" \
    --output-dir "${CANDIDATE_DIR}/diagnostics" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "p1_prompt_optimization_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"p1_prompt_optimization_log_check","status":"FAIL","reason":"missing_optimization_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p1_prompt_optimization_runner_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p1_prompt_optimization_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
