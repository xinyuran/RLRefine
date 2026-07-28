#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-reports/training/p1_sft_lora_v1/v1-20260722-172249/checkpoint-390}"
MODEL_PATH="${2:-Qwen/Qwen2.5-7B-Instruct}"
DEV_FILE="${3:-data/canonical/keyword_v1/splits/grpo_dev.jsonl}"
BASELINE_DIR="${4:-reports/baselines/qwen2_5_7b_dev/v1}"
CANDIDATE_DIR="${5:-reports/training/p1_sft_lora_v1/v1-20260722-172249/teacher_dev_checkpoint_390}"
LOG_FILE="${6:-new_plan/logs/p1_sft_teacher_dev.log}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
mkdir -p "$(dirname "${LOG_FILE}")"
status=0

echo "{\"event\":\"p1_sft_teacher_dev_runner_start\",\"checkpoint\":\"${CHECKPOINT}\",\"baseline_dir\":\"${BASELINE_DIR}\",\"candidate_dir\":\"${CANDIDATE_DIR}\",\"cuda_visible_devices\":\"${CUDA_VISIBLE_DEVICES}\"}" | tee "${LOG_FILE}"

for required_path in "${CHECKPOINT}" "${MODEL_PATH}"; do
  if [ ! -d "${required_path}" ]; then
    echo "{\"event\":\"p1_sft_teacher_dev_input\",\"status\":\"FAIL\",\"reason\":\"missing_directory\",\"path\":\"${required_path}\"}" | tee -a "${LOG_FILE}"
    status=1
  fi
done
for required_file in \
  "${DEV_FILE}" \
  "${BASELINE_DIR}/dev_teacher_reference.jsonl" \
  "${BASELINE_DIR}/b1_structured_v3_predictions.jsonl"; do
  if [ ! -f "${required_file}" ]; then
    echo "{\"event\":\"p1_sft_teacher_dev_input\",\"status\":\"FAIL\",\"reason\":\"missing_file\",\"path\":\"${required_file}\"}" | tee -a "${LOG_FILE}"
    status=1
  fi
done

if [ "${status}" -eq 0 ]; then
  python -m unittest -v \
    tests.test_keyword_evaluator \
    tests.test_baseline_inference \
    tests.test_baseline_diagnostics \
    tests.test_sft_acceptance 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.baseline_inference \
    --model-path "${MODEL_PATH}" \
    --model-id "Qwen2.5-7B-Instruct-sft-lora-v1-checkpoint-390" \
    --adapter-path "${CHECKPOINT}" \
    --dev-file "${DEV_FILE}" \
    --output-dir "${CANDIDATE_DIR}" \
    --variants b1_structured_v3 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.sft_acceptance \
    --baseline-dir "${BASELINE_DIR}" \
    --candidate-dir "${CANDIDATE_DIR}" \
    --output-dir "${CANDIDATE_DIR}/acceptance" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "p1_sft_teacher_dev_acceptance_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"p1_sft_teacher_dev_log_check","status":"FAIL","reason":"missing_acceptance_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p1_sft_teacher_dev_runner_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p1_sft_teacher_dev_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
