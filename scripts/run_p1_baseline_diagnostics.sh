#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="${1:-reports/baselines/qwen2_5_7b_dev/v1}"
OUTPUT_DIR="${2:-reports/baselines/qwen2_5_7b_dev/v1/diagnostics}"
LOG_FILE="${3:-new_plan/logs/p1_baseline_diagnostics.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
status=0

echo "{\"event\":\"p1_baseline_diagnostics_runner_start\",\"baseline_dir\":\"${BASELINE_DIR}\",\"output_dir\":\"${OUTPUT_DIR}\",\"server_repo\":\".\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

for required_file in \
  "${BASELINE_DIR}/run_manifest.json" \
  "${BASELINE_DIR}/dev_teacher_reference.jsonl" \
  "${BASELINE_DIR}/b0_simple_predictions.jsonl" \
  "${BASELINE_DIR}/b1_structured_v3_predictions.jsonl"; do
  if [ ! -f "${required_file}" ]; then
    echo "{\"event\":\"p1_baseline_diagnostics_input\",\"status\":\"FAIL\",\"reason\":\"missing_file\",\"path\":\"${required_file}\"}" | tee -a "${LOG_FILE}"
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
    tests.test_baseline_diagnostics 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m evaluation.baseline_diagnostics \
    --baseline-dir "${BASELINE_DIR}" \
    --output-dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "p1_baseline_diagnostics_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"p1_baseline_diagnostics_log_check","status":"FAIL","reason":"missing_diagnostics_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"p1_baseline_diagnostics_runner_complete","status":"PASS"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"p1_baseline_diagnostics_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status}}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
