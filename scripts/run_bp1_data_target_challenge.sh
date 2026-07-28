#!/usr/bin/env bash
set -euo pipefail

TRAIN_FILE="${1:-data/canonical/keyword_v1/splits/sft_train.jsonl}"
DEV_FILE="${2:-data/canonical/keyword_v1/splits/sft_dev.jsonl}"
QUARANTINE_FILE="${3:-data/canonical/keyword_v1/sft_quarantine.jsonl}"
GOLD_FILE="${4:-data/canonical/keyword_v1/gold/v1/gold_test.jsonl}"
OUTPUT_DIR="${5:-data/canonical/keyword_v2}"
REPORT_DIR="${6:-reports/bp1_keyword_v2}"
LOG_FILE="${7:-new_plan/logs/bp1_data_target_challenge.log}"

status=0
mkdir -p "$(dirname "${LOG_FILE}")" "${OUTPUT_DIR}" "${REPORT_DIR}"

echo "{\"event\":\"bp1_runner_start\",\"work_package\":\"BP1_DATA_TARGET_CHALLENGE\",\"train_file\":\"${TRAIN_FILE}\",\"dev_file\":\"${DEV_FILE}\",\"quarantine_file\":\"${QUARANTINE_FILE}\",\"gold_file\":\"${GOLD_FILE}\",\"gold_usage\":\"hash_and_overlap_only\",\"output_dir\":\"${OUTPUT_DIR}\",\"report_dir\":\"${REPORT_DIR}\"}" | tee "${LOG_FILE}"
python --version 2>&1 | tee -a "${LOG_FILE}"

for required_path in "${TRAIN_FILE}" "${DEV_FILE}" "${QUARANTINE_FILE}" "${GOLD_FILE}"; do
  if [ ! -f "${required_path}" ]; then
    echo "{\"event\":\"bp1_input_check\",\"status\":\"FAIL\",\"reason\":\"missing_file\",\"path\":\"${required_path}\"}" | tee -a "${LOG_FILE}"
    status=1
  fi
done

if [ "${status}" -eq 0 ]; then
  python -m unittest discover -v tests 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.build_bp1_assets \
    --train-file "${TRAIN_FILE}" \
    --dev-file "${DEV_FILE}" \
    --quarantine-file "${QUARANTINE_FILE}" \
    --gold-file "${GOLD_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --report-dir "${REPORT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ]; then
  python -m scripts.validate_bp1_assets \
    --output-dir "${OUTPUT_DIR}" \
    --report-dir "${REPORT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || status=$?
fi

if [ "${status}" -eq 0 ] && ! grep -q '"event": "bp1_gate_complete".*"status": "PASS"' "${LOG_FILE}"; then
  echo '{"event":"bp1_log_check","status":"FAIL","reason":"missing_gate_pass"}' | tee -a "${LOG_FILE}"
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo '{"event":"bp1_runner_complete","status":"PASS","next_work_package":"BP2_POST_TRAINING_PIPELINE"}' | tee -a "${LOG_FILE}"
else
  echo "{\"event\":\"bp1_runner_complete\",\"status\":\"FAIL\",\"exit_code\":${status},\"next_work_package\":\"BP1_REPAIR_FROM_FAILED_CHECKS\"}" | tee -a "${LOG_FILE}"
fi

exit "${status}"
